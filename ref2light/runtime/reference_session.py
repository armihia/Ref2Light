from pathlib import Path

import torch
import torch.nn.functional as F

from ref2light.common.controls import NEUTRAL, POSE_KEYS, validate_pose
from ref2light.common.io import read_json, write_json
from ref2light.models.face_adapter import amp
from ref2light.local.output_regions import detect_output_regions, boxes_from_detection, validate_output_boxes, BOX_SPACE
from ref2light.common.io import timestamp
from ref2light.reference.data import ref_tensor
from ref2light.reference.model import encode
from ref2light.reference.reference import prepare_inference

from ref2light.local.local import LOCAL_FORMAT, LocalControls
from ref2light import ROOT
from ref2light.local.runtime import canonical_pose, pixel_box, soft_mask, validate_eye_mode
from ref2light.local.roi_tracking import prepare_tracking, track_boxes
from ref2light.local.masks import patch_mask
from ref2light.local.region_config import build_mask_config, configure_boxes
from ref2light.local.inference_speed import branch_is_inactive
from ref2light.local.local_graph import LocalGraphRunner
from ref2light.local.body_graph import FrozenBodyGraph


class ReferenceLocalizationError(ValueError):
    """Carry ordinary-view previews for a candidate reference without committing failed detections."""
    def __init__(self, request):
        self.request = request
        super().__init__('Localization failed in ordinary output views. Mark facial regions in the Tk preview or load them with --output-boxes. ' + request['error'])


def _read_manual_boxes(path, reference, align):
    value = read_json(path)
    if 'alignment_enabled' in value and value['alignment_enabled'] != bool(align):
        raise ValueError('The manual output boxes use a different alignment setting; reannotate or restore that setting')
    return validate_output_boxes(value, reference)


def _localization_request(anchors, reference, align, destination, error, *, reason='detection_failed'):
    """Export unpadded ordinary views so manual annotations use the correct image coordinates."""
    from PIL import Image
    directory = Path(destination) / 'manual_preview'
    directory.mkdir(parents=True, exist_ok=True)
    count, height, width = anchors.shape[1], anchors.shape[-2], anchors.shape[-1]
    views = []
    for index in sorted(set((0, count // 2, count - 1))):
        pixels = (anchors[0, index].detach().float().clamp(0, 1).mul(255).round()
                  .to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        path = directory / f'ordinary_{index:02d}.png'
        Image.fromarray(pixels).save(path)
        views.append(dict(index=index, path=str(path.resolve())))
    request = dict(request_id=Path(destination).name, reference=str(Path(reference).resolve()),
                   align=bool(align), view_count=count, width=width, height=height,
                   views=views, destination=str(Path(destination).resolve()), error=str(error), reason=reason)
    write_json(directory / 'request.json', request)
    return request


def resolve_model(run, choice='best', model=None):
    if model:
        path = Path(model).resolve()
    else:
        if choice not in ('best', 'last'):
            raise ValueError('The local model selection must be best or last')
        path = Path(run).resolve() / ('best_local.pt' if choice == 'best' else 'last_local.pt')
    if not path.is_file():
        raise FileNotFoundError('The selected local model does not exist: ' + str(path))
    return path


class Session:
    def __init__(self, model, reference, device, landmark_model, folder, align=False, layout=None,
                 *, output_boxes=None, roi_tracking='source_uv', local_graphs=True, body_graphs=True,
                 defer_reference=False, detector_recovery=True):
        from ref2light.model_io import read_model, build_model
        self.device = torch.device(device)
        self.model_path = self.base_path = Path(model).resolve()
        package = read_model(self.model_path)
        networks = build_model(package,self.device)
        self.base,self.local,self.rad_head,self.brow_head = (networks[k] for k in ('base','local','rad','brows'))
        self.base_payload = package['base']
        self.stats = self.base_payload['stats']
        self.payload = dict(format=package['format'],step=self.base_payload.get('step'))
        config = package['runtime']
        self.eye_mode = validate_eye_mode(config['eye_mode'])
        self.detector_policy = config['detector_policy']
        self.mask_config = build_mask_config(config['mask'])
        self.region_config = dict(config['regions'])
        self.resolved_runtime_config = dict(config)
        self.runtime_config_source = 'complete_model'
        self._pose_mean = torch.tensor(self.stats['mean'],device=self.device,dtype=torch.float32)
        self._pose_std = torch.tensor(self.stats['std'],device=self.device,dtype=torch.float32)
        self.body_graphs_requested = bool(body_graphs)
        self.body_graph_runner = FrozenBodyGraph(self.base) if body_graphs and self.device.type=='cuda' else None
        self.local_graphs_requested = bool(local_graphs)
        self.local_graph_runner = LocalGraphRunner(self.local) if local_graphs and self.device.type=='cuda' else None
        self._inactive_skip_enabled,self._batch_color_enabled = True,False
        self.size = tuple(int(self.stats['calibration'][key]) for key in ('native_width','native_height'))
        self.landmark_model,self.folder,self.align = landmark_model,Path(folder),bool(align)
        self.detector_recovery = bool(detector_recovery)
        self._initial_reference = str(Path(reference).resolve())
        self.output_boxes_file = str(Path(output_boxes).resolve()) if output_boxes else None
        if roi_tracking not in ('source_uv','neutral_fixed'):raise ValueError('Unknown region-tracking mode')
        self.roi_tracking,self.mode = roi_tracking,'trained'
        if not defer_reference:self.set_reference(reference,layout)

    def _clear_cache(self):
        self._context_key = self._context_value = None
        self._branch_cache = {}
        self._frame_key = self._frame_value = None
        self._last_skipped_branches = []

    def _normalized_pose(self, raw):
        return (torch.tensor(raw, device=self.device, dtype=torch.float32)[None] - self._pose_mean) / self._pose_std

    def _anchors(self, ref, features, raw):
        normalized = self._normalized_pose(raw)
        runner = self.body_graph_runner
        # Run uncommitted candidate references without overwriting the current graph input buffers.
        if runner is not None and ref is getattr(self, 'ref', None) and features is self.features:
            return runner(normalized)
        with amp(self.device, self.device.type == 'cuda'):
            return self.base.render_anchors_from_features(ref, features, normalized)

    @torch.no_grad()
    def set_reference(self, path, layout=None, align=None, *, output_boxes=None):
        chosen_align = self.align if align is None else bool(align)
        destination = self.folder / ('reference_' + timestamp())
        image, report = prepare_inference(path, self.base_payload['alignment_points'], self.landmark_model,
                                          destination, chosen_align, layout)
        ref = ref_tensor(image).to(self.device).float()[None] / 255
        features = encode(self.base, ref)
        neutral_raw = [NEUTRAL[key] for key in POSE_KEYS]
        # Register regions on neutral views with open eyes and reuse the same source-UV calibration for later eye states.
        neutral_raw[6] = 1.
        neutral = canonical_pose(neutral_raw, getattr(self, 'eye_mode', 'local'))
        anchors, aux = self._anchors(ref, features, neutral)
        chosen_manual = str(Path(output_boxes).resolve()) if output_boxes is not None else self.output_boxes_file
        bound_reference = getattr(self, 'reference', self._initial_reference)
        if output_boxes is None and (Path(bound_reference).resolve() != Path(path).resolve()
                                     or chosen_align != self.align):
            chosen_manual = None
        if chosen_manual:
            try:
                raw_boxes = _read_manual_boxes(chosen_manual, path, chosen_align)
            except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
                request = _localization_request(anchors, path, chosen_align, destination, exc)
                raise ReferenceLocalizationError(request) from exc
            region_report = dict(source='manual_output_boxes', file=chosen_manual)
        else:
            try:
                detected, region_report = detect_output_regions(anchors, self.landmark_model,
                                                                 destination / 'output_detection', policy=self.detector_policy,
                                                                 recovery=self.detector_recovery)
            except (ValueError, FileNotFoundError) as exc:
                request = _localization_request(anchors, path, chosen_align, destination, exc)
                raise ReferenceLocalizationError(request) from exc
            raw_boxes = boxes_from_detection(detected)
        boxes = configure_boxes(raw_boxes, **self.region_config)
        if boxes.get('view_boxes') is not None and len(boxes['view_boxes']) != anchors.shape[1]:
            raise ValueError('The number of per-view facial boxes does not match the base anchor count')
        tracking, tracking_report = prepare_tracking(aux['flow'], boxes) if self.roi_tracking == 'source_uv' else ({}, {})
        saved_boxes = dict(coordinate_space=BOX_SPACE, reference=str(Path(path).resolve()), boxes=raw_boxes,
                           alignment_enabled=chosen_align,
                           effective_boxes=boxes, region_config=self.region_config,
                           tracking=self.roi_tracking, large_pose_tracking_validated=False)
        write_json(destination / 'output_boxes.json', saved_boxes)
        report.update(output_regions=region_report, output_boxes=saved_boxes,
                      output_boxes_file=str(destination / 'output_boxes.json'), source_uv_tracking=tracking_report)
        write_json(destination / 'source_uv_tracking.json', tracking_report)
        prepared_context = self._make_context(anchors, aux, features, boxes)
        neutral_flow = aux['flow'].clone()
        if self.body_graph_runner is not None:
            # Update graph inputs only after detection, regions, context, and files are ready.
            self.body_graph_runner.prepare(ref, features, self._normalized_pose(neutral))
        # Commit the reference only after its image, features, and regions are ready.
        self.reference, self.ref, self.features = str(Path(path).resolve()), ref, features
        self.reference_report, self.align, self.boxes = report, chosen_align, boxes
        self.raw_boxes = raw_boxes
        self.output_boxes_file = chosen_manual
        self.tracking, self.tracking_report = tracking, tracking_report
        self.current_boxes, self.current_tracking_report = boxes, dict(mode='neutral_detection')
        self._neutral_flow = neutral_flow
        self._clear_cache()
        self._context_value = prepared_context
        self._context_key = tuple(neutral)

    def _make_context(self, anchors, aux, features, boxes=None):
        boxes = self.boxes if boxes is None else boxes
        if self.body_graph_runner is not None:
            # Keep persistent context separate because base mode and body-pose changes overwrite graph outputs.
            anchors = anchors.clone()
        result = dict(anchors=anchors, identity=features[-1].float().mean((2, 3)), patches={})
        for group, size in (('eyes', (96, 192)), ('mouth', (96, 128))):
            box = pixel_box(boxes[group], anchors.shape[-1], anchors.shape[-2])
            x0, y0, x1, y1 = box
            before = anchors[0, :, :, y0:y1, x0:x1].float().contiguous()
            warped = aux['warped'][0, :, :, y0:y1, x0:x1].float().contiguous()
            if boxes.get('view_boxes') is not None:
                mask = patch_mask(group, box, anchors.shape[-2:], size, boxes['view_boxes'], self.device, **self.mask_config)
            else:
                mask = soft_mask(*size, self.device).expand(before.shape[0], -1, -1, -1)
            result['patches'][group] = dict(box=box, before=before,
                canonical=F.interpolate(before, size, mode='bilinear', align_corners=False),
                reference=F.interpolate(warped, size, mode='bilinear', align_corners=False),
                mask=mask)
        return result

    @torch.no_grad()
    def manual_reference_request(self):
        """Prepare ordinary neutral views for region editing without changing the current pose or image cache."""
        if not hasattr(self, 'reference'):
            raise ValueError('No reference image has been loaded; select one first')
        neutral_raw = [NEUTRAL[key] for key in POSE_KEYS]
        neutral_raw[6] = 1.
        neutral = canonical_pose(neutral_raw, self.eye_mode)
        anchors, _ = self._anchors(self.ref, self.features, neutral)
        destination = self.folder / ('manual_' + timestamp())
        return _localization_request(anchors, self.reference, self.align, destination, 'Manually edit regions for the current reference',
                                     reason='manual_adjustment')

    def set_mode(self, mode):
        if mode not in ('trained', 'base'):
            raise ValueError('mode must be trained or base')
        self.mode = mode
        self._frame_key = self._frame_value = None

    def set_output_boxes(self, path):
        raw_boxes = _read_manual_boxes(path, self.reference, self.align)
        boxes = configure_boxes(raw_boxes, **self.region_config)
        if boxes.get('view_boxes') is not None and len(boxes['view_boxes']) != self._neutral_flow.shape[1]:
            raise ValueError('The number of per-view facial boxes does not match the base anchor count')
        self.output_boxes_file = str(Path(path).resolve())
        self.boxes = boxes
        self.raw_boxes = raw_boxes
        self.reference_report = dict(self.reference_report, output_boxes=dict(
            coordinate_space=BOX_SPACE, reference=self.reference, boxes=raw_boxes, effective_boxes=boxes,
            alignment_enabled=self.align, region_config=self.region_config, tracking=self.roi_tracking),
            output_boxes_file=self.output_boxes_file)
        self.tracking, self.tracking_report = prepare_tracking(self._neutral_flow, boxes) if self.roi_tracking == 'source_uv' else ({}, {})
        self.reference_report['source_uv_tracking'] = self.tracking_report
        self.current_boxes, self.current_tracking_report = boxes, dict(mode='manual_neutral')
        self._clear_cache()


