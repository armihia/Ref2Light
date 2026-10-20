import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ref2light.common.landmarks import EYE_A, EYE_B, LIPS, region_box
from ref2light.common.render.output_regions import interpolate_boxes


BOX_SPACE = 'ordinary_anchor_normalized_xy'
DETECTOR_POLICIES = ('whole_first', 'upper_first')
# Contour vertices from MediaPipe FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL.
# Include eye and lip points when checking whether an upper-image crop truncates the face.
FACE_OVAL = (10,338,297,332,284,251,389,356,454,323,361,288,397,365,
             379,378,400,377,152,148,176,149,150,136,172,58,132,93,
             234,127,162,21,54,103,67,109)
CROP_COMPLETENESS_POINTS = tuple(sorted(set(FACE_OVAL+EYE_A+EYE_B+LIPS)))


def recovery_crops(image_size):
    """Retry only after all regular detections fail; square crops preserve image proportions."""
    width, height = image_size
    if min(width, height) < 1:
        raise ValueError('Ordinary-view dimensions must be positive')
    seen = set()
    # Search the central upper body before trying left- and right-shifted crops.
    for centre_x in (.5, .35, .65):
        for scale in (.4, .3, .24, .6):
            side = max(1, round(min(width, height)*scale))
            for centre_y in (.2, .3, .4):
                left = max(0, min(width-side, round(width*centre_x-side/2)))
                top = max(0, min(height-side, round(height*centre_y-side/2)))
                box = (left, top, left+side, top+side)
                if box not in seen:
                    seen.add(box)
                    yield f'upper_recovery_{len(seen):02d}', box


def validate_output_boxes(value, reference=None):
    if value.get('coordinate_space') != BOX_SPACE:
        raise ValueError('Output-box coordinate_space must be ' + BOX_SPACE)
    if reference and value.get('reference') and Path(value['reference']).resolve() != Path(reference).resolve():
        raise ValueError('The manual output boxes belong to another reference image; annotate the new image')
    result = {}
    for group in ('eyes', 'mouth'):
        box = torch.as_tensor(value['boxes'][group], dtype=torch.float64)
        if (box.shape != (4,) or not bool(torch.isfinite(box).all()) or bool((box < 0).any())
                or bool((box > 1).any()) or not bool((box[2:] > box[:2]).all())):
            raise ValueError(group + ' boxes must be normalized finite rectangles with positive area')
        result[group] = box.tolist()
    view_boxes = value.get('view_boxes', value['boxes'].get('view_boxes'))
    if view_boxes is not None:
        views = torch.as_tensor(view_boxes, dtype=torch.float64)
        if (views.ndim != 3 or views.shape[0] < 1 or views.shape[1:] != (3, 4)
                or not bool(torch.isfinite(views).all()) or bool((views < 0).any())
                or bool((views > 1).any()) or not bool((views[..., 2:] > views[..., :2]).all())):
            raise ValueError('Per-view regions must be valid normalized eye and mouth boxes with shape K x 3 x 4')
        result['view_boxes'] = views.tolist()
    return result


def boxes_from_detection(detected):
    """Combine three detected views and interpolated boxes using the same expansion scales in training and inference."""
    value = detected.detach().float().cpu()
    if value.ndim != 3 or value.shape[1:] != (3, 4):
        raise ValueError('Expected ordinary-view eye and mouth boxes with shape K x 3 x 4')
    result = {}
    for group, region, scale in (('eyes', value[:, :2].reshape(-1, 4), (1.8, 1.8)),
                                  ('mouth', value[:, 2], (2.2, 3.0))):
        low, high = region[:, :2].amin(0), region[:, 2:].amax(0)
        centre, radius = (low + high) / 2, (high - low) / 2 * torch.tensor(scale)
        result[group] = torch.cat((centre-radius, centre+radius)).clamp(0, 1).tolist()
    return validate_output_boxes(dict(coordinate_space=BOX_SPACE, boxes=result, view_boxes=value.tolist()))


class OutputRegionDetector:
    """Reuse one MediaPipe IMAGE CPU detector across reference images."""
    def __init__(self, model, policy='whole_first'):
        if policy not in DETECTOR_POLICIES:
            raise ValueError('The detection policy must be whole_first or upper_first')
        import mediapipe as mp
        self.policy = policy
        self.model = Path(model).resolve()
        if not self.model.is_file():
            raise FileNotFoundError('Missing face landmark model: ' + str(self.model))
        self.mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1)
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()

    def _square(self, image):
        width, height = image.size
        attempts = []
        crops = [('whole', (0, 0, width, height)), ('upper_centre', (width//4, 0, 3*width//4, height//2))]
        policy = getattr(self, 'policy', 'whole_first')
        if policy == 'upper_first':
            crops.reverse()
        for name, (x0, y0, x1, y1) in crops:
            array = np.asarray(image.crop((x0, y0, x1, y1)).resize((512, 512))).copy()
            result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=array))
            attempt = dict(crop=name, faces=len(result.face_landmarks), accepted=False)
            attempts.append(attempt)
            if not result.face_landmarks:
                attempt['rejection_reason'] = 'no_face_detected'
                continue
            local_points = np.asarray([(p.x, p.y) for p in result.face_landmarks[0]], np.float32)
            if policy == 'upper_first' and name == 'upper_centre':
                if len(local_points) <= max(CROP_COMPLETENESS_POINTS):
                    attempt['rejection_reason'] = 'incomplete_landmark_output'
                    continue
                boundary = local_points[list(CROP_COMPLETENESS_POINTS)]
                invalid = (~np.isfinite(boundary).all(1) | (boundary < 0).any(1) | (boundary > 1).any(1))
                attempt['completeness_check'] = 'face_oval_eyes_lips_inside_crop_0_1'
                if bool(invalid.any()):
                    attempt['rejection_reason'] = 'key_face_landmarks_outside_crop'
                    attempt['outside_landmark_indices'] = [CROP_COMPLETENESS_POINTS[i] for i in np.flatnonzero(invalid)]
                    continue
                attempt['key_landmark_bounds_in_crop'] = np.r_[boundary.min(0), boundary.max(0)].tolist()
            points = np.array([[(point.x*(x1-x0)+x0)/width, (point.y*(y1-y0)+y0)/height]
                               for point in result.face_landmarks[0]], dtype=np.float32)
            eyes = [region_box(points[list(ids)], 1.5, .45) for ids in (EYE_A, EYE_B)]
            eyes.sort(key=lambda box: (box[0]+box[2])/2)
            attempt['accepted'] = True
            return eyes + [region_box(points[list(LIPS)], 1.5, .35)], attempts
        error = ValueError('No complete valid face was found in the ordinary output view: ' + json.dumps(attempts, ensure_ascii=False))
        error.attempts = attempts
        raise error

    def _recover_square(self, image, content_box=None):
        """Retry cropped detection on CPU and map coordinates back to the original ordinary view."""
        width, height = image.size
        bounds = np.asarray(content_box if content_box is not None else (0,0,width,height), np.float32)
        if (bounds.shape != (4,) or not np.isfinite(bounds).all()
                or np.any(bounds < 0) or np.any(bounds > [width,height,width,height])
                or np.any(bounds[2:] <= bounds[:2])):
            raise ValueError('Invalid visible-content bounds in the ordinary view')
        attempts = []
        for name, (x0,y0,x1,y1) in recovery_crops(image.size):
            # Square crops preserve character proportions during resizing.
            array = np.asarray(image.crop((x0,y0,x1,y1)).resize((512,512))).copy()
            result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=array))
            attempt = dict(crop=name, crop_pixels=[x0,y0,x1,y1], faces=len(result.face_landmarks),
                           accepted=False, resize='square_uniform_scale_to_512')
            attempts.append(attempt)
            if not result.face_landmarks:
                attempt['rejection_reason'] = 'no_face_detected'
                continue
            local = np.asarray([(point.x,point.y) for point in result.face_landmarks[0]], np.float32)
            if len(local) <= max(CROP_COMPLETENESS_POINTS) or not np.isfinite(local).all():
                attempt['rejection_reason'] = 'incomplete_or_nonfinite_landmark_output'
                continue
            boundary = local[list(CROP_COMPLETENESS_POINTS)]
            # Use a two-pixel margin to reject landmarks touching the crop boundary rather than clipping them.
            margin = 2/512
            invalid = (boundary < margin).any(1) | (boundary > 1-margin).any(1)
            attempt['completeness_check'] = 'face_oval_eyes_lips_inside_crop_with_2px_margin'
            if bool(invalid.any()):
                attempt['rejection_reason'] = 'key_face_landmarks_touch_or_cross_crop'
                attempt['outside_landmark_indices'] = [CROP_COMPLETENESS_POINTS[i] for i in np.flatnonzero(invalid)]
                continue
            attempt['key_landmark_bounds_in_crop'] = np.r_[boundary.min(0),boundary.max(0)].tolist()
            boundary_pixels = boundary*np.array([x1-x0,y1-y0])+[x0,y0]
            if np.any(boundary_pixels < bounds[:2]) or np.any(boundary_pixels > bounds[2:]):
                attempt['rejection_reason'] = 'key_face_landmarks_outside_actual_ordinary_view'
                continue
            points = (local*np.array([x1-x0,y1-y0])+[x0,y0])/[width,height]
            eyes = [region_box(points[list(ids)],1.5,.45) for ids in (EYE_A,EYE_B)]
            eyes.sort(key=lambda box:(box[0]+box[2])/2)
            regions = np.asarray(eyes+[region_box(points[list(LIPS)],1.5,.35)],np.float32)
            pixel_regions = regions*[width,height,width,height]
            if (not np.isfinite(regions).all() or np.any(regions[:,2:] <= regions[:,:2])
                    or np.any(pixel_regions[:,:2] < bounds[:2]) or np.any(pixel_regions[:,2:] > bounds[2:])):
                attempt['rejection_reason'] = 'regions_outside_actual_ordinary_view'
                continue
            attempt['accepted'] = True
            return regions.tolist(), attempts
        error = ValueError('Cropped retries found no complete valid face in the ordinary output view')
        error.attempts = attempts
        raise error

    def detect(self, anchors, destination=None, *, recovery=False):
        if type(recovery) is not bool:
            raise ValueError('recovery must be a boolean')
        if anchors.ndim != 5 or anchors.shape[0] != 1 or anchors.shape[2] != 3:
            raise ValueError('Expected ordinary anchor views with shape 1 x K x 3 x H x W')
        _, count, _, height, width = anchors.shape
        side = max(height, width)
        left, top = (side-width)//2, (side-height)//2
        destination = Path(destination) if destination is not None else None
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=False)
        detected, attempts, canvases = {}, [], {}
        for index in sorted(set((0, count//2, count-1))):
            array = anchors[0, index].detach().float().clamp(0, 1).mul(255).add(.5).byte().permute(1, 2, 0).cpu().numpy()
            canvas = Image.new('RGB', (side, side))
            canvas.paste(Image.fromarray(array), (left, top))
            if recovery:
                canvases[index] = canvas
            if destination is not None:
                canvas.save(destination/f'ordinary_view_{index:02d}.png')
            try:
                regions, details = self._square(canvas)
                boxes = np.asarray(regions, np.float32)*np.array([side, side, side, side], np.float32)
                boxes -= np.array([left, top, left, top], np.float32)
                boxes /= np.array([width, height, width, height], np.float32)
                if (not np.isfinite(boxes).all() or np.any(boxes < 0) or np.any(boxes > 1)
                        or np.any(boxes[:, 2:] <= boxes[:, :2])):
                    raise ValueError('A detected region lies outside the ordinary view')
                detected[index] = boxes.tolist()
                attempts.append(dict(view=index, detected=True, boxes=boxes.tolist(), attempts=details))
            except ValueError as exc:
                attempts.append(dict(view=index, detected=False, error=str(exc), attempts=getattr(exc, 'attempts', [])))
        report = dict(source='generated_pre_expression_ordinary_views', selected_views=attempts,
                      detector_policy=getattr(self, 'policy', 'whole_first'),
                      model=str(self.model), images_are_predictions_not_GT=True,
                      uses_target_Y=False, uses_character_geometry=False)
        if not detected and recovery:
            # Keep any successful regular detection; run recovery only when all three regular views fail.
            report['recovery'] = dict(enabled=True, attempted=True,
                trigger='all_regular_selected_views_failed',strategy='upper_square_multiscale_CPU',
                changes_reference_input=False,uses_target_Y=False,detected_views=[])
            by_view = {item['view']:item for item in attempts}
            for index in sorted(canvases,key=lambda value:(abs(value-count//2),value)):
                record = by_view[index]
                record['regular_detected'] = False
                record['regular_error'] = record.pop('error','')
                try:
                    regions,details = self._recover_square(canvases[index],(left,top,left+width,top+height))
                    boxes = np.asarray(regions,np.float32)*[side,side,side,side]
                    boxes = (boxes-[left,top,left,top])/[width,height,width,height]
                    if (not np.isfinite(boxes).all() or np.any(boxes < 0) or np.any(boxes > 1)
                            or np.any(boxes[:,2:] <= boxes[:,:2])):
                        raise ValueError('A recovery detection lies outside the ordinary view')
                    detected[index] = boxes.tolist()
                    record.update(detected=True,boxes=boxes.tolist(),recovery_detected=True,recovery_attempts=details)
                    report['recovery']['detected_views'].append(index)
                except ValueError as exc:
                    record.update(recovery_detected=False,recovery_error=str(exc),
                                  recovery_attempts=getattr(exc,'attempts',[]))
            report['recovery']['succeeded'] = bool(detected)
        if destination is not None:
            (destination/'output_landmarks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if not detected:
            error = ValueError('Face detection failed in all ordinary anchor views')
            error.detection_report = report
            raise error
        return interpolate_boxes(detected, count, anchors.device), report


def detect_output_regions(anchors, model, dest=None, *, policy='whole_first', recovery=False):
    """Single-session entry point sharing detection and coordinate conversion with batch processing."""
    with OutputRegionDetector(model, policy=policy) as detector:
        return detector.detect(anchors, dest, recovery=recovery)


def read_prepared_records(path):
    path = Path(path)
    path = path/'regions.jsonl' if path.is_dir() else path
    records = {}
    if not path.is_file():
        return records
    with path.open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Preprocessing record on line {line_number} is incomplete; repair it before continuing') from exc
            reference = str(Path(row['reference']).resolve())
            if reference in records:
                raise ValueError('The preprocessing file contains a duplicate reference: ' + reference)
            records[reference] = row
    return records


def load_prepared_regions(path):
    """Return successful reference-to-box mappings without teacher or historical-box fallbacks."""
    result = {}
    for key, row in read_prepared_records(path).items():
        if not row['success']:
            continue
        if 'view_boxes' not in row and 'view_boxes' not in row['boxes']:
            # Reuse the three recorded detections and the same linear interpolation to restore
            # the twelve anchor boxes without GPU execution or inferred facial positions.
            observations = row.get('detection', {}).get('selected_views')
            if observations is not None:
                detected = {int(item['view']): item['boxes'] for item in observations if item['detected']}
                count = int(row.get('anchor_count', 12))
                if not detected or count < 1 or min(detected) < 0 or max(detected) >= count:
                    raise ValueError('A successful localization record lacks valid measured views for restoring per-view boxes: ' + key)
                row = dict(row, view_boxes=interpolate_boxes(detected, count, 'cpu').tolist())
        result[key] = validate_output_boxes(row, key)
    return result
