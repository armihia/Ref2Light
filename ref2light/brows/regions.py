from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ref2light.local.output_regions import FACE_OVAL, recovery_crops
from ref2light.local.roi_tracking import source_uv, _footprint, _search
from ref2light.local.runtime import pixel_box
from ref2light.common.landmarks import EYE_A, EYE_B, region_box


def validate_layout(layout, count, reference=None):
    boxes = np.asarray(layout['view_boxes'], dtype=float)
    if boxes.shape != (count, 2, 4) or not np.isfinite(boxes).all() or np.any(boxes[..., 2:] <= boxes[..., :2]):
        raise ValueError('Eyebrow boxes must be finite K x 2 x 4 rectangles with positive area')
    if reference and layout.get('reference') and Path(layout['reference']).resolve() != Path(reference).resolve():
        raise ValueError('The eyebrow boxes belong to another reference image')
    return dict(layout, view_boxes=boxes.tolist())


class BrowDetector:
    def __init__(self, model):
        import mediapipe as mp
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as C
        self.mp = mp
        self.groups = [sorted({i for edge in edges for i in (edge.start, edge.end)})
                       for edges in (C.FACE_LANDMARKS_LEFT_EYEBROW, C.FACE_LANDMARKS_RIGHT_EYEBROW)]
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(Path(model).resolve()),
                                             delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1)
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self):
        self.detector.close()

    def detect(self, image):
        image = image.convert('RGB')
        width, height = image.size
        side = max(width, height)
        left, top = (side-width)//2, (side-height)//2
        canvas = Image.new('RGB', (side, side))
        canvas.paste(image, (left, top))
        attempts = [('upper_centre', (side//4, 0, 3*side//4, side//2)),
                    ('whole', (0, 0, side, side)), *recovery_crops((side, side))]
        for label, crop in attempts:
            result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                data=np.asarray(canvas.crop(crop).resize((512, 512))).copy()))
            if not result.face_landmarks:
                continue
            local = np.array([(p.x, p.y) for p in result.face_landmarks[0]])
            if np.any(local[list(FACE_OVAL)] < .005) or np.any(local[list(FACE_OVAL)] > .995):
                continue
            points = local*[crop[2]-crop[0], crop[3]-crop[1]] + np.array(crop[:2])
            points = (points-[left, top])/[width, height]
            if np.any(points[list(FACE_OVAL)] < 0) or np.any(points[list(FACE_OVAL)] > 1):
                continue
            boxes = []
            for ids in self.groups:
                selected = points[ids]
                low, high = selected.min(0), selected.max(0)
                size = high-low
                # Reserve room for vertical eyebrow movement using ordinary-image annotations.
                pad = np.array([.18*size[0], max(1.5*size[1], .15*size[0]*width/height)])
                box = np.r_[low-pad, high+pad].clip(0, 1)
                if np.any(box[2:] <= box[:2]):
                    break
                boxes.append(box.tolist())
            if len(boxes) == 2:
                boxes.sort(key=lambda b: b[0]+b[2])
                eyes = []
                for ids in (EYE_A, EYE_B):
                    square = points[list(ids)]*[width/side, height/side]
                    eye = np.array(region_box(square, 1.5, .45))*[side/width, side/height, side/width, side/height]
                    eyes.append(eye.tolist())
                eyes.sort(key=lambda b: b[0]+b[2])
                return dict(boxes=boxes, eyes=eyes, method=label, uses_native_Y=False)
        raise ValueError('Eyebrow detection failed in ordinary views; eyebrow control is disabled, while eye and mouth controls remain available')

    def anchors(self, anchors):
        count = anchors.shape[1]
        found = {}
        failures = []
        for index in sorted({0, count//2, count-1}):
            array = anchors[0, index].detach().float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            try:
                found[index] = self.detect(Image.fromarray(array))
            except ValueError as exc:
                failures.append(dict(view=index, error=str(exc)))
        if not found:
            raise ValueError('Eyebrow localization failed in all ordinary anchor views')
        keys = sorted(found)
        values = np.array([found[k]['boxes'] for k in keys])
        interpolated = np.stack([np.interp(np.arange(count), keys, values[:, i, j])
                                 for i in range(2) for j in range(4)], -1).reshape(count, 2, 4)
        return dict(view_boxes=interpolated.tolist(), source='predicted_ordinary_anchors',
                    detected={str(k): v for k, v in found.items()}, failures=failures,
                    uses_native_Y=False, uses_character_geometry=False)

    def from_reference(self, image, output_boxes):
        """Estimate output eyebrows from reference brow-to-eye geometry and detected output eyes when direct localization fails."""
        detected = self.detect(image)
        source_eyes = np.asarray(detected['eyes'])
        source_brows = np.asarray(detected['boxes'])
        output_eyes = np.asarray(output_boxes['view_boxes'])[:, :2]
        scale = (output_eyes[..., 2:]-output_eyes[..., :2])/(source_eyes[..., 2:]-source_eyes[..., :2])
        lower = output_eyes[..., :2]+(source_brows[..., :2]-source_eyes[..., :2])*scale
        upper = output_eyes[..., :2]+(source_brows[..., 2:]-source_eyes[..., :2])*scale
        boxes = np.concatenate((lower, upper), -1)
        return dict(view_boxes=boxes.tolist(), source='reference_brow_eye_relation_projected_to_output_eyes',
                    direct_brow_detection_failed=True, estimated=True, reference_detection=detected,
                    uses_native_Y=False, uses_character_geometry=False)

    def locate(self, anchors, reference, output_boxes):
        try:
            return self.anchors(anchors)
        except ValueError as exc:
            result = self.from_reference(reference, output_boxes)
            result['ordinary_brow_detection_error'] = str(exc)
            return result


def crop_bounds(layout):
    boxes = np.asarray(layout['view_boxes'])
    low, high = boxes[..., :2].min((0, 1)), boxes[..., 2:].max((0, 1))
    pad = (high-low)*.1
    return np.r_[low-pad, high+pad].clip(0, 1).tolist()


def support_mask(layout, crop, full_hw, patch_hw, device):
    height, width = full_hw
    ph, pw = patch_hw
    x0, y0, x1, y1 = crop
    x = (x0+(torch.arange(pw, device=device)+.5)*(x1-x0)/pw)/width
    y = (y0+(torch.arange(ph, device=device)+.5)*(y1-y0)/ph)/height
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    xy = torch.stack((xx, yy), -1)
    boxes = torch.tensor(layout['view_boxes'], dtype=torch.float32, device=device)
    centre, radius = (boxes[..., :2]+boxes[..., 2:])/2, (boxes[..., 2:]-boxes[..., :2])/2
    distance = ((xy[None, None]-centre[:, :, None, None])/radius[:, :, None, None]).abs().amax(-1)
    edge = ((1-distance)/.2).clamp(0, 1)
    return (edge.square()*(3-2*edge)).amax(1)[:, None]


@torch.no_grad()
def prepare_brow_tracking(flow, layout):
    box = crop_bounds(layout)
    uv = source_uv(flow)
    h, w = flow.shape[-2:]
    x0, y0, x1, y1 = pixel_box(box, w, h)
    points = uv[:, y0:y1, x0:x1].reshape(uv.shape[0], -1, 2)
    if points.shape[1] < 16:
        return None
    bounds = torch.cat((points.quantile(.02, dim=1), points.quantile(.98, dim=1)), 1)
    search = _search(box)
    footprint, _ = _footprint(uv, bounds, search)
    return dict(bounds=bounds, search=search, footprint=footprint)


@torch.no_grad()
def track_brows(flow, layout, state):
    if state is None:
        return layout
    current, _ = _footprint(source_uv(flow), state['bounds'], state['search'])
    old, new = np.asarray(state['footprint']), np.asarray(current)
    scale = (new[2:]-new[:2])/(old[2:]-old[:2])
    if not np.isfinite(scale).all() or np.any(scale < .35) or np.any(scale > 3):
        return layout
    boxes = np.asarray(layout['view_boxes'])
    centre = (boxes[..., :2]+boxes[..., 2:])/2 + (new[:2]+new[2:]-old[:2]-old[2:])/2
    radius = (boxes[..., 2:]-boxes[..., :2])/2*scale
    return dict(layout, view_boxes=np.concatenate((centre-radius, centre+radius), -1).tolist())

