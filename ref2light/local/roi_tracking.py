import torch

from ref2light.local.runtime import pixel_box
from ref2light.local.inference_speed import footprint_in_search


def source_uv(flow):
    if flow.ndim != 5 or flow.shape[0] != 1 or flow.shape[2] != 2:
        raise ValueError('Expected source sampling displacement with shape 1 x K x 2 x H x W')
    _, _, _, height, width = flow.shape
    y = 2 * (torch.arange(height, device=flow.device, dtype=torch.float32) + .5) / height - 1
    x = 2 * (torch.arange(width, device=flow.device, dtype=torch.float32) + .5) / width - 1
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return (torch.stack((xx, yy), dim=-1)[None] + flow[0].float().permute(0, 2, 3, 1)).clamp(-1.2, 1.2)


def _footprint(uv, source_bounds, search_box):
    # Crop to the search area before comparing source coordinates across the twelve views.
    return footprint_in_search(uv, source_bounds, search_box)


def _search(box, factor=3.):
    value = torch.tensor(box, dtype=torch.float32)
    centre, radius = (value[:2] + value[2:]) / 2, (value[2:] - value[:2]) / 2 * factor
    return torch.cat((centre - radius, centre + radius)).clamp(0, 1).tolist()


@torch.no_grad()
def prepare_tracking(flow, boxes):
    uv = source_uv(flow)
    height, width = uv.shape[1:3]
    state, report = {}, {}
    for group in ('eyes', 'mouth'):
        box = boxes[group]
        try:
            x0, y0, x1, y1 = pixel_box(box, width, height)
            # Exclude box edges and use per-anchor UV quantiles to suppress displacement outliers.
            padx, pady = int((x1-x0)*.1), int((y1-y0)*.1)
            points = uv[:, y0+pady:y1-pady, x0+padx:x1-padx].reshape(uv.shape[0], -1, 2)
            if points.shape[1] < 16:
                raise ValueError('The neutral facial box is too small for stable source-UV calibration')
            stride = max(1, points.shape[1] // 4096)
            points = points[:, ::stride]
            lower, upper = torch.quantile(points, .02, dim=1), torch.quantile(points, .98, dim=1)
            source_bounds = torch.cat((lower, upper), dim=1)
            if not bool(torch.isfinite(source_bounds).all()) or not bool((upper > lower).all()):
                raise ValueError('The neutral source-UV region has no finite positive area')
            search = _search(box)
            footprint, pixels = _footprint(uv, source_bounds, search)
            state[group] = dict(source_bounds=source_bounds, footprint=footprint,
                                output_box=list(box), search_box=search)
            report[group] = dict(available=True, source_uv_bounds=source_bounds.cpu().tolist(),
                                 neutral_footprint=footprint, pixels=pixels)
        except ValueError as exc:
            report[group] = dict(available=False, fallback='neutral_fixed', error=str(exc))
    return state, report


@torch.no_grad()
def track_boxes(flow, boxes, state):
    uv = source_uv(flow)
    result, report = {}, {}
    for group in ('eyes', 'mouth'):
        original = boxes[group]
        if group not in state:
            result[group] = list(original)
            report[group] = dict(tracked=False, fallback='neutral_fixed', error='Neutral UV calibration is unavailable')
            continue
        item = state[group]
        try:
            current, pixels = _footprint(uv, item['source_bounds'], item['search_box'])
            neutral = torch.tensor(item['footprint'])
            current = torch.tensor(current)
            scale = (current[2:] - current[:2]) / (neutral[2:] - neutral[:2])
            if not bool(torch.isfinite(scale).all()) or bool(((scale < .35) | (scale > 3.)).any()):
                raise ValueError('The source-UV region scale is out of range')
            original = torch.tensor(original)
            centre = (original[:2] + original[2:]) / 2 + (current[:2]+current[2:]-neutral[:2]-neutral[2:])/2
            radius = (original[2:] - original[:2]) / 2 * scale
            box = torch.cat((centre-radius, centre+radius)).clamp(0, 1)
            if not bool((box[2:] > box[:2]).all()):
                raise ValueError('The tracked region no longer has valid area')
            result[group] = box.tolist()
            report[group] = dict(tracked=True, pixels=pixels, scale=scale.tolist())
        except ValueError as exc:
            result[group] = list(boxes[group])
            report[group] = dict(tracked=False, fallback='neutral_fixed', error=str(exc))
    if boxes.get('view_boxes') is not None:
        original_views = torch.as_tensor(boxes['view_boxes'], dtype=torch.float64)
        moved_views = original_views.clone()
        for group, indices in (('eyes', (0, 1)), ('mouth', (2,))):
            old, new = torch.tensor(boxes[group], dtype=torch.float64), torch.tensor(result[group], dtype=torch.float64)
            scale = (new[2:]-new[:2])/(old[2:]-old[:2])
            for index in indices:
                moved_views[:, index, :2] = new[:2] + (original_views[:, index, :2]-old[:2])*scale
                moved_views[:, index, 2:] = new[:2] + (original_views[:, index, 2:]-old[:2])*scale
        # Keep out-of-frame box extents; the shared elliptical mask clips the written region naturally.
        result['view_boxes'] = moved_views.tolist()
    return result, report
