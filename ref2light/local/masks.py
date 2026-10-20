import math

import torch


def _integer_tuple(values, length, label):
    if len(values) != length:
        raise ValueError(label+' has invalid dimensions')
    result = []
    for value in values:
        if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value:
            raise ValueError(label+' must contain finite integers')
        result.append(int(value))
    return tuple(result)


def patch_mask(group, crop_box, full_hw, patch_hw, view_boxes, device,
               eyes_scale=1.0, mouth_scale=1.6, *, mode='ellipse',
               context_feather=.08, split_feather=.10, split_mode='per_view'):
    """Return a K x 1 x patchH x patchW mask from pixel crop_box and normalized K x 3 x 4 view_boxes.
    context_soft splits at the eye-mouth midline; global_median shares a split from median centers across views."""
    if group not in ('eyes', 'gaze', 'mouth'):
        raise ValueError('The write region must be eyes, gaze, or mouth')
    if mode not in ('ellipse', 'context_soft'):
        raise ValueError('The write mode must be ellipse or context_soft')
    if split_mode not in ('per_view', 'global_median'):
        raise ValueError('The split mode must be per_view or global_median')
    if mode != 'context_soft' and split_mode != 'per_view':
        raise ValueError('global_median is only supported with context_soft')
    height, width = _integer_tuple(full_hw, 2, 'Ordinary-view dimensions')
    patch_height, patch_width = _integer_tuple(patch_hw, 2, 'Patch dimensions')
    x0, y0, x1, y1 = _integer_tuple(crop_box, 4, 'Crop coordinates')
    if min(height, width, patch_height, patch_width) <= 0:
        raise ValueError('Ordinary-view and patch dimensions must be positive')
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError('The crop must lie within the ordinary view and have positive area')
    for scale in (eyes_scale, mouth_scale):
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError('Eye and mouth region scales must be finite and positive')
    if not math.isfinite(float(context_feather)) or not 0 < context_feather < .5:
        raise ValueError('The crop feather fraction must be between 0 and 0.5')
    if not math.isfinite(float(split_feather)) or not 0 < split_feather < 1:
        raise ValueError('The eye-mouth split feather fraction must be between 0 and 1')
    # Validate and process small CPU box lists before sending them to the GPU, avoiding
    # per-frame CUDA synchronization for finiteness and area checks.
    boxes = (view_boxes.to(dtype=torch.float32) if torch.is_tensor(view_boxes)
             else torch.as_tensor(view_boxes, dtype=torch.float32, device='cpu'))
    if boxes.ndim != 3 or boxes.shape[1:] != (3, 4) or boxes.shape[0] < 1:
        raise ValueError('view_boxes must have shape K x 3 x 4')
    if not bool(torch.isfinite(boxes).all()) or not bool((boxes[..., 2:] > boxes[..., :2]).all()):
        raise ValueError('Facial boxes must be finite and have positive area')
    if mode == 'context_soft':
        centers = (boxes[..., :2]+boxes[..., 2:])/2
        if split_mode == 'global_median':
            centers = centers.median(dim=0).values[None].expand(boxes.shape[0], -1, -1)
        eye_center = centers[:, :2].mean(1)
        mouth_center = centers[:, 2]
        dimensions = boxes.new_tensor([width, height])
        axis = (mouth_center-eye_center)*dimensions
        squared_length = axis.square().sum(-1)
        if not bool((squared_length > 1e-8).all()):
            raise ValueError('Eye and mouth centers coincide; the geometric split is undefined')
        eye_center, dimensions, axis, squared_length = (
            value.to(device=device) for value in (eye_center, dimensions, axis, squared_length))
    # Map patch pixel centers to the unstretched ordinary crop, matching align_corners=False.
    x = (x0+(torch.arange(patch_width, device=device, dtype=torch.float32)+.5)
         * ((x1-x0)/patch_width))/width
    y = (y0+(torch.arange(patch_height, device=device, dtype=torch.float32)+.5)
         * ((y1-y0)/patch_height))/height
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    xy = torch.stack((xx, yy), -1)
    if mode == 'context_soft':
        # Write within a broad context crop, with the soft rectangle independent of detector-box aspect ratios.
        local_x = (torch.arange(patch_width, device=device, dtype=torch.float32)+.5)/patch_width
        local_y = (torch.arange(patch_height, device=device, dtype=torch.float32)+.5)/patch_height
        local_yy, local_xx = torch.meshgrid(local_y, local_x, indexing='ij')
        edge_distance = torch.minimum(torch.minimum(local_xx, 1-local_xx),
                                      torch.minimum(local_yy, 1-local_yy))
        edge = (edge_distance/float(context_feather)).clamp(0, 1)
        context = edge.square()*(3-2*edge)
        # Define the split perpendicular to the eye-mouth axis in full-image pixels to preserve its direction.
        offset = (xy[None]-eye_center[:, None, None])*dimensions
        projection = (offset*axis[:, None, None]).sum(-1)/squared_length[:, None, None]
        transition = ((projection-(.5-float(split_feather)/2))/float(split_feather)).clamp(0, 1)
        mouth_share = transition.square()*(3-2*transition)
        # Eye and mouth shares sum to one before multiplication by crop feathers bounded by one.
        share = mouth_share if group == 'mouth' else 1-mouth_share
        return (share*context[None])[:, None]
    # Reuse the mask equations after validating boxes and scales once on CPU.
    boxes = boxes.to(device=device)
    centers = (boxes[..., :2]+boxes[..., 2:])/2
    scales = boxes.new_tensor([eyes_scale, eyes_scale, mouth_scale])
    radii = (boxes[..., 2:]-boxes[..., :2])*scales[None, :, None]/2
    distance = (((xy[None, None]-centers[:, :, None, None])/radii[:, :, None, None])
                .square().sum(-1)).sqrt()
    transition = ((distance-.65)/(1-.65)).clamp(0, 1)
    ellipses = 1-transition.square()*(3-2*transition)
    eyes = torch.maximum(ellipses[:, 0], ellipses[:, 1])
    region = ellipses[:, 2]*(1-eyes) if group == 'mouth' else eyes
    return region[:, None]
