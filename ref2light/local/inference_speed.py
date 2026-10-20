import torch


from ref2light.local.runtime import pixel_box


def branch_is_inactive(group, raw):
    """Skip only branches with analytically zero output; always evaluate the mouth branch."""
    if torch.is_tensor(raw):
        raise TypeError('Provide a validated CPU pose list to avoid extra CUDA synchronization')
    if len(raw) != 14:
        raise ValueError('Expected 14 raw pose controls')
    if group == 'eyes':
        return float(raw[6]) >= 1.
    if group == 'gaze':
        return float(raw[6]) <= 0. or (float(raw[12]) == 0. and float(raw[13]) == 0.)
    if group == 'mouth':
        return False
    raise ValueError('branch must be eyes, mouth, or gaze')


@torch.no_grad()
def footprint_in_search(uv, source_bounds, search_box):
    """Match source UV within a bounded search window while preserving cross-view union bounds."""
    count, height, width, _ = uv.shape
    x0, y0, x1, y1 = pixel_box(search_box, width, height)
    local = uv[:, y0:y1, x0:x1]
    low, high = source_bounds[:, None, None, :2], source_bounds[:, None, None, 2:]
    selected = ((local >= low) & (local <= high)).all(-1).any(0)
    occupied_y, occupied_x = selected.any(1), selected.any(0)
    yy = torch.arange(y0, y1, device=uv.device)
    xx = torch.arange(x0, x1, device=uv.device)
    low_x = torch.where(occupied_x, xx, width).amin()
    high_x = torch.where(occupied_x, xx+1, 0).amax()
    low_y = torch.where(occupied_y, yy, height).amin()
    high_y = torch.where(occupied_y, yy+1, 0).amax()
    numbers = torch.stack((low_x, low_y, high_x, high_y, selected.sum())).float().cpu().tolist()
    if numbers[-1] < 12:
        raise ValueError('Fewer than 12 visible pixels have matching source UV coordinates')
    return [numbers[0]/width, numbers[1]/height, numbers[2]/width, numbers[3]/height], int(numbers[-1])

