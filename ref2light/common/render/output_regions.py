import numpy as np


import torch


def interpolate_boxes(detected, count, device):
    """Interpolate by view index, extending the nearest successful view at missing endpoints."""
    if not detected:
        raise ValueError('No usable facial regions were detected in the generated ordinary views')
    indices = sorted(detected)
    array = np.stack([detected[index] for index in indices]).astype(np.float32)
    if array.shape[1:]!=(3,4) or not np.isfinite(array).all() or np.any(array[:,:,2:]<=array[:,:,:2]):
        raise ValueError('Invalid ordinary-view annotation dimensions or region area')
    result = np.stack([np.interp(np.arange(count),indices,array[:,r,d])
                       for r in range(3) for d in range(4)],-1).reshape(count,3,4)
    return torch.as_tensor(result,dtype=torch.float32,device=device)

