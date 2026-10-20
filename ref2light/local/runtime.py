import math


import numpy as np


import torch


import torch.nn.functional as F


def validate_eye_mode(mode):
    if mode not in ('local', 'frozen_base'):
        raise ValueError('eye_mode must be local or frozen_base')
    return mode


def canonical_pose(raw, eye_mode='local'):
    validate_eye_mode(eye_mode)
    value = list(map(float, raw))
    if len(value) != 14 or not all(map(math.isfinite, value)):
        raise ValueError('Expected 14 finite raw pose controls')
    if eye_mode == 'local':
        value[6] = 1.
    for index, number in ((7, 0.), (12, 0.), (13, 0.)):
        value[index] = number
    return value


def pixel_box(box, width, height):
    a = np.asarray(box, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all() or np.any(a[2:] <= a[:2]):
        raise ValueError('Local boxes must be finite and have positive area')
    x0, y0 = np.floor(a[:2]*[width, height]).astype(int)
    x1, y1 = np.ceil(a[2:]*[width, height]).astype(int)
    x0, y0, x1, y1 = max(0,x0), max(0,y0), min(width,x1), min(height,y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('The local box lies outside the image')
    return int(x0), int(y0), int(x1), int(y1)


def soft_mask(height, width, device, feather=.08):
    """Feather the boundary; composition and sampling give exactly zero outside the region."""
    if not 0 < feather < .5:
        raise ValueError('The feather fraction must be between 0 and 0.5')
    y = (torch.arange(height,device=device,dtype=torch.float32)+.5)/height
    x = (torch.arange(width,device=device,dtype=torch.float32)+.5)/width
    yy,xx = torch.meshgrid(y,x,indexing='ij')
    d = torch.minimum(torch.minimum(xx,1-xx),torch.minimum(yy,1-yy))/feather
    t=d.clamp(0,1)
    return (t*t*(3-2*t))[None,None]


def patch_grid(grid, box, full_hw):
    """Convert ordinary-view coordinates of native subpixels to unscaled crop coordinates."""
    height,width=full_hw
    x0,y0,x1,y1=box
    result=grid.clone()
    result[...,0]=((grid[...,0]+1)*width/2-x0)*2/(x1-x0)-1
    result[...,1]=((grid[...,1]+1)*height/2-y0)*2/(y1-y0)-1
    return result


def sample_patch(patch, grid, padding='zeros'):
    # Use zero padding for local delta and border padding for the frozen base-image halo.
    volume=patch.float().permute(1,0,2,3)[:,None]
    return F.grid_sample(volume,grid.float(),mode='bilinear',padding_mode=padding,
                         align_corners=False)[:,0,0,0]

