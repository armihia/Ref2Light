import math
import torch
from torch import nn
from torch.nn import functional as F
from ref2light.rad.model import _pose, _fp32

BROW_FORMAT = 'companion1_v67_brow_head'


class BrowHead(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        values = dict(width=32, flow_limit=.35, rgb_limit=.25, axis_max=.8,
                      patch_height=64, patch_width=192)
        if config:
            if set(config)-set(values):
                raise ValueError('Unknown eyebrow-head configuration')
            values.update(config)
        for key in ('width', 'patch_height', 'patch_width'):
            if type(values[key]) is not int or values[key] < 2:
                raise ValueError(key+' must be an integer of at least 2')
        for key in ('flow_limit', 'rgb_limit', 'axis_max'):
            if not math.isfinite(values[key]) or not 0 < values[key] <= 1:
                raise ValueError(key+' must be in (0, 1]')
        self.config = values
        width = values['width']
        self.trunk = nn.Sequential(nn.Conv2d(9, width, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.output = nn.Conv2d(width, 30, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, visible, reference, pose, mask):
        if visible.ndim != 4 or visible.shape[1] != 3 or reference.shape != visible.shape:
            raise ValueError('Eyebrow inputs must be ordinary-view tensors of matching shape K x 3 x H x W')
        count, _, height, width = visible.shape
        raw = _pose(pose, count, visible.device)
        ys = torch.linspace(-1, 1, height, device=visible.device)
        xs = torch.linspace(-1, 1, width, device=visible.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        zz = torch.linspace(-1, 1, count, device=visible.device) if count > 1 else xs.new_zeros(1)
        coordinates = torch.cat((torch.stack((xx, yy))[None].expand(count, -1, -1, -1),
                                 zz[:, None, None, None].expand(-1, 1, height, width)), 1)
        inputs = torch.cat((visible.detach(), reference.detach().to(visible.dtype),
                            coordinates.to(visible.dtype)), 1)
        predicted = self.output(self.trunk(inputs)).reshape(count, 3, 2, 5, height, width)
        with _fp32(visible.device):
            axes = (raw[:, 9:12]/self.config['axis_max']).clamp(-1, 1)
            weights = torch.stack(((-axes).clamp(min=0), axes.clamp(min=0)), 2)
            fields = (predicted.float()*weights[:, :, :, None, None, None]).sum((1, 2))
            # Keep the same displacement and colour bounds when all three axes reach their extremes.
            flow = self.config['flow_limit']*fields[:, :2].tanh()
            residual = self.config['rgb_limit']*fields[:, 2:].tanh()
            grid = torch.stack((xx, yy), -1)[None].expand(count, -1, -1, -1)
            before = visible.detach().float()
            warped = F.grid_sample(before, grid+flow.permute(0, 2, 3, 1),
                                   mode='bilinear', padding_mode='border', align_corners=True)
            enabled = axes.abs().sum(1)[:, None, None, None] > 0
            delta = torch.where(enabled, warped+residual-before, torch.zeros_like(before))*mask
            return dict(delta=delta, flow=flow, residual=residual)

    def export(self):
        return dict(format=BROW_FORMAT, config=dict(self.config),
                    brow={k: v.detach().cpu().clone() for k, v in self.state_dict().items()})

    @classmethod
    def from_export(cls, payload, device='cpu'):
        if payload.get('format') != BROW_FORMAT:
            raise ValueError('Expected v6.7 eyebrow-head weights')
        model = cls(payload['config'])
        model.load_state_dict(payload['brow'], strict=True)
        return model.to(device)

