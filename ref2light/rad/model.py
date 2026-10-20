from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


RAD_FORMAT = 'companion1_v66_mouth_rad_head'


def _fp32(device):
    return (torch.autocast(device_type=device.type, enabled=False)
            if device.type in ('cpu', 'cuda') else nullcontext())


def _mask(mask, count, height, width, device):
    if mask is None:
        return torch.ones((1, 1, height, width), device=device, dtype=torch.float32)
    support = torch.as_tensor(mask, dtype=torch.float32, device=device)
    if support.ndim == 2:
        support = support[None, None]
    elif support.ndim == 3:
        support = support.unsqueeze(1)
    if (support.ndim != 4 or support.shape[1] != 1 or support.shape[0] not in (1, count)
            or support.shape[-2:] != (height, width)):
        raise ValueError('mask must have shape [H, W], [1/K, H, W], or [1/K, 1, H, W]')
    return support.clamp(0, 1)


def _pose(pose, count, device):
    if not torch.is_tensor(pose):
        pose = torch.as_tensor(pose, device=device, dtype=torch.float32)
    if pose.device != device:
        pose = pose.to(device=device)
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    if pose.ndim != 2 or pose.shape[1] != 14 or pose.shape[0] not in (1, count):
        raise ValueError('pose must contain raw controls with shape [14], [1, 14], or [K, 14]')
    return pose.float().expand(count, -1)


@torch.no_grad()
def extract_old_features(legacy_local, canonical, reference, pose, mask=None, identity=None):
    """Extract frozen mouth up0 features and the raw8=0 output in a single CNN pass.
    legacy_visible keeps unclamped patch values; insert delta into the full image before clamping."""
    from ref2light.local.local import LocalControls
    if not isinstance(legacy_local, LocalControls):
        raise TypeError('Feature reuse requires the supported LocalControls architecture')
    if legacy_local.training or any(parameter.requires_grad for parameter in legacy_local.parameters()):
        raise ValueError('Set the existing model to eval().requires_grad_(False) before training the rad head')
    if canonical.ndim != 4 or canonical.shape[1] != 3 or canonical.shape != reference.shape:
        raise ValueError('canonical and reference must both have shape [K, 3, H, W]')
    if canonical.device != reference.device or min(canonical.shape[-2:]) < 8:
        raise ValueError('Both patches must share a device and have height and width of at least 8')
    if not canonical.is_floating_point() or not reference.is_floating_point():
        raise ValueError('Ordinary RGB patches must use floating-point tensors')
    count, _, height, width = canonical.shape
    raw = legacy_local._pose(pose, count, canonical.device)
    # Use basic slicing to avoid CPU index allocation inside CUDA Graphs; preserve the raw pose.
    raw = torch.cat((raw[:, :8], torch.zeros_like(raw[:, 8:9]), raw[:, 9:]), dim=1)
    ys = torch.linspace(-1, 1, height, device=canonical.device, dtype=torch.float32)
    xs = torch.linspace(-1, 1, width, device=canonical.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    zz = (torch.linspace(-1, 1, count, device=canonical.device, dtype=torch.float32)
          if count > 1 else torch.zeros(1, device=canonical.device))
    coordinates = torch.stack((xx, yy)).unsqueeze(0).expand(count, -1, -1, -1)
    coordinates = torch.cat((coordinates, zz[:, None, None, None].expand(-1, 1, height, width)), 1)
    inputs = torch.cat((canonical, reference, coordinates.to(canonical.dtype)), dim=1)
    feature_pose = raw
    if not legacy_local.config['mouth_open_in_features']:
        feature_pose = torch.cat((raw[:, :7], torch.zeros_like(raw[:, 7:8]), raw[:, 8:]), dim=1)
    branch = legacy_local.branches['mouth']
    if branch.axes != (7, 8):
        raise ValueError('The mouth-conditioning axes do not match the reusable feature structure')
    condition = branch.condition(torch.stack([feature_pose[:, index] for index in branch.axes], dim=1).to(inputs.dtype))
    if legacy_local.config['mouth_identity_condition']:
        if (not torch.is_tensor(identity) or identity.shape != (1, 512)
                or identity.device != canonical.device or not identity.is_floating_point()):
            raise ValueError('Mouth identity conditioning must be a floating-point [1, 512] tensor on the same device')
        condition = condition + branch.identity_projection(branch.identity_norm(identity).to(condition.dtype))
    x0 = branch.down0(inputs, condition)
    x1 = branch.down1(F.avg_pool2d(x0, 2), condition)
    x2 = branch.down2(F.avg_pool2d(x1, 2), condition)
    feature = branch.up1(torch.cat((F.interpolate(x2, size=x1.shape[-2:], mode='bilinear',
                                                  align_corners=False), x1), dim=1), condition)
    feature = branch.up0(torch.cat((F.interpolate(feature, size=x0.shape[-2:], mode='bilinear',
                                                  align_corners=False), x0), dim=1), condition)
    predicted = branch.output(feature)
    with _fp32(canonical.device):
        base = canonical.float()
        predicted = predicted.float()
        if legacy_local.config['mouth_endpoints']:
            closed, opened = predicted.chunk(2, dim=1)
            amount = (raw[:, 7]/legacy_local.config['mouth_endpoint_open']).clamp(0, 1)
            predicted = torch.lerp(closed, opened, amount[:, None, None, None])
        flow = legacy_local.config['flow_limit']*torch.tanh(predicted[:, :2].float())
        residual = legacy_local.config['rgb_limit']*torch.tanh(predicted[:, 2:].float())
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(count, -1, -1, -1)
        sampled = F.grid_sample(base, grid+flow.permute(0, 2, 3, 1), mode='bilinear',
                                padding_mode='border', align_corners=True)
        support = _mask(mask, count, height, width, base.device)
        delta = (sampled+residual-base)*support*torch.ones_like(raw[:, 6])[:, None, None, None]
        return dict(features=feature, legacy_visible=base+delta, legacy_delta=delta)


class RadHead(nn.Module):
    """Interpolate closed and open correction fields and scale them by positive rad.
    Detach the frozen input features; this branch is zero for negative rad."""
    def __init__(self, config=None):
        super().__init__()
        values = dict(feature_channels=32, hidden_channels=0, flow_limit=.25, rgb_limit=.5,
                      open_max=.75, rad_max=.8)
        if config is not None:
            unknown = set(config)-set(values)
            if unknown:
                raise ValueError('Unsupported rad-head configuration: '+', '.join(sorted(unknown)))
            values.update(config)
        if isinstance(values['feature_channels'], bool) or not isinstance(values['feature_channels'], int) or values['feature_channels'] < 1:
            raise ValueError('feature_channels must be a positive integer')
        if (isinstance(values['hidden_channels'], bool) or not isinstance(values['hidden_channels'], int)
                or values['hidden_channels'] < 0):
            raise ValueError('hidden_channels must be non-negative; 0 selects the legacy linear head')
        for key in ('flow_limit', 'rgb_limit', 'open_max', 'rad_max'):
            if isinstance(values[key], bool) or not 0 < float(values[key]) <= 1:
                raise ValueError(key+' must be in (0, 1]')
            values[key] = float(values[key])
        self.config = dict(values)
        hidden = values['hidden_channels']
        # Keep legacy linear parameter names and shapes; the nonlinear head adds two local convolutions.
        self.adapter = (nn.Sequential(
            nn.Conv2d(values['feature_channels'] + 6, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU()) if hidden else None)
        self.output = nn.Conv2d(hidden or values['feature_channels'], 10, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, visible, pose, mask=None):
        if (features.ndim != 4 or visible.ndim != 4 or features.shape[0] != visible.shape[0]
                or features.shape[1] != self.config['feature_channels'] or visible.shape[1] != 3
                or features.shape[-2:] != visible.shape[-2:] or min(visible.shape[-2:]) < 2):
            raise ValueError('Features must have shape [K, C, H, W] and the existing mouth image must have matching shape [K, 3, H, W]')
        if features.device != visible.device or not features.is_floating_point() or not visible.is_floating_point():
            raise ValueError('Features and the existing mouth image must be floating-point tensors on the same device')
        count, _, height, width = visible.shape
        raw = _pose(pose, count, visible.device)
        ys = torch.linspace(-1, 1, height, device=visible.device, dtype=torch.float32)
        xs = torch.linspace(-1, 1, width, device=visible.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        inputs = features.detach()
        if self.adapter is not None:
            zz = (torch.linspace(-1, 1, count, device=visible.device, dtype=torch.float32)
                  if count > 1 else torch.zeros(1, device=visible.device, dtype=torch.float32))
            coordinates = torch.stack((xx, yy)).unsqueeze(0).expand(count, -1, -1, -1)
            coordinates = torch.cat((coordinates, zz[:, None, None, None].expand(-1, 1, height, width)), 1)
            inputs = torch.cat((inputs, visible.detach().float().to(inputs.dtype),
                                coordinates.to(inputs.dtype)), dim=1)
            inputs = self.adapter(inputs)
        predicted = self.output(inputs)
        with _fp32(visible.device):
            before = visible.detach().float()
            closed, opened = predicted.float().chunk(2, dim=1)
            u = (raw[:, 7]/self.config['open_max']).clamp(0, 1)[:, None, None, None]
            r = (raw[:, 8]/self.config['rad_max']).clamp(0, 1)[:, None, None, None]
            fields = torch.lerp(closed, opened, u)
            flow = self.config['flow_limit']*fields[:, :2].tanh()*r
            residual = self.config['rgb_limit']*fields[:, 2:].tanh()*r
            grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(count, -1, -1, -1)
            sampled = F.grid_sample(before, grid+flow.permute(0, 2, 3, 1), mode='bilinear',
                                    padding_mode='border', align_corners=True)
            support = _mask(mask, count, height, width, before.device)
            # Use on-device where to avoid identity-resampling error at r=0 without an extra sampling pass.
            delta = torch.where(r > 0, sampled+residual-before, torch.zeros_like(before))*support
            return dict(delta=delta, flow=flow, residual=residual, visible=before+delta)

    def export(self):
        """Export only the rad head; the trainer records source weights, configuration, and progress."""
        return dict(format=RAD_FORMAT, config=dict(self.config),
                    rad={key: value.detach().cpu().clone() for key, value in self.state_dict().items()})

    @classmethod
    def from_export(cls, payload, device='cpu'):
        if payload.get('format') != RAD_FORMAT:
            raise ValueError('Expected v6.6 mouth rad-head weights')
        model = cls(payload['config'])
        model.load_state_dict(payload['rad'], strict=True)
        return model.to(device)
