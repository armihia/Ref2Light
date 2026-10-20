from contextlib import nullcontext


import torch


from torch import nn


from torch.nn import functional as F


LOCAL_FORMAT = 'companion1_v65_local_patch_controls'
AXES = {'eyes': (6,), 'mouth': (7, 8), 'gaze': (6, 12, 13)}


def _groups(channels):
    return next(n for n in (8, 4, 2, 1) if channels % n == 0)


class _Block(nn.Module):
    """Two convolutions share a local control vector with small FiLM adjustments."""

    def __init__(self, cin, cout, condition):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.film = nn.Linear(condition, 2 * cout)

    def forward(self, x, condition):
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        return F.silu(x * (1 + .25 * torch.tanh(gamma)[..., None, None])
                      + beta[..., None, None])


class _ZeroIdentityProjection(nn.Linear):
    """Zero-initialize without consuming random numbers used by other branches."""
    def __init__(self, condition):
        super().__init__(512, condition, bias=False)

    def reset_parameters(self):
        nn.init.zeros_(self.weight)


class _Branch(nn.Module):
    def __init__(self, axes, width, condition, output_channels=5, identity_condition=False):
        super().__init__()
        self.axes = tuple(axes)
        self.condition = nn.Sequential(nn.Linear(len(axes), condition), nn.SiLU(),
                                       nn.Linear(condition, condition), nn.SiLU())
        self.down0 = _Block(9, width, condition)
        self.down1 = _Block(width, width * 2, condition)
        self.down2 = _Block(width * 2, width * 3, condition)
        self.up1 = _Block(width * 5, width * 2, condition)
        self.up0 = _Block(width * 3, width, condition)
        self.output = nn.Conv2d(width, output_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.identity_norm = self.identity_projection = None
        if identity_condition:
            self.enable_identity_condition()

    def enable_identity_condition(self):
        if self.identity_projection is not None:
            return
        parameter = self.output.weight
        self.identity_norm = nn.LayerNorm(512, elementwise_affine=False).to(device=parameter.device, dtype=parameter.dtype)
        self.identity_projection = _ZeroIdentityProjection(self.condition[-2].out_features).to(device=parameter.device, dtype=parameter.dtype)
        self.identity_projection.requires_grad_(parameter.requires_grad)
        self.identity_norm.train(self.training)
        self.identity_projection.train(self.training)

    def forward(self, features, raw_pose, identity=None):
        condition = self.condition(torch.stack([raw_pose[:, index] for index in self.axes], dim=1).to(features.dtype))
        if self.identity_projection is not None:
            condition = condition+self.identity_projection(self.identity_norm(identity).to(condition.dtype))
        x0 = self.down0(features, condition)
        x1 = self.down1(F.avg_pool2d(x0, 2), condition)
        x2 = self.down2(F.avg_pool2d(x1, 2), condition)
        x = self.up1(torch.cat((F.interpolate(x2, size=x1.shape[-2:], mode='bilinear',
                                            align_corners=False), x1), dim=1), condition)
        x = self.up0(torch.cat((F.interpolate(x, size=x0.shape[-2:], mode='bilinear',
                                            align_corners=False), x0), dim=1), condition)
        return self.output(x)


class LocalControls(nn.Module):
    """Process K ordinary-view patches and 14 raw controls with three independent branches.
    delta includes masking and gating; visible = canonical + delta, with normalized displacement in flow."""

    def __init__(self, config=None):
        super().__init__()
        values = {'width': 32, 'condition': 64, 'flow_limit': .25, 'rgb_limit': .5,
                  'mouth_endpoints': False, 'mouth_endpoint_open': .75, 'mouth_open_in_features': True,
                  'mouth_identity_condition': False}
        if config is not None:
            unknown = set(config) - set(values)
            if unknown:
                raise ValueError('Unsupported local-network configuration: ' + ', '.join(sorted(unknown)))
            values.update(config)
        if (not isinstance(values['width'], int) or values['width'] < 4
                or not isinstance(values['condition'], int) or values['condition'] < 4):
            raise ValueError('width and condition must be integers of at least 4')
        if not isinstance(values['mouth_endpoints'], bool):
            raise ValueError('mouth_endpoints must be a boolean')
        if not isinstance(values['mouth_open_in_features'], bool):
            raise ValueError('mouth_open_in_features must be a boolean')
        if not isinstance(values['mouth_identity_condition'], bool):
            raise ValueError('mouth_identity_condition must be a boolean')
        if not values['mouth_open_in_features'] and not values['mouth_endpoints']:
            raise ValueError('Disabling mouth-opening feature conditioning requires mouth_endpoints')
        for key in ('flow_limit', 'rgb_limit', 'mouth_endpoint_open'):
            if not 0 < float(values[key]) <= 1:
                raise ValueError(key + ' must be in (0, 1]')
            values[key] = float(values[key])
        self.config = dict(values)
        self.branches = nn.ModuleDict({name: _Branch(axes, values['width'], values['condition'],
                                                   10 if name == 'mouth' and values['mouth_endpoints'] else 5,
                                                   name == 'mouth' and values['mouth_identity_condition'])
                                       for name, axes in AXES.items()})

    def _pose(self, pose, anchors, device):
        if not isinstance(pose, torch.Tensor):
            pose = torch.as_tensor(pose, device=device, dtype=torch.float32)
        if pose.device != device:
            pose = pose.to(device=device)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)
        if pose.ndim != 2 or pose.shape[1] != 14 or pose.shape[0] not in (1, anchors):
            raise ValueError('pose must contain raw controls with shape [14], [1, 14], or [K, 14]')
        return pose.float().expand(anchors, -1)

    def forward(self, group, canonical, reference, pose, mask=None, identity=None):
        # Extract appearance from both RGB patches; optional mouth conditioning adds global identity features.
        if group not in self.branches:
            raise ValueError('The local branch must be eyes, mouth, or gaze')
        if canonical.ndim != 4 or canonical.shape[1] != 3 or canonical.shape != reference.shape:
            raise ValueError('canonical and reference must both have shape [K, 3, H, W]')
        if canonical.device != reference.device or min(canonical.shape[-2:]) < 8:
            raise ValueError('Both patches must share a device and have height and width of at least 8')
        if not canonical.is_floating_point() or not reference.is_floating_point():
            raise ValueError('Both patches must be floating-point RGB tensors')
        count, _, height, width = canonical.shape
        raw = self._pose(pose, count, canonical.device)
        ys = torch.linspace(-1, 1, height, device=canonical.device, dtype=torch.float32)
        xs = torch.linspace(-1, 1, width, device=canonical.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        zz = (torch.linspace(-1, 1, count, device=canonical.device, dtype=torch.float32)
              if count > 1 else torch.zeros(1, device=canonical.device))
        coordinates = torch.stack((xx, yy)).unsqueeze(0).expand(count, -1, -1, -1)
        coordinates = torch.cat((coordinates, zz[:, None, None, None].expand(-1, 1, height, width)), 1)
        features = torch.cat((canonical, reference, coordinates.to(canonical.dtype)), dim=1)
        feature_pose = raw
        if group == 'mouth' and not self.config['mouth_open_in_features']:
            # Both endpoints share image and mouth-curve conditioning; retain raw controls for opening interpolation.
            # Use on-device slicing and zeros_like to avoid CPU indices inside CUDA Graphs.
            feature_pose = torch.cat((raw[:, :7], torch.zeros_like(raw[:, 7:8]), raw[:, 8:]), dim=1)
        if group == 'mouth' and self.config['mouth_identity_condition']:
            if not torch.is_tensor(identity) or identity.shape != (1, 512):
                raise ValueError('Mouth identity conditioning requires a globally encoded 1 x 512 tensor')
            if identity.device != canonical.device or not identity.is_floating_point():
                raise ValueError('identity must be a floating-point tensor on the patch device')
            predicted = self.branches[group](features, feature_pose, identity)
        else:
            predicted = self.branches[group](features, feature_pose)

        # Allow mixed-precision convolutions while keeping displacement, resampling, and RGB composition in FP32.
        context = torch.autocast(device_type=canonical.device.type, enabled=False) \
            if canonical.device.type in ('cpu', 'cuda') else nullcontext()
        with context:
            base = canonical.float()
            predicted = predicted.float()
            if group == 'mouth' and self.config['mouth_endpoints']:
                # Predict flow2+RGB3 independently for closed and open endpoints, then interpolate fields.
                # Endpoint supervision targets separate output channels; hidden layers retain FiLM and mouthCurve conditioning.
                closed, opened_fields = predicted.chunk(2, dim=1)
                t = (raw[:, 7]/self.config['mouth_endpoint_open']).clamp(0, 1)
                predicted = torch.lerp(closed, opened_fields, t[:, None, None, None])
            flow = self.config['flow_limit'] * torch.tanh(predicted[:, :2].float())
            residual = self.config['rgb_limit'] * torch.tanh(predicted[:, 2:].float())
            grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(count, -1, -1, -1)
            sampled = F.grid_sample(base, grid + flow.permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='border', align_corners=True)
            candidate = sampled + residual
            opened = raw[:, 6].clamp(0, 1)
            if group == 'eyes':
                gate = 1 - opened
            elif group == 'gaze':
                gate = raw[:, 12:14].abs().sum(dim=1).clamp(0, 1) * opened
            else:
                gate = torch.ones_like(opened)
            if mask is None:
                support = torch.ones((1, 1, height, width), device=base.device)
            else:
                support = torch.as_tensor(mask, dtype=torch.float32, device=base.device)
                if support.ndim == 2:
                    support = support[None, None]
                elif support.ndim == 3:
                    support = support.unsqueeze(1)
                if (support.ndim != 4 or support.shape[1] != 1
                        or support.shape[0] not in (1, count) or support.shape[-2:] != (height, width)):
                    raise ValueError('mask must have shape [H, W], [1/K, H, W], or [1/K, 1, H, W]')
                support = support.clamp(0, 1)
            delta = (candidate - base) * support * gate[:, None, None, None]
            return {'delta': delta, 'flow': flow, 'residual': residual,
                    'visible': base + delta, 'gate': gate[:, None, None, None]}

    def export(self):
        """Export local parameters without copying or modifying the full-body model."""
        return {'format': LOCAL_FORMAT, 'config': dict(self.config),
                'local': {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}}

    @classmethod
    def from_export(cls, payload, device='cpu'):
        if payload.get('format') != LOCAL_FORMAT:
            raise ValueError('Expected v6.5 local-control weights')
        model = cls(payload['config'])
        model.load_state_dict(payload['local'], strict=True)
        return model.to(device)

