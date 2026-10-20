import torch
from torch.nn import functional as F
from ref2light.rad.model import _fp32, _mask


def extract_mouth_features(legacy_local, canonical, reference, pose, mask=None, identity=None, *, zero_rad=True):
    """Reuse mouth features while allowing gradients through the closure head and input features."""

    from ref2light.local.local import LocalControls
    from ref2light.joint.pose_mouth import PoseLocal
    if not isinstance(legacy_local, (LocalControls,PoseLocal)):
        raise TypeError('Unsupported mouth module type')
    if canonical.ndim != 4 or canonical.shape[1] != 3 or canonical.shape != reference.shape:
        raise ValueError('canonical and reference must both have shape [K, 3, H, W]')
    if canonical.device != reference.device or min(canonical.shape[-2:]) < 8:
        raise ValueError('Both patches must share a device and have height and width of at least 8')
    if not canonical.is_floating_point() or not reference.is_floating_point():
        raise ValueError('Ordinary RGB patches must use floating-point tensors')
    count, _, height, width = canonical.shape
    raw = legacy_local._pose(pose, count, canonical.device)
    # Use basic slicing to avoid CPU index allocation inside CUDA Graphs; preserve the raw pose.
    if zero_rad:
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
    if hasattr(legacy_local,'closed_adapter'):
        correction = legacy_local.closed_adapter(feature,canonical,reference,raw)
        predicted = torch.cat((predicted[:,:5]+correction,predicted[:,5:]),1)
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
        return dict(features=feature, legacy_visible=base+delta, legacy_delta=delta,
                    flow=flow,residual=residual,gate=torch.ones_like(raw[:,6])[:,None,None,None])

