from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class _Entry:
    graph: object
    inputs: dict
    outputs: dict


class LocalGraphRunner:
    """Cache CUDA Graphs by branch and input shape, updating pose through FP32 CUDA tensors.
    Replay overwrites output buffers; clear the cache after changing model device or precision."""
    def __init__(self, model, *, use_amp=True, warmup=3):
        if not isinstance(model, nn.Module):
            raise TypeError('The local graph runner requires a torch.nn.Module')
        if type(use_amp) is not bool:
            raise TypeError('use_amp must be a boolean')
        if type(warmup) is not int or warmup < 1:
            raise ValueError('The number of warm-up iterations must be a positive integer')
        self.model = model
        self.use_amp = use_amp
        self.warmup = warmup
        self.entries = {}
        self.replays = 0
        self._validate_frozen()

    def _validate_frozen(self):
        if any(module.training for module in self.model.modules()):
            raise ValueError('The local graph runner requires a model in eval mode')
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError('The local graph runner requires all model parameters to be frozen')

    @property
    def graph_count(self):
        return len(self.entries)

    def clear(self):
        self.entries.clear()
        self.replays = 0

    def _prepare(self, group, canonical, reference, pose, mask, identity, color):
        if group not in ('eyes', 'mouth', 'gaze'):
            raise ValueError('The local branch must be eyes, mouth, or gaze')
        if not torch.is_tensor(canonical) or canonical.ndim != 4 or canonical.shape[1] != 3:
            raise ValueError('canonical must be a K x 3 x H x W tensor')
        if not torch.is_tensor(reference) or reference.shape != canonical.shape:
            raise ValueError('reference must have the same shape as canonical')
        if not canonical.is_floating_point() or not reference.is_floating_point():
            raise ValueError('Ordinary RGB patches must be floating-point tensors')
        if not torch.is_tensor(pose):
            raise TypeError('The graph pose must be a CUDA tensor supplied by the caller, not a Python list')
        if pose.shape not in ((14,), (1, 14)) or pose.dtype != torch.float32:
            raise ValueError('The graph pose must be FP32 with shape [14] or [1, 14]')
        if group == 'mouth' and getattr(self.model, 'config', {}).get('mouth_identity_condition', False):
            if not torch.is_tensor(identity) or identity.shape != (1, 512):
                raise ValueError('Mouth identity conditioning requires an explicit 1 x 512 identity tensor; zero substitution is unsupported')
            if identity.device != canonical.device or not identity.is_floating_point():
                raise ValueError('Mouth identity must be a floating-point tensor on the canonical device')
        if canonical.device.type != 'cuda':
            raise ValueError('LocalGraphRunner requires CUDA; call the local model directly on CPU')
        self._validate_frozen()
        count, _, height, width = canonical.shape
        if count < 1 or min(height, width) < 8:
            raise ValueError('Expected at least one anchor view and patches of at least 8 x 8')
        device = canonical.device
        if reference.device != device or pose.device != device:
            raise ValueError('Reference patches, pose, and canonical must share a CUDA device')
        model_tensors = list(self.model.parameters())+list(self.model.buffers())
        if any(value.device != device for value in model_tensors):
            raise ValueError('Local-model parameters, buffers, and inputs must share a CUDA device')
        mask = canonical.new_ones(count, 1, height, width) if mask is None else mask
        identity = canonical.new_zeros(1, 512) if identity is None else identity
        if not torch.is_tensor(mask) or mask.shape != (count, 1, height, width):
            raise ValueError('mask must be a K x 1 x H x W tensor')
        if not torch.is_tensor(identity) or identity.shape != (1, 512):
            raise ValueError('identity must be a 1 x 512 tensor')
        values = dict(canonical=canonical, reference=reference, pose=pose.reshape(1, 14),
                      mask=mask, identity=identity)
        if group == 'mouth' and hasattr(self.model, '_patch_color') and color is None:
            raise ValueError('Precompute v5 mouth colour outside graph capture')
        if color is not None:
            if group != 'mouth' or not hasattr(self.model, '_patch_color'):
                raise ValueError('color is only supported by the v5 mouth branch')
            if not torch.is_tensor(color) or color.shape != (count, 3, 1, 1):
                raise ValueError('color must be a K x 3 x 1 x 1 tensor')
            values['color'] = color
        if any(value.device != device or not value.is_floating_point() for value in values.values()):
            raise ValueError('All static graph inputs must be floating-point tensors on the same device')
        key = (group, self.use_amp, tuple((name, tuple(value.shape), value.dtype, value.device)
                                       for name, value in values.items()))
        return key, values

    def _evaluate(self, group, values):
        extra = {'color': values['color']} if 'color' in values else {}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=self.use_amp, cache_enabled=False):
            result = self.model(group, values['canonical'], values['reference'], values['pose'],
                                mask=values['mask'], identity=values['identity'], **extra)
        if not isinstance(result, dict) or not torch.is_tensor(result.get('delta')):
            raise ValueError('The local model must return a dictionary containing a delta tensor')
        if result['delta'].shape != values['canonical'].shape:
            raise ValueError('Local delta must have the same shape as canonical')
        return result

    def _capture(self, group, values):
        device = values['canonical'].device
        static = {name: value.detach().clone() for name, value in values.items()}
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self._evaluate(group, static)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # Propagate capture failures instead of reporting a successful accelerated path.
        with torch.cuda.graph(graph, stream=stream):
            outputs = self._evaluate(group, static)
        current.wait_stream(stream)
        return _Entry(graph=graph, inputs=static, outputs=outputs)

    @torch.no_grad()
    def forward(self, group, canonical, reference, pose, mask=None, identity=None, *, color=None):
        key, values = self._prepare(group, canonical, reference, pose, mask, identity, color)
        if key not in self.entries:
            self.entries[key] = self._capture(group, values)
        entry = self.entries[key]
        for name, current in values.items():
            entry.inputs[name].copy_(current)
        entry.graph.replay()
        self.replays += 1
        return entry.outputs

    __call__ = forward
