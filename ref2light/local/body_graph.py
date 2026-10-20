from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class _BodyEntry:
    graph: object
    inputs: dict
    outputs: tuple


class FrozenBodyGraph:
    """Cache fixed-shape reference features with normalized FP32 CUDA pose tensors.
    Replay overwrites returned tensors; clone persistent values and clear the cache after model changes."""
    def __init__(self, model, *, use_amp=True, warmup=3):
        if not isinstance(model, nn.Module) or not callable(getattr(model, 'render_anchors_from_features', None)):
            raise TypeError('The body graph requires a torch.nn.Module with render_anchors_from_features')
        if type(use_amp) is not bool:
            raise TypeError('use_amp must be a boolean')
        if type(warmup) is not int or warmup < 1:
            raise ValueError('The number of warm-up iterations must be a positive integer')
        self.model = model
        self.use_amp = use_amp
        self.warmup = warmup
        self.entry = None
        self.signature = None
        self.replays = 0
        self._validate_frozen()

    def _validate_frozen(self):
        if any(module.training for module in self.model.modules()):
            raise ValueError('The body graph requires a model in eval mode')
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError('The body graph requires all model parameters to be frozen')

    @property
    def graph_count(self):
        return int(self.entry is not None)

    def clear(self):
        self.entry = None
        self.signature = None
        self.replays = 0

    @staticmethod
    def _pose_tensor(pose, batch):
        if not torch.is_tensor(pose):
            raise TypeError('The normalized pose must be a CUDA tensor, not a Python list')
        allowed_shapes = ((batch, 14), (14,)) if batch == 1 else ((batch, 14),)
        if pose.dtype != torch.float32 or pose.shape not in allowed_shapes:
            raise ValueError('The normalized pose must be FP32 with shape B x 14; [14] is also accepted for B=1')
        return pose.reshape(batch, 14)

    def _prepare_inputs(self, ref, features, normalized_pose):
        if not torch.is_tensor(ref) or ref.ndim != 4 or ref.shape[1] != 3 or ref.shape[0] < 1:
            raise ValueError('ref must be a B x 3 x H x W tensor')
        if not isinstance(features, (tuple, list)) or len(features) != 6:
            raise ValueError('features must contain all six encoder feature levels')
        if any(not torch.is_tensor(value) or value.ndim != 4 or value.shape[0] != ref.shape[0]
               for value in features):
            raise ValueError('All six feature levels must have shape B x C x H x W and match the reference batch size')
        pose = self._pose_tensor(normalized_pose, ref.shape[0])
        values = {'ref': ref, **{f'feature_{index}': value for index, value in enumerate(features)}, 'pose': pose}
        if any(not value.is_floating_point() or min(value.shape) < 1 for value in values.values()):
            raise ValueError('Body-graph inputs must be non-empty floating-point tensors')
        if ref.device.type != 'cuda':
            raise ValueError('FrozenBodyGraph requires CUDA; call the decoder directly on CPU')
        if any(value.device != ref.device for value in values.values()):
            raise ValueError('The reference, features, and pose must share a CUDA device')
        self._validate_frozen()
        if any(value.device != ref.device for value in list(self.model.parameters())+list(self.model.buffers())):
            raise ValueError('Body-model parameters, buffers, and inputs must share a CUDA device')
        return values

    def _evaluate(self, values):
        features = tuple(values[f'feature_{index}'] for index in range(6))
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=self.use_amp, cache_enabled=False):
            result = self.model.render_anchors_from_features(values['ref'], features, values['pose'])
        if not isinstance(result, tuple) or len(result) != 2 or not torch.is_tensor(result[0]) or not isinstance(result[1], dict):
            raise ValueError('The body decoder must return (anchors Tensor, aux dict)')
        return result

    def _capture(self, values):
        device = values['ref'].device
        static = {name: value.detach().clone() for name, value in values.items()}
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                # Warm up the FP32 base sampling grids outside graph capture.
                self._evaluate(static)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self._evaluate(static)
        current.wait_stream(stream)
        return _BodyEntry(graph=graph, inputs=static, outputs=outputs)

    @torch.no_grad()
    def prepare(self, ref, features, normalized_pose):
        """Prepare a reference once, without recopying its features in later forward calls."""
        values = self._prepare_inputs(ref, features, normalized_pose)
        signature = tuple((name, tuple(value.shape), value.dtype, value.device) for name, value in values.items())
        if self.entry is not None and signature != self.signature:
            raise ValueError('Input shape, dtype, or device changed; call clear() and prepare() to rebuild the body graph')
        if self.entry is None:
            self.entry = self._capture(values)
            self.signature = signature
        else:
            for name, value in values.items():
                self.entry.inputs[name].copy_(value)
        return self

    @torch.no_grad()
    def forward(self, normalized_pose):
        if self.entry is None:
            raise RuntimeError('Call prepare(ref, features, normalized_pose) first')
        static_pose = self.entry.inputs['pose']
        pose = self._pose_tensor(normalized_pose, static_pose.shape[0])
        if pose.device != static_pose.device:
            raise ValueError('The per-frame pose must use the prepared body-graph device')
        self.entry.inputs['pose'].copy_(pose)
        self.entry.graph.replay()
        self.replays += 1
        return self.entry.outputs

    __call__ = forward
