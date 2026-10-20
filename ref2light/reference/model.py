from contextlib import nullcontext
import torch
from ref2light.models.native import NativeGenerator
from ref2light.models.face_adapter import amp
from ref2light.models.precision import configure_precision


def create_base(stats, architecture, state, device):
    configure_precision()
    base = NativeGenerator(stats['calibration'], **architecture)
    base.load_state_dict(state, strict=True)
    return base.to(device).eval()


def encode(base, ref, train_encoder=False):
    with nullcontext() if train_encoder else torch.no_grad():
        with amp(ref.device, ref.device.type == 'cuda'):
            return tuple(base.encode_reference(ref))


def predict_anchors(base, ref, features, raw, stats):
    mean = ref.new_tensor(stats['mean'])
    std = ref.new_tensor(stats['std'])
    pose = (torch.as_tensor(raw, device=ref.device, dtype=torch.float32).reshape(1, 14)-mean)/std
    with amp(ref.device, ref.device.type == 'cuda'):
        return base.render_anchors_from_features(ref, features, pose)[0]


