import torch
from ref2light.common.timing import native_frame_gpu as original_transfer


class NativeRGBTransfer:
    """Alternate between two borrowed output buffers, reusing each after two calls."""
    def __init__(self):
        self.signature = None
        self.buffers = []
        self.index = 0

    @torch.no_grad()
    def __call__(self, tensor, size):
        if tuple(tensor.shape) != (3, size[1], size[0]):
            raise ValueError('Expected an RGB tensor at the native display resolution')
        if tensor.device.type != 'cuda':
            return original_transfer(tensor, size)
        signature = (tuple(size), tensor.device)
        if signature != self.signature:
            self.buffers = [torch.empty((size[1], size[0], 3), dtype=torch.uint8,
                                       device='cpu', pin_memory=True) for _ in range(2)]
            self.signature, self.index = signature, 0
        # Preserve operation order, rounding, channel layout, and interlacing calibration.
        packed = tensor.detach().float().clamp(0, 1).mul(255.).add(.5).to(torch.uint8)
        packed = packed.permute(1, 2, 0).contiguous()
        target = self.buffers[self.index]
        target.copy_(packed, non_blocking=True)
        torch.cuda.current_stream(tensor.device).synchronize()
        self.index = 1-self.index
        return target.numpy()
