import torch


def native_frame_gpu(tensor, size):
    """Convert to uint8 on the source device before transferring RGB pixels."""
    if tuple(tensor.shape) != (3, size[1], size[0]):
        raise ValueError('Expected an RGB tensor at the native display resolution')
    with torch.no_grad():
        packed = tensor.detach().float().clamp(0, 1).mul(255.).add(.5).to(torch.uint8)
        return packed.permute(1, 2, 0).contiguous().cpu().numpy()

