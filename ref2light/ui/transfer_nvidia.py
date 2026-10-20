import numpy as np
import torch
from ref2light.ui.transfer import NativeRGBTransfer


class DisplayFrame(np.ndarray):
    """Expose an RGBA display view of the same pinned RGB allocation to avoid per-frame CPU conversion."""


class NvidiaTransfer:
    def __init__(self):
        self.signature=None
        self.index=0
        self.fallback=NativeRGBTransfer()

    @torch.no_grad()
    def __call__(self,tensor,size):
        if tensor.device.type!='cuda':return self.fallback(tensor,size)
        if tuple(tensor.shape)!=(3,size[1],size[0]):raise ValueError('Expected a native-resolution RGB output tensor')
        signature=(tuple(size),tensor.device)
        if signature!=self.signature:
            self.gpu=torch.empty((size[1],size[0],4),device=tensor.device,dtype=torch.uint8)
            self.gpu[:,:,3].fill_(255)
            self.buffers=[torch.empty(self.gpu.shape,dtype=torch.uint8,pin_memory=True) for _ in range(2)]
            self.frames=[]
            for buffer in self.buffers:
                rgba=buffer.numpy()
                rgb=rgba[:,:,:3].view(DisplayFrame);rgb.rgba=rgba
                self.frames.append(rgb)
            self.signature,self.index=signature,0
        rgb=tensor.detach().float().clamp(0,1).mul(255.).add(.5).to(torch.uint8)
        self.gpu[:,:,:3].copy_(rgb.permute(1,2,0))
        self.buffers[self.index].copy_(self.gpu,non_blocking=True)
        torch.cuda.current_stream(tensor.device).synchronize()
        result=self.frames[self.index]
        self.index=1-self.index
        return result
