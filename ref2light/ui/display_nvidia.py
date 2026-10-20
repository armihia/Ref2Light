import ctypes as C
from ctypes import wintypes as W
import os
import uuid
import numpy as np
import cv2
import pygame

ACTIVE_DISPLAY = {}
U=C.c_uint
P=C.c_void_p
HR=C.c_long


class GUID(C.Structure):
    _fields_=[('data',C.c_ubyte*16)]
    @classmethod
    def of(cls,value):return cls((C.c_ubyte*16).from_buffer_copy(uuid.UUID(value).bytes_le))


class LUID(C.Structure):
    _fields_=[('LowPart',U),('HighPart',C.c_long)]


class AdapterDesc(C.Structure):
    _fields_=[('Description',C.c_wchar*128),('VendorId',U),('DeviceId',U),('SubSysId',U),('Revision',U),
              ('DedicatedVideoMemory',C.c_size_t),('DedicatedSystemMemory',C.c_size_t),('SharedSystemMemory',C.c_size_t),
              ('AdapterLuid',LUID),('Flags',U)]


class Rational(C.Structure):_fields_=[('Numerator',U),('Denominator',U)]
class Mode(C.Structure):_fields_=[('Width',U),('Height',U),('RefreshRate',Rational),('Format',U),('ScanlineOrdering',U),('Scaling',U)]
class Samples(C.Structure):_fields_=[('Count',U),('Quality',U)]
class SwapDesc(C.Structure):
    _fields_=[('BufferDesc',Mode),('SampleDesc',Samples),('BufferUsage',U),('BufferCount',U),
              ('OutputWindow',W.HWND),('Windowed',W.BOOL),('SwapEffect',U),('Flags',U)]
class TextureDesc(C.Structure):
    _fields_=[('Width',U),('Height',U),('MipLevels',U),('ArraySize',U),('Format',U),('SampleDesc',Samples),
              ('Usage',U),('BindFlags',U),('CPUAccessFlags',U),('MiscFlags',U)]
class Mapped(C.Structure):_fields_=[('pData',P),('RowPitch',U),('DepthPitch',U)]


def call(obj,index,result,*args):
    table=C.cast(obj,C.POINTER(C.POINTER(P))).contents
    return C.WINFUNCTYPE(result,P,*args)(table[index])


def check(code,where):
    if code<0:raise RuntimeError(f'{where} failed: 0x{code & 0xffffffff:08x}')


def release(obj):
    if obj and obj.value:call(obj,2,U)(obj)


def factory():
    lib=C.WinDLL('dxgi')
    fn=lib.CreateDXGIFactory1
    fn.argtypes=[C.POINTER(GUID),C.POINTER(P)];fn.restype=HR
    result=P()
    check(fn(C.byref(GUID.of('770aae78-f26f-4dba-a829-253c83d1b387')),C.byref(result)),'Create DXGI factory')
    return result


def adapters():
    f=factory();result=[]
    try:
        for index in range(16):
            adapter=P()
            code=call(f,12,HR,U,C.POINTER(P))(f,index,C.byref(adapter))
            if code & 0xffffffff == 0x887a0002:break
            check(code,'Enumerate graphics adapters')
            try:
                desc=AdapterDesc()
                check(call(adapter,10,HR,C.POINTER(AdapterDesc))(adapter,C.byref(desc)),'Read adapter description')
                result.append(dict(index=index,name=desc.Description,vendor=desc.VendorId,
                    dedicated_bytes=desc.DedicatedVideoMemory,
                    luid=f'0x{desc.AdapterLuid.HighPart & 0xffffffff:08x}_0x{desc.AdapterLuid.LowPart:08x}'))
            finally:release(adapter)
    finally:release(f)
    return result


class NvidiaDisplay:
    def __init__(self,size,position=(0,0),backend='nvidia',hidden=False):
        if os.name!='nt':raise RuntimeError('The NVIDIA D3D11 display backend requires Windows')
        self.size=tuple(size)
        self.factory=self.adapter=self.device=self.context=self.swap=self.texture=None
        os.environ['SDL_VIDEO_WINDOW_POS']=f'{position[0]},{position[1]}'
        self.surface=pygame.display.set_mode(self.size,pygame.NOFRAME|(pygame.HIDDEN if hidden else 0))
        pygame.display.set_caption('Ref2Light native interlaced display')
        hwnd=pygame.display.get_wm_info()['window']
        options=[a for a in adapters() if a['vendor']==0x10de]
        if not options:raise RuntimeError('DXGI found no NVIDIA adapter; use --display-backend software if needed')
        choice=max(options,key=lambda a:a['dedicated_bytes'])
        self.name=choice['name']
        try:
            self.factory=factory();self.adapter=P()
            check(call(self.factory,12,HR,U,C.POINTER(P))(self.factory,choice['index'],C.byref(self.adapter)),'Select NVIDIA adapter')
            call(self.factory,8,HR,W.HWND,U)(self.factory,hwnd,2)
            desc=SwapDesc(Mode(size[0],size[1],Rational(0,1),28,0,0),Samples(1,0),0x20,2,hwnd,True,4,0)
            self.device,self.context,self.swap=P(),P(),P()
            feature=U()
            fn=C.WinDLL('d3d11').D3D11CreateDeviceAndSwapChain
            fn.argtypes=[P,U,P,U,C.POINTER(U),U,U,C.POINTER(SwapDesc),C.POINTER(P),C.POINTER(P),C.POINTER(U),C.POINTER(P)]
            fn.restype=HR
            check(fn(self.adapter,0,None,0,None,0,7,C.byref(desc),C.byref(self.swap),C.byref(self.device),
                     C.byref(feature),C.byref(self.context)),'Create NVIDIA display device')
            self.texture=P()
            check(call(self.swap,9,HR,U,C.POINTER(GUID),C.POINTER(P))(
                self.swap,0,C.byref(GUID.of('6f15aaf2-d208-4e89-9ab4-489535d34f9c')),C.byref(self.texture)),'Get native back buffer')
            self.rgba=np.empty((size[1],size[0],4),dtype=np.uint8)
            self.rgba[:,:,3]=255
            ACTIVE_DISPLAY.clear()
            ACTIVE_DISPLAY.update(backend='nvidia_d3d11',adapter=self.name,adapter_luid=choice['luid'],
                                  size=list(size),scaling=False,windows_compositor_gpu_not_forced=True)
            print('Native display device: ',ACTIVE_DISPLAY,flush=True)
        except BaseException:
            self.close();raise

    def draw(self,frame,present=True):
        if frame.shape!=(self.size[1],self.size[0],3) or frame.dtype!=np.uint8:
            raise ValueError('Display input must be native-resolution H x W x 3 uint8 without scaling')
        rgba=getattr(frame,'rgba',None)
        if rgba is None:
            cv2.cvtColor(frame,cv2.COLOR_RGB2RGBA,dst=self.rgba)
            rgba=self.rgba
        if rgba.shape!=(self.size[1],self.size[0],4) or not rgba.flags.c_contiguous:
            raise ValueError('The RGBA display buffer does not match the native layout')
        call(self.context,48,None,P,U,P,P,U,U)(self.context,self.texture,0,None,rgba.ctypes.data,self.size[0]*4,0)
        if present:check(call(self.swap,8,HR,U,U)(self.swap,0,0),'Present native frame')

    def readback(self):
        desc=TextureDesc()
        call(self.texture,10,None,C.POINTER(TextureDesc))(self.texture,C.byref(desc))
        if (desc.Width,desc.Height)!=self.size:raise RuntimeError('Back-buffer dimensions do not match the native calibration')
        desc.Usage=3;desc.BindFlags=0;desc.CPUAccessFlags=0x20000;desc.MiscFlags=0
        staging=P()
        check(call(self.device,5,HR,C.POINTER(TextureDesc),P,C.POINTER(P))(
            self.device,C.byref(desc),None,C.byref(staging)),'Create pixel-verification buffer')
        try:
            call(self.context,47,None,P,P)(self.context,staging,self.texture)
            mapped=Mapped()
            check(call(self.context,14,HR,P,U,U,U,C.POINTER(Mapped))(
                self.context,staging,0,1,0,C.byref(mapped)),'Read display buffer')
            try:
                data=np.ctypeslib.as_array((C.c_ubyte*(mapped.RowPitch*self.size[1])).from_address(mapped.pData))
                rgb=data.reshape(self.size[1],mapped.RowPitch)[:,:self.size[0]*4].reshape(self.size[1],self.size[0],4)[:,:,:3].copy()
            finally:call(self.context,15,None,P,U)(self.context,staging,0)
            return pygame.image.frombuffer(rgb,self.size,'RGB').copy()
        finally:release(staging)

    def close(self):
        for name in ('texture','swap','context','device','adapter','factory'):
            release(getattr(self,name,None));setattr(self,name,None)
