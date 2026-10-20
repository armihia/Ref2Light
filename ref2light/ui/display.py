import os
import pygame


class NativeDisplay:
    def __init__(self, size, position=(0,0), backend='accelerated', hidden=False):
        self.size,self.backend=tuple(size),backend
        self.window=self.renderer=self.texture=None
        if backend=='software':
            os.environ['SDL_VIDEO_WINDOW_POS']=f'{position[0]},{position[1]}'
            self.surface=pygame.display.set_mode(self.size,pygame.NOFRAME | (pygame.HIDDEN if hidden else 0))
            pygame.display.set_caption('Ref2Light native interlaced display')
            if self.surface.get_size()!=self.size:raise RuntimeError('Display dimensions do not match the calibration')
        elif backend=='accelerated':
            from pygame._sdl2.video import Window,Renderer,Texture
            os.environ['SDL_RENDER_SCALE_QUALITY']='0'
            self.window=Window('Ref2Light native interlaced display',size=self.size,position=position,
                               borderless=True,hidden=hidden,allow_highdpi=False)
            self.renderer=Renderer(self.window,accelerated=1,vsync=False)
            self.renderer.scale=(1.,1.)
            if tuple(self.window.size)!=self.size or tuple(self.renderer.get_viewport().size)!=self.size:
                self.window.destroy();raise RuntimeError('Render-target dimensions do not match the native resolution')
            self.texture=Texture(self.renderer,self.size,depth=32,streaming=True)
            self.texture.blend_mode=0
        else:raise ValueError('Display mode must be accelerated or software')

    def draw(self, frame, present=True):
        if frame.shape != (self.size[1],self.size[0],3) or str(frame.dtype)!='uint8':
            raise ValueError('Display input must be native-resolution H x W x 3 uint8')
        surface=pygame.image.frombuffer(frame,self.size,'RGB')
        if self.backend=='software':
            self.surface.blit(surface,(0,0))
            if present:pygame.display.flip()
        else:
            self.texture.update(surface)
            self.texture.draw(dstrect=(0,0,*self.size))
            if present:self.renderer.present()

    def readback(self):
        return self.renderer.to_surface() if self.renderer is not None else self.surface.copy()

    def close(self):
        self.texture=self.renderer=None
        if self.window is not None:
            self.window.destroy();self.window=None

