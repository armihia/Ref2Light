import argparse
import math
import multiprocessing as mp
from pathlib import Path
from ref2light import ROOT,DEFAULT_MODEL,LANDMARK_MODEL


def arguments(argv=None):
    p=argparse.ArgumentParser(description='Launch reference switching, 14-axis controls, full mouth opening, and camera capture.')
    p.add_argument('--model',default=str(DEFAULT_MODEL))
    local=ROOT/'examples/local/ref1.png'
    p.add_argument('--ref',default=str(local) if local.is_file() else None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output-boxes')
    p.add_argument('--brow-boxes')
    p.add_argument('--detector-recovery',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--roi-tracking',choices=('source_uv','neutral_fixed'),default='source_uv')
    p.add_argument('--local-graphs',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--body-graphs',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--display-backend',choices=('nvidia','software','accelerated'),default='nvidia')
    p.add_argument('--fps',type=float,default=60.)
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--body-mode',choices=('head_follow','pose','manual'),default='head_follow')
    p.add_argument('--landmark-model',default=str(LANDMARK_MODEL))
    p.add_argument('--align',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--save-root',default=str(ROOT/'runs/sessions'))
    a=p.parse_args(argv)
    if not a.ref:p.error('Specify a reference image with --ref')
    if any(not math.isfinite(v) or v<=0 for v in (a.fps,)) or a.camera<0:
        p.error('Frame and log rates must be finite and positive; camera index must be non-negative')
    return a


def main(diagnostics_factory=None):
    a=arguments()
    for path in (a.model,a.ref,a.landmark_model):
        if not Path(path).is_file():raise FileNotFoundError('Required file does not exist: '+str(path))
    from ref2light.ui.app import gui
    gui(a, diagnostics_factory=diagnostics_factory)


