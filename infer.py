import argparse
import json
from pathlib import Path
from ref2light import DEFAULT_MODEL,LANDMARK_MODEL
from ref2light.common.controls import POSE_KEYS,NEUTRAL,validate_pose


def arguments(argv=None):
    p=argparse.ArgumentParser(description='Generate a display-native PNG from a single reference image and pose.')
    p.add_argument('--model',default=str(DEFAULT_MODEL))
    p.add_argument('--ref',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--set',action='append',default=[],metavar='NAME=VALUE')
    p.add_argument('--pose',help='JSON file containing a 14-dimensional array or a dictionary of named parameters')
    p.add_argument('--mouth-amount',type=float,help='Combined mouth opening: 0 closed, 0.5 small O, 1 wide open')
    p.add_argument('--output-boxes')
    p.add_argument('--brow-boxes')
    p.add_argument('--align',action='store_true')
    p.add_argument('--landmark-model',default=str(LANDMARK_MODEL))
    return p.parse_args(argv)


def pose_values(a):
    values=dict(NEUTRAL)
    if a.pose:
        data=json.loads(Path(a.pose).read_text(encoding='utf-8-sig'))
        if isinstance(data,list):values=dict(zip(POSE_KEYS,validate_pose(data)))
        elif isinstance(data,dict):
            if set(data)-set(POSE_KEYS):raise ValueError('Pose JSON contains unknown parameters')
            values.update(data)
        else:raise ValueError('Pose JSON must contain an array or a dictionary')
    for item in a.set:
        key,number=item.split('=',1)
        if key not in POSE_KEYS:raise ValueError('Unknown parameter: '+key)
        values[key]=float(number)
    if a.mouth_amount is not None:
        from ref2light.ui.app import complete_mouth_values
        values.update(complete_mouth_values(a.mouth_amount))
    return validate_pose([values[k] for k in POSE_KEYS])


def main():
    import torch
    from PIL import Image
    from ref2light.runtime.session import Session
    from ref2light.common.io import write_json
    from ref2light.ui.transfer import NativeRGBTransfer
    a=arguments();raw=pose_values(a)
    output=Path(a.out).resolve()
    if output.suffix.lower()!='.png':raise ValueError('Save native interlaced images as lossless PNG files')
    if output.exists():raise FileExistsError('Use a new output filename: '+str(output))
    torch.set_num_threads(2)
    session=Session(a.model,a.ref,a.device,a.landmark_model,output.parent/(output.stem+'_session'),a.align,
        output_boxes=a.output_boxes,local_graphs=False,body_graphs=False)
    if a.brow_boxes:session.set_brow_boxes(a.brow_boxes)
    with torch.inference_mode():
        frame=NativeRGBTransfer()(session.render(raw),session.size)
    Image.fromarray(frame).save(output)
    write_json(output.with_suffix('.json'),dict(session.metadata(),raw_pose=list(raw)))
    print('Saved native interlaced image: ',output)


if __name__=='__main__':main()
