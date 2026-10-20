from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw


EYE_A = (33,7,163,144,145,153,154,155,133,246,161,160,159,158,157,173)

EYE_B = (263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398)

LIPS = (61,146,91,181,84,17,314,405,321,375,291,185,40,39,37,0,267,269,270,409)

def valid_layout(layout,reference=None):
    if layout.get('coordinate_space')!='original_square_reference_normalized_xy':
        raise ValueError('Annotations must use normalized coordinates on the original square reference')
    boxes = np.asarray(layout['regions'],dtype=np.float32)
    if boxes.shape!=(3,4) or not np.isfinite(boxes).all():
        raise ValueError('Expected finite boxes for the screen-left eye, screen-right eye, and mouth')
    if not ((boxes>=0).all() and (boxes<=1).all() and (boxes[:,2:]>boxes[:,:2]).all()):
        raise ValueError('Annotation boxes must lie within the reference image and have positive area')
    if reference and Path(layout['reference']).resolve()!=Path(reference).resolve():
        raise ValueError('The annotation reference does not match the current reference image')
    return layout

def region_box(points,factor,min_height_ratio):
    lo,hi = points.min(0),points.max(0)
    centre = (lo+hi)/2
    size = hi-lo
    size[1] = max(size[1],size[0]*min_height_ratio)
    size = np.maximum(size,1/1024)*factor
    return np.r_[np.maximum(0,centre-size/2),np.minimum(1,centre+size/2)].tolist()

def detect_reference(reference,model):
    import mediapipe as mp
    reference,model = Path(reference).resolve(),Path(model).resolve()
    if not model.is_file():
        raise FileNotFoundError('Missing face landmark model: '+str(model))
    with Image.open(reference) as source:
        if source.width!=source.height:
            raise ValueError('The reference image must be square; do not stretch it before annotation')
        image = source.convert('RGB')
    w,h = image.size
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model),delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=1)
    attempts = []
    # Retry the central upper-body region after full-image detection fails; map detections back to the original image.
    crops = [('whole',(0,0,w,h)),('upper_centre',(w//4,0,3*w//4,h//2))]
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for name,(x0,y0,x1,y1) in crops:
            array = np.asarray(image.crop((x0,y0,x1,y1)).resize((512,512))).copy()
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=array))
            attempts.append(dict(crop=name,faces=len(result.face_landmarks)))
            if not result.face_landmarks:
                continue
            points = np.array([[(p.x*(x1-x0)+x0)/w,(p.y*(y1-y0)+y0)/h]
                for p in result.face_landmarks[0]],dtype=np.float32)
            eyes = [region_box(points[list(ids)],1.5,.45) for ids in (EYE_A,EYE_B)]
            eyes.sort(key=lambda b:(b[0]+b[2])/2)
            layout = dict(reference=str(reference),coordinate_space='original_square_reference_normalized_xy',
                source='mediapipe_face_landmarker_v1_CPU',model=str(model),mediapipe_version=mp.__version__,
                regions=eyes+[region_box(points[list(LIPS)],1.5,.35)],
                landmarks=points.tolist(),attempts=attempts,
                interpretation='Approximate eye and mouth positions; inspect the ordinary reference preview for illustrated characters')
            return valid_layout(layout)
    raise ValueError('No face was detected; mark regions manually or provide compatible JSON annotations from another landmark model.')

def save_annotation(layout,dest):
    valid_layout(layout)
    dest = Path(dest)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'face_layout.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    image = Image.open(layout['reference']).convert('RGB')
    draw = ImageDraw.Draw(image)
    colors = ('#20e080','#20e080','#ff8060')
    for box,color in zip(layout['regions'],colors):
        x0,y0,x1,y1 = box
        draw.ellipse((x0*image.width,y0*image.height,x1*image.width,y1*image.height),outline=color,width=2)
    for point in layout.get('landmarks',[]):
        x,y = point[0]*image.width,point[1]*image.height
        draw.ellipse((x-1,y-1,x+1,y+1),fill='#60b0ff')
    image.save(dest/'ordinary_ref_annotation.png')
    return dest/'face_layout.json'
