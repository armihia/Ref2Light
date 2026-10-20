from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def square_rgb(path, side=512):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGBA')
        # Composite transparent pixels over black before encoding.
        rgb = Image.new('RGBA', image.size, (0, 0, 0, 255))
        rgb.alpha_composite(image)
        image = rgb.convert('RGB')
    scale = side / max(image.size)
    image = image.resize(tuple(max(1, round(n * scale)) for n in image.size), Image.Resampling.LANCZOS)
    output = Image.new('RGB', (side, side))
    output.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return output


def similarity(regions, target):
    boxes = np.asarray(regions, dtype=np.float64)
    if boxes.shape != (3, 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError('Expected valid eye and mouth coordinates on the ordinary square reference')
    # Use the eyes for scale so mouth opening does not change the character size.
    source = ((boxes[:, :2] + boxes[:, 2:])/2)[:2]
    target = np.asarray(target, dtype=np.float64)[:2]
    if target.shape != (2, 2) or not np.isfinite(target).all():
        raise ValueError('Expected valid target coordinates for both eyes')
    a, b = source-source.mean(0), target-target.mean(0)
    scale = float((a*b).sum()/max(float((a*a).sum()), 1e-12))
    if not .15 <= scale <= 6:
        raise ValueError(f'Landmark localization produced an invalid scale {scale:.3f}; provide corrected annotations')
    return scale, target.mean(0)-scale*source.mean(0)


def foreground_bounds(image):
    """Find visible bounds on a black or transparent reference, retaining all non-black pixels."""
    mask = np.asarray(image.convert('RGB')).max(-1) > 6
    yy, xx = np.where(mask)
    if not len(xx):
        return np.array([0., 0., 1., 1.])
    return np.array([xx.min()/image.width, yy.min()/image.height,
                     (xx.max()+1)/image.width, (yy.max()+1)/image.height])


def contained_transform(image, scale, shift, margin=.008, anchor=None):
    """Solve scale and translation together so the reference point reaches its target after scale limiting."""
    bounds = foreground_bounds(image)
    source = (bounds[:2]+bounds[2:])/2 if anchor is None else np.asarray(anchor, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0 or shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError('Scale and translation must be finite, with positive scale')
    if source.shape != (2,) or not np.isfinite(source).all() or (source < bounds[:2]).any() or (source > bounds[2:]).any():
        raise ValueError('The alignment reference point lies outside the visible character bounds')
    destination = np.clip(scale*source+shift, margin, 1-margin)
    # Constrain all four bounds around the reference point to avoid shifting the face after limiting scale.
    distances = np.r_[source-bounds[:2], bounds[2:]-source]
    room = np.r_[destination-margin, 1-margin-destination]
    positive = distances > 1e-10
    fitted = min(float(scale), float((room[positive]/distances[positive]).min()))
    if fitted <= 0:
        raise ValueError('The reference point and bounds admit no positive scale that preserves visible content')
    return fitted, destination-fitted*source, bounds


def transform_image(image, scale, shift):
    w, h = image.size
    return image.transform(image.size, Image.Transform.AFFINE,
        (1/scale, 0, -shift[0]*w/scale, 0, 1/scale, -shift[1]*h/scale),
        Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))


def align_image(image, regions, target):
    scale, shift = similarity(regions, target)
    requested = dict(scale=scale, translation_normalized=shift.tolist())
    boxes = np.asarray(regions, dtype=np.float64)
    source = ((boxes[:2, :2]+boxes[:2, 2:])/2).mean(0)
    goal = np.asarray(target, dtype=np.float64)[:2].mean(0)
    # Apply only mild framing alignment without forcing identical eye spacing across characters.
    scale = float(np.clip(1+.25*(scale-1), .94, 1.06))
    destination = source+np.clip(.25*(goal-source), -.025, .025)
    shift = destination-scale*source
    try:
        scale, shift, bounds = contained_transform(image, scale, shift, anchor=source)
    except ValueError as exc:
        return image.copy(), dict(alignment_applied=False, reason=str(exc), scale=1.,
                                  translation_normalized=[0., 0.], requested=requested)
    return transform_image(image, scale, shift), dict(scale=scale, translation_normalized=shift.tolist(),
        requested=requested, foreground_bounds=bounds.tolist(), alignment_uses='bounded_eye_midpoint',
        source_anchor=source.tolist(), target_anchor=destination.tolist(), alignment_applied=True,
        visible_content_preserved=True)


def canvas_array(image):
    from ref2light.common.canvas import canonical_ref
    return canonical_ref(np.asarray(image, dtype=np.uint8).copy())


def prepare_inference(path, target, landmark_model, destination, align=False, layout=None):
    """Run detection only when changing the reference, outside the per-frame path."""
    from ref2light.common.landmarks import detect_reference
    from ref2light.common.io import write_json
    from ref2light.reference.planning import read_json, reference_pose
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    image = square_rgb(path)
    square = destination/'reference_square.png'
    image.save(square)
    report = dict(source=str(Path(path).resolve()), aligned=False,
                  scope='Adjust 2D framing while preserving visible bounds; the model may still predict unseen body content',
                  source_pose=None, default_pose_semantics='Absolute teacher-pose parameters rather than motion relative to an arbitrary reference')
    sidecar = Path(path).with_suffix('.json')
    if sidecar.is_file():
        metadata = read_json(sidecar)
        if metadata.get('schema') == 'geometry_face_12d_extra_reference_v1':
            report['source_pose'] = reference_pose(metadata)
    if align:
        if layout is None:
            layout = detect_reference(square, landmark_model)
        image, transform = align_image(image, layout['regions'], target)
        report.update(aligned=transform.get('alignment_applied', False), alignment_requested=True,
                      transform=transform, regions=layout['regions'])
    output = destination/'reference_input.png'
    image.save(output)
    write_json(destination/'reference.json', report)
    return image, report
