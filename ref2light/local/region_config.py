from collections.abc import Mapping
from copy import deepcopy
import math


def _finite_number(value, label):
    if isinstance(value, bool):
        raise ValueError(label + ' must be finite')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(label + ' must be finite') from error
    if not math.isfinite(result):
        raise ValueError(label + ' must be finite')
    return result


def _rectangle(values, label):
    try:
        if len(values) != 4:
            raise ValueError(label + ' box must contain four coordinates')
        result = [_finite_number(value, label + ' coordinates') for value in values]
    except TypeError as error:
        raise ValueError(label + ' box must contain four coordinates') from error
    x0, y0, x1, y1 = result
    if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
        raise ValueError(label + ' box must be a normalized rectangle within the image with positive area')
    return result


def configure_boxes(boxes, mouth_context_scale=1., eyes_context_scale=1.):
    """Expand normalized crop boxes about their centers and clip them to the image.
    Apply once to the original boxes to avoid repeated expansion."""
    if not isinstance(boxes, Mapping):
        raise ValueError('Predicted boxes must be a mapping containing eyes and mouth')
    scales = {
        'eyes': _finite_number(eyes_context_scale, 'Eye context scale'),
        'mouth': _finite_number(mouth_context_scale, 'Mouth context scale'),
    }
    if any(value < 1 for value in scales.values()):
        raise ValueError('Context scales must be at least 1')
    rectangles = {}
    for group in scales:
        if group not in boxes:
            raise ValueError('Predicted boxes are missing ' + group)
        rectangles[group] = _rectangle(boxes[group], group)
    result = deepcopy(dict(boxes))
    for group, scale in scales.items():
        if scale == 1:
            continue
        x0, y0, x1, y1 = rectangles[group]
        cx, cy = (x0+x1)/2, (y0+y1)/2
        rx, ry = (x1-x0)*scale/2, (y1-y0)*scale/2
        result[group] = [max(0., cx-rx), max(0., cy-ry),
                         min(1., cx+rx), min(1., cy+ry)]
    return result


def build_mask_config(arguments_mapping):
    """Read mask configuration, requiring mask_mode and mode to agree when both are supplied."""
    if not isinstance(arguments_mapping, Mapping):
        raise ValueError('Write-region arguments must be a mapping')
    if ('mask_mode' in arguments_mapping and 'mode' in arguments_mapping
            and arguments_mapping['mask_mode'] != arguments_mapping['mode']):
        raise ValueError('mask_mode and mode disagree')
    config = {
        'mode': arguments_mapping.get('mask_mode', arguments_mapping.get('mode', 'ellipse')),
        'eyes_scale': _finite_number(arguments_mapping.get('eyes_scale', 1.), 'Eye region scale'),
        'mouth_scale': _finite_number(arguments_mapping.get('mouth_scale', 1.6), 'Mouth region scale'),
        'context_feather': _finite_number(arguments_mapping.get('context_feather', .08), 'Crop feather fraction'),
        'split_feather': _finite_number(arguments_mapping.get('split_feather', .10), 'Eye-mouth split feather fraction'),
        'split_mode': arguments_mapping.get('split_mode', 'per_view'),
    }
    if config['mode'] not in ('ellipse', 'context_soft'):
        raise ValueError('The write mode must be ellipse or context_soft')
    if config['split_mode'] not in ('per_view', 'global_median'):
        raise ValueError('The split mode must be per_view or global_median')
    if config['mode'] != 'context_soft' and config['split_mode'] != 'per_view':
        raise ValueError('global_median is only supported with context_soft')
    if min(config['eyes_scale'], config['mouth_scale']) <= 0:
        raise ValueError('Eye and mouth region scales must be finite and positive')
    if not 0 < config['context_feather'] < .5:
        raise ValueError('The crop feather fraction must be between 0 and 0.5')
    if not 0 < config['split_feather'] < 1:
        raise ValueError('The eye-mouth split feather fraction must be between 0 and 1')
    return config
