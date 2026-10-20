import math
from ref2light.common.capture_backend import POSE_KEYS, NEUTRAL, CONTEXT_KEYS, RANGES


CONTEXT_INDICES = tuple(POSE_KEYS.index(key) for key in CONTEXT_KEYS)

LABELS = dict(zip(POSE_KEYS, (
    'Head yaw', 'Head pitch', 'Head roll', 'Body yaw', 'Body pitch', 'Body roll',
    'Eye openness', 'Mouth opening', 'Mouth curve', 'Brow height', 'Brow distance', 'Brow curve',
    'Gaze horizontal', 'Gaze vertical')))

PRESETS = {
    'Neutral reset': dict(NEUTRAL), 'Close eyes': {'eyesOpen': 0.}, 'Open eyes': {'eyesOpen': 1.},
    'Close mouth': {'mouthOpen': 0.}, 'Open mouth': {'mouthOpen': .75},
    'Turn head left': {'headYawDeg': -10.}, 'Turn head right': {'headYawDeg': 10.},
    'Center head': {'headYawDeg': 0., 'headPitchDeg': 0., 'headRollDeg': 0.},
    'Look left': {'gazeX': -1., 'gazeY': 0.},
    'Look right': {'gazeX': 1., 'gazeY': 0.}, 'Center gaze': {'gazeX': 0., 'gazeY': 0.},
}

def validate_pose(values):
    """Require all 14 raw controls without implicitly resetting any axis."""
    if len(values) != len(POSE_KEYS):
        raise ValueError('Expected 14 pose parameters')
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError('Pose parameters must be finite')
    return result

def limit_tk_head(values):
    """Apply Tk head limits to manual, capture, and reference-metadata inputs."""
    result = list(validate_pose(values))
    for index in range(3):
        lo, hi = RANGES[POSE_KEYS[index]]
        result[index] = min(hi, max(lo, result[index]))
    return tuple(result)

def context_key(values):
    values = validate_pose(values)
    return tuple(values[index] for index in CONTEXT_INDICES)

def apply_preset(values, name):
    result = list(validate_pose(values))
    for key, value in PRESETS[name].items():
        result[POSE_KEYS.index(key)] = value
    return tuple(result)
