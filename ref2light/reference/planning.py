from ref2light.common.controls import POSE_KEYS


import json


import math


from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference_pose(metadata):
    """The 12-dimensional metadata omits gaze, which was zero during capture; require all other fields."""
    pose = [float(metadata[key]) for key in POSE_KEYS[:12]]
    pose.extend(float(metadata.get(key, 0.)) for key in POSE_KEYS[12:])
    if not all(map(math.isfinite, pose)):
        raise ValueError('Reference pose contains non-finite values')
    return pose

