import math
from ref2light.common.controls import RANGES


FOLLOW = {
    'bodyYawDeg': ('headYawDeg', .25, 4.),
    'bodyPitchDeg': ('headPitchDeg', .15, 1.5),
    'bodyRollDeg': ('headRollDeg', .30, 2.),
}

class BodyFollower:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {key: 0. for key in FOLLOW}
        self.targets = dict(self.values)
        self.last_time = None

    def step(self, pose, now, gain=1., response=.22, invert_x=False, invert_y=False):
        if not all(math.isfinite(float(value)) for value in (now, gain, response)) or gain < 0 or response < 0:
            raise ValueError('Body-follow time and gains must be finite and non-negative')
        for body, (head, ratio, limit) in FOLLOW.items():
            if head not in pose or not math.isfinite(float(pose[head])):
                continue
            lo, hi = RANGES[head]
            angle = min(hi, max(lo, float(pose[head])))
            sign = -1. if ((body == 'bodyYawDeg' and invert_x) or (body == 'bodyPitchDeg' and invert_y)) else 1.
            self.targets[body] = min(limit, max(-limit, angle * ratio * gain * sign))
        # Smooth exponentially using elapsed time to avoid jumps after tracking loss or pauses.
        dt = 0. if self.last_time is None else min(.1, max(0., now-self.last_time))
        self.last_time = now
        amount = 1. if response == 0 else -math.expm1(-dt/response)
        for key, target in self.targets.items():
            self.values[key] += amount * (target-self.values[key])
        return dict(self.values)
