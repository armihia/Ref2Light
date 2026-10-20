from ref2light.models.backbone import ReferenceGenerator

OLD12_KEYS = [
    "headYawDeg", "headPitchDeg", "headRollDeg",
    "bodyYawDeg", "bodyPitchDeg", "bodyRollDeg",
    "eyesOpen", "mouthOpen", "mouthCurve",
    "browHeight", "browDistance", "browCurve",
]

POSE14_KEYS = OLD12_KEYS + [
    "gazeX",
    "gazeY",
]

class ConditionedGenerator(
    ReferenceGenerator
):
    def __init__(
        self,
        calibration,
        anchors=12,
        flow_scale=0.35,
    ):
        super().__init__(
            calibration,
            pose_in_dim=14,
            anchors=anchors,
            flow_scale=flow_scale,
        )
