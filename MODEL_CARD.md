# Model card

## Inputs and outputs

The model takes one RGB character reference image and 14 raw controls. The image is fitted into a square canvas without stretching, then mapped to the encoder canvas. Inference does not require a character mesh, depth map, or ground-truth geometry. When a reference is loaded, eyes, mouth, and eyebrows are localized in ordinary anchor views before interlacing.

The output is a display-native **1440 × 2560 RGB image** containing multi-view subpixel information arranged using the display calibration. Present it at its original resolution.

| Index | Parameter | Meaning | Tk range |
| --- | --- | --- | --- |
| 0–2 | `headYawDeg` / `headPitchDeg` / `headRollDeg` | Head yaw, pitch, and roll in degrees | ±10 each |
| 3–5 | `bodyYawDeg` / `bodyPitchDeg` / `bodyRollDeg` | Body yaw, pitch, and roll in degrees | ±8 / ±4 / ±4 |
| 6 | `eyesOpen` | Eye openness | 0–1 |
| 7 | `mouthOpen` | Mouth opening | 0–0.75 |
| 8 | `mouthCurve` | Negative mouth-curve control; positive values apply the dataset mouth-rad expansion | −0.25–0.8 |
| 9–11 | `browHeight` / `browDistance` / `browCurve` | Eyebrow height, spacing, and curvature | ±0.8 each |
| 12–13 | `gazeX` / `gazeY` | Horizontal and vertical gaze | ±1 each |

The default pose uses zero head and body angles, open eyes, and zero for the other expressions. Controls have the absolute semantics of the teacher parameters. The original pose of an arbitrary input image may differ from this default. For dataset references with accompanying JSON metadata, the controller reads the recorded source pose.

Tk head limits apply to sliders, face capture, and source-pose reset. Command-line and training controls retain their parameter semantics and internal normalization coefficients.

## Generation pipeline

1. The reference encoder extracts multi-scale convolutional features, which are cached until the reference changes. The normalized pose conditions decoder features through FiLM. A decoder with skip connections predicts sampling displacement, direct colour, and blending weights for twelve ordinary anchor views.
2. The base views represent the full body, rigid pose, and eye openness. Local networks use the current ordinary views, sampled reference textures, and identity features to correct gaze and mouth appearance. The mouth includes a pose-conditioned closure residual head.
3. A separate head expands the mouth for positive `mouthCurve`; another local head controls the eyebrows. Facial regions follow source-UV correspondences from the base network. The base views are retained outside the local write regions.
4. After corrections in ordinary-view space, a fixed, non-trainable interlacer produces the native display image. Critical colour, coordinate-sampling, and interlacing operations use FP32, while CNN operations use BF16 automatic mixed precision.

The implementation is in `ref2light/models/`, `ref2light/local/`, `ref2light/joint/`, `ref2light/rad/`, `ref2light/brows/`, and `ref2light/runtime/session.py`.

## Model file

`model/ref2light.pt` contains the complete generation pipeline:

| Field | Contents |
| --- | --- |
| `base` | Full-body weights, architecture, pose statistics, calibration, alignment points, and training step |
| `local` | Local control networks and the pose-conditioned closure head |
| `rad` | Positive mouth-expansion head |
| `brows` | Eyebrow head |
| `runtime` | Eye mode, region sizes, write masks, and localization policy |
| `provenance` | Version and selected training-checkpoint information |

The initial release uses the best checkpoint at step **16,000** of the `pose_expression` run. All **431 state tensors** were equal before and after assembly into the complete inference model. Optimizer state and unrelated run paths are not part of the inference file. The complete weights can initialize further training. For later model updates, `base.step` and `provenance` identify the checkpoint; the controller and export tool report its step.

## Scope and evidence

The model targets character images, limited head and body motion, and the calibrated glasses-free light-field display. The maintainer accepted the visual result on their device. Packaging checks also confirmed identical native outputs for two reference images across eighteen control conditions.

These checks do not establish cross-dataset benchmark performance or guarantee quality for arbitrary characters and poses. Occlusion, body content missing from the reference, unusual framing, and large motions can still produce errors. Further training uses paired supervision and native-image replay. Visual quality must be assessed on the target display.
