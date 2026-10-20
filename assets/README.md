# Localization and capture models

Download the third-party model files separately. From the repository root, run:

```powershell
python tools/download_preprocess.py
```

Alternatively, download the files manually into this directory:

- [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task): facial localization and webcam face capture.
- [pose_landmarker_lite.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task): shoulder and hip tracking.

See [THIRD_PARTY.md](../THIRD_PARTY.md) for sources and licensing notes. The project generation model belongs at `model/ref2light.pt`; keep third-party landmark files in `assets/`.

