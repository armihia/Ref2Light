# Third-party dependencies and sources

Ref2Light uses PyTorch, NumPy, Pillow, OpenCV, MediaPipe, pygame, tqdm, and Python/Tk. Each dependency retains its own licence. Distributing this project does not change the terms of third-party code or weights.

Facial localization in ordinary views and webcam face capture use [Google MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker). Download the model from the official [Face Landmarker float16/1 endpoint](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task).

Shoulder and hip tracking uses [Google MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) with the official [Lite float16/1 model](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task). The default body-follow mode uses head motion. Store the separately downloaded capture models in `assets/`.

Consult the official model documentation for the terms governing each pretrained file. Terms for website example code do not necessarily apply to all pretrained weights. No new open-source licence is selected or granted on behalf of the authors here. The maintainer must specify the distribution terms for project code, generation weights, and training data separately.

Local reference images, training images, 3D teacher assets, and experimental results are excluded from the source package. The generation model contains display-specific interlacing calibration and is not automatically interchangeable across light-field display models.
