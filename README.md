# Ref2Light

English | [Chinese](README_zh-CN.md)

Ref2Light generates a display-native **1440 × 2560 RGB interlaced image** from a single character reference image and pose and expression controls. This repository provides command-line inference, a Tk controller, and webcam face capture for a calibrated glasses-free light-field display.

## Installation

The current implementation supports Windows, NVIDIA CUDA, and native-resolution display output. Create the environment from the repository root:

```powershell
conda env create -f environment.yml
conda activate ref2light
```

Install a CUDA build of PyTorch suitable for your GPU using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The tested configuration uses Python 3.11, PyTorch 2.14.0+cu130, and an RTX 5090 Laptop GPU. Other dependency versions are listed in `requirements.txt`.

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Model and third-party dependencies

**Ref2Light model download:** [🤗 Hugging Face – Ref2Light](https://huggingface.co/armihia/Ref2Light)

Download the pretrained model weights from Hugging Face and place `ref2light.pt` in the `model/` directory at the repository root, alongside `infer.py`.

The approximately 225 MB `ref2light.pt` file contains the full-body network and all local generation branches.

```text
Ref2Light/
├── ref2light/
├── model/
│   └── ref2light.pt
├── assets/
├── examples/pose.json
├── infer.py
├── tk_controller.py
└── tools/download_preprocess.py
```

The final model path must be `Ref2Light/model/ref2light.pt`. Do not add an extra `model/model/` level.

Facial localization and webcam tracking use Google MediaPipe. Its pretrained models are downloaded separately from the project weights. From the repository root, run:

```powershell
python tools/download_preprocess.py
```

The script downloads the following files from Google and keeps existing valid files. You can also download them manually:

| Model | Destination | Documentation |
| --- | --- | --- |
| [Face Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) | `assets/face_landmarker.task` | [Face Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| [Pose Landmarker Lite](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task) | `assets/pose_landmarker_lite.task` | [Pose Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |

Once assembled, the project folder can be moved as a whole. See [THIRD_PARTY.md](THIRD_PARTY.md) for dependency sources and licensing notes.

## Tk controller and face capture

```powershell
python tk_controller.py --ref "your_reference.png"
```

Use **Change reference** to switch images in the same window. Adjust pose and expression parameters under **Manual controls**, or start webcam tracking under **Camera capture**. Head yaw, pitch, and roll are limited to ±10°. By default, the body follows reduced head motion. Use `--body-mode pose` for shoulder and hip tracking, or `--body-mode manual` for manual body control.

The combined mouth slider moves from closed through a small O to a wide-open mouth. Reference framing is preserved by default. If automatic localization fails, mark the facial regions in the ordinary-view preview.

Native interlaced images must be displayed at **1:1 pixel resolution**, without scaling or lossy compression. The default inference device is `cuda:0`, using the NVIDIA display backend. `--fps 60` sets an update-rate limit; the achieved rate depends on hardware and controls. Use **Save native frame** to save an output. Closing the public controller does not create a diagnostics archive.

## Command-line inference

```powershell
python infer.py --ref "your_reference.png" --mouth-amount 1 --out runs/demo/native.png
python infer.py --ref "your_reference.png" --set eyesOpen=0 --set headYawDeg=10 --out runs/demo/closed.png
python infer.py --ref "your_reference.png" --pose examples/pose.json --out runs/demo/pose.png
```

Use a new output filename for each command. See [MODEL_CARD.md](MODEL_CARD.md) for the 14 control parameters and generation pipeline.

## Updating weights

Close the application, back up `model/ref2light.pt`, and replace it with the new model file. To inspect a candidate first, pass `--model "candidate.pt"` to the controller or inference command.

The model contains display-specific interlacing calibration. It cannot be used unchanged on arbitrary light-field displays. Unseen characters, occlusion, missing body content, and large pose changes may produce errors. Assess visual quality on the target display.
