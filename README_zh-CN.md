# Ref2Light

[English](README.md) | 简体中文

Ref2Light 根据单张人物参考图及姿态、表情参数，生成可直接送到指定裸眼光场屏的 **1440 × 2560 RGB 原生交织图像**。本仓库提供命令行推理、Tk 控制器和摄像头面捕。

## 安装

目前支持 Windows、NVIDIA CUDA 和原生分辨率显示。在源码根目录创建环境：

```powershell
conda env create -f environment.yml
conda activate ref2light
```

随后按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择适合显卡的 CUDA 构建。已测试环境为 Python 3.11、PyTorch 2.14.0+cu130 和 RTX 5090 Laptop GPU；其他依赖版本见 `requirements.txt`。

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 模型与第三方依赖

**Ref2Light 预训练模型下载地址：** [🤗 Hugging Face – Ref2Light](https://huggingface.co/armihia/Ref2Light)

从 Hugging Face 下载预训练模型权重 `ref2light.pt`，并将其放入源码根目录下的 `model/` 文件夹中，与 `infer.py` 位于同一级目录结构。

约 225 MB 的 `ref2light.pt` 文件包含全身网络及全部局部生成分支。

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

模型文件的正确路径为 `Ref2Light/model/ref2light.pt`，请勿额外嵌套一层 `model/model/` 目录。

人脸定位和摄像头追踪使用 Google MediaPipe，其预训练模型需单独下载。在源码根目录执行：

```powershell
python tools/download_preprocess.py
```

脚本从 Google 官方地址下载以下文件，并保留已有有效文件；也可手动下载后放到对应位置：

| 模型 | 放置位置 | 官方说明 |
| --- | --- | --- |
| [Face Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) | `assets/face_landmarker.task` | [人脸定位文档](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| [Pose Landmarker Lite](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task) | `assets/pose_landmarker_lite.task` | [姿态定位文档](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |

组装完成后可以整体移动工程。第三方来源与授权说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## Tk 控制与面捕

```powershell
python tk_controller.py --ref "your_reference.png"
```

在同一窗口点击 **Change reference** 换图，在 **Manual controls** 页调节姿态和表情，在 **Camera capture** 页开启面捕。头部左右、俯仰和侧倾均限制为 ±10°。默认身体弱幅跟随头部；`--body-mode pose` 使用肩髋追踪，`--body-mode manual` 保留手动身体控制。

复合张嘴滑杆从闭嘴依次经过小 O 到大开口。默认保持参考图构图；自动定位失败时可在普通视图预览中框选五官。

原生交织图像必须按 **1:1 像素分辨率**显示，不要缩放或有损压缩。默认推理设备为 `cuda:0`，使用 NVIDIA 显示路径。`--fps 60` 设置更新上限，实际帧率取决于硬件和控制条件。点击 **Save native frame** 手动保存输出；公开控制器退出时不生成诊断包。

## 命令行推理

```powershell
python infer.py --ref "your_reference.png" --mouth-amount 1 --out runs/demo/native.png
python infer.py --ref "your_reference.png" --set eyesOpen=0 --set headYawDeg=10 --out runs/demo/closed.png
python infer.py --ref "your_reference.png" --pose examples/pose.json --out runs/demo/pose.png
```

每条命令使用新的输出文件名。十四维参数及生成流程见 [MODEL_CARD.md](MODEL_CARD.md)。

## 更新权重

关闭程序，备份已有 `model/ref2light.pt`，再用新模型替换。也可先通过 `--model "candidate.pt"` 测试候选模型，确认后再替换默认文件。

模型包含设备交织标定，不能直接用于任意型号的光场屏。未见过的人物、遮挡、缺失的身体内容及大角度变化仍可能产生误差，实际效果需要在目标设备上检查。
