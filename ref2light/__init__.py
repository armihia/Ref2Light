from pathlib import Path

__version__ = '6.14.0'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT/'model/ref2light.pt'
LANDMARK_MODEL = ROOT/'assets/face_landmarker.task'
