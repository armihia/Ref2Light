from __future__ import annotations
import ctypes
import math
import os
import time
from ctypes import wintypes
from pathlib import Path

from ref2light import ROOT

POSE_KEYS = [
    "headYawDeg", "headPitchDeg", "headRollDeg",
    "bodyYawDeg", "bodyPitchDeg", "bodyRollDeg",
    "eyesOpen", "mouthOpen", "mouthCurve",
    "browHeight", "browDistance", "browCurve",
    "gazeX", "gazeY",
]

POSE_INDEX = {k: i for i, k in enumerate(POSE_KEYS)}

NEUTRAL = {
    "headYawDeg": 0.0,
    "headPitchDeg": 0.0,
    "headRollDeg": 0.0,
    "bodyYawDeg": 0.0,
    "bodyPitchDeg": 0.0,
    "bodyRollDeg": 0.0,
    "eyesOpen": 1.0,
    "mouthOpen": 0.0,
    "mouthCurve": 0.0,
    "browHeight": 0.0,
    "browDistance": 0.0,
    "browCurve": 0.0,
    "gazeX": 0.0,
    "gazeY": 0.0,
}

RANGES = {
    "headYawDeg": (-10.0, 10.0),
    "headPitchDeg": (-10.0, 10.0),
    "headRollDeg": (-10.0, 10.0),
    "bodyYawDeg": (-8.0, 8.0),
    "bodyPitchDeg": (-4.0, 4.0),
    "bodyRollDeg": (-4.0, 4.0),
    "eyesOpen": (0.0, 1.0),
    "mouthOpen": (0.0, 0.75),
    "mouthCurve": (-0.25, 0.80),
    "browHeight": (-0.80, 0.80),
    "browDistance": (-0.80, 0.80),
    "browCurve": (-0.80, 0.80),
    "gazeX": (-1.0, 1.0),
    "gazeY": (-1.0, 1.0),
}

FACE4_KEYS = {"eyesOpen", "mouthOpen", "gazeX", "gazeY"}

CONTEXT_KEYS = [k for k in POSE_KEYS if k not in FACE4_KEYS]

CONTEXT_QUANT = {
    "headYawDeg": 0.35, "headPitchDeg": 0.30, "headRollDeg": 0.30,
    "bodyYawDeg": 0.30, "bodyPitchDeg": 0.25, "bodyRollDeg": 0.25,
    "mouthCurve": 0.02, "browHeight": 0.03,
    "browDistance": 0.03, "browCurve": 0.03,
}

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def set_dpi_awareness():
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long), ("top", ctypes.c_long),
        ("right", ctypes.c_long), ("bottom", ctypes.c_long),
    ]

class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]

MONITORENUMPROC = getattr(ctypes, 'WINFUNCTYPE', ctypes.CFUNCTYPE)(
    ctypes.c_int,
    wintypes.HMONITOR,
    wintypes.HDC,
    ctypes.POINTER(RECT),
    wintypes.LPARAM,
)

def enumerate_monitors():
    if os.name != "nt":
        return []
    monitors = []
    user32 = ctypes.windll.user32

    def callback(h_monitor, hdc, lprc, data):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if user32.GetMonitorInfoW(h_monitor, ctypes.byref(info)):
            r = info.rcMonitor
            monitors.append({
                "device": info.szDevice,
                "left": int(r.left), "top": int(r.top),
                "width": int(r.right-r.left), "height": int(r.bottom-r.top),
                "primary": bool(info.dwFlags & 1),
            })
        return 1

    cb = MONITORENUMPROC(callback)
    user32.EnumDisplayMonitors(None, None, cb, 0)
    return monitors

def find_native_monitor(width=1440, height=2560):
    monitors = enumerate_monitors()
    matches = [m for m in monitors if m["width"] == width and m["height"] == height]
    if not matches:
        lines = [f"{m['device']} {m['width']}x{m['height']} at ({m['left']},{m['top']})" for m in monitors]
        raise RuntimeError(
            f"No exact {width}x{height} native light-field display found.\nDetected:\n" + "\n".join(lines)
        )
    return matches[0]

def _dist2(a, b):
    return math.hypot(float(a[0]-b[0]), float(a[1]-b[1]))

def _landmark_xy(landmarks, idx):
    lm = landmarks[idx]
    return (float(lm.x), float(lm.y))

def _ear(landmarks, ids):
    p = [_landmark_xy(landmarks, i) for i in ids]
    den = 2.0*_dist2(p[0], p[3])
    if den < 1e-8:
        return 0.0
    return (_dist2(p[1],p[5]) + _dist2(p[2],p[4]))/den

def _iris_position(landmarks, iris_ids, corner_ids, lid_ids):
    iris = [_landmark_xy(landmarks, i) for i in iris_ids]
    cx = sum(p[0] for p in iris)/len(iris)
    cy = sum(p[1] for p in iris)/len(iris)
    c0 = _landmark_xy(landmarks, corner_ids[0])
    c1 = _landmark_xy(landmarks, corner_ids[1])
    top = _landmark_xy(landmarks, lid_ids[0])
    bot = _landmark_xy(landmarks, lid_ids[1])
    xmin,xmax = sorted((c0[0],c1[0]))
    ymin,ymax = sorted((top[1],bot[1]))
    gx = (cx-xmin)/max(xmax-xmin,1e-6)
    gy = (cy-ymin)/max(ymax-ymin,1e-6)
    return gx,gy

def _head_pose_cv2(cv2, landmarks, w, h):
    import numpy as np
    ids = [1,152,33,263,61,291]
    image_points = np.asarray(
        [[landmarks[i].x*w, landmarks[i].y*h] for i in ids], dtype=np.float64
    )
    # Generic face template. Absolute offset is removed by neutral calibration;
    # only frame-to-frame rotation is used.
    model_points = np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ], dtype=np.float64)
    focal = float(w)
    camera = np.asarray([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1), dtype=np.float64)
    ok,rvec,tvec = cv2.solvePnP(
        model_points, image_points, camera, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rot,_ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rot)[0]
    # x,y,z rotations in degrees: pitch, yaw, roll.
    return float(angles[0]), float(angles[1]), float(angles[2])

def face_capture_loop(
    camera_index,
    out_queue,
    stop_event,
    calibrate_event,
    config,
    config_lock,
):
    """Track the camera using the installed MediaPipe Solutions or Tasks interface."""
    cv2 = None
    face_mesh = None
    pose_estimator = None
    try:
        import cv2 as _cv2
        cv2 = _cv2
        import mediapipe as mp_lib
    except Exception as exc:
        out_queue.put({
            "type": "error",
            "message": (
                f"Face capture requires opencv-python + mediapipe: {exc}. "
                "Install with: python -m pip install opencv-python mediapipe"
            ),
        })
        return

    def download_task_model(filename, url):
        model_dir = ROOT / "assets"
        model_dir.mkdir(parents=True, exist_ok=True)
        dst = model_dir / filename
        if dst.is_file() and dst.stat().st_size > 1024:
            return dst
        import urllib.request
        tmp = dst.with_suffix(dst.suffix + ".download")
        out_queue.put({"type":"status", "message":f"Downloading MediaPipe model: {filename} ..."})
        try:
            urllib.request.urlretrieve(url, tmp)
            if not tmp.is_file() or tmp.stat().st_size <= 1024:
                raise RuntimeError(f"Downloaded model is unexpectedly small: {tmp}")
            tmp.replace(dst)
        finally:
            if tmp.exists():
                try: tmp.unlink()
                except Exception: pass
        return dst

    # MediaPipe changed its Python surface after 0.10.21.  Prefer the old
    # Solutions API when present; otherwise use the supported Tasks API.
    use_legacy_solutions = hasattr(mp_lib, "solutions")
    try:
        if use_legacy_solutions:
            face_mesh = mp_lib.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            pose_estimator = mp_lib.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            backend_name = f"MediaPipe Solutions {getattr(mp_lib,'__version__','')}"
        else:
            if not hasattr(mp_lib, "tasks") or not hasattr(mp_lib.tasks, "vision"):
                raise RuntimeError(
                    "This MediaPipe build has neither mp.solutions nor mp.tasks.vision. "
                    f"Imported from: {getattr(mp_lib,'__file__','unknown')}"
                )
            vision = mp_lib.tasks.vision
            base_options = mp_lib.tasks.BaseOptions
            face_model = download_task_model(
                "face_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            )
            pose_model = download_task_model(
                "pose_landmarker_lite.task",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            )
            face_options = vision.FaceLandmarkerOptions(
                base_options=base_options(model_asset_path=str(face_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            pose_options = vision.PoseLandmarkerOptions(
                base_options=base_options(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            face_mesh = vision.FaceLandmarker.create_from_options(face_options)
            pose_estimator = vision.PoseLandmarker.create_from_options(pose_options)
            backend_name = f"MediaPipe Tasks {getattr(mp_lib,'__version__','')}"
        out_queue.put({"type":"status", "message":f"Face capture backend ready: {backend_name}"})
    except Exception as exc:
        out_queue.put({"type":"error", "message":"MediaPipe tracker init failed: "+repr(exc)})
        return

    cap = cv2.VideoCapture(
        int(camera_index),
        cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY,
    )
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        out_queue.put({"type":"error","message":f"Cannot open camera index {camera_index}"})
        try:
            face_mesh.close(); pose_estimator.close()
        except Exception:
            pass
        return

    # 480x360 is ample for face/iris landmarks and materially cheaper than 640x480.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    calibration = None
    smooth_pose = {}
    face_seen = 0
    last_timestamp_ms = 0
    capture_frame_index = 0
    last_pose_legacy = None
    last_pose_tasks = None

    def cfg_snapshot():
        with config_lock:
            return dict(config)

    try:
        while not stop_event.is_set():
            ok,frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            cfg = cfg_snapshot()
            capture_frame_index += 1
            if cfg.get("mirror", True):
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            body_interval = max(1, int(cfg.get("body_interval", 3)))
            run_body = bool(cfg.get("track_body", False)) and (capture_frame_index % body_interval == 0)
            if use_legacy_solutions:
                fresult = face_mesh.process(rgb)
                if run_body:
                    last_pose_legacy = pose_estimator.process(rgb)
                presult = last_pose_legacy if cfg.get("track_body", False) else None
                have_face = bool(fresult.multi_face_landmarks)
                lm = fresult.multi_face_landmarks[0].landmark if have_face else None
                pl = (
                    presult.pose_landmarks.landmark
                    if presult is not None and presult.pose_landmarks is not None
                    else None
                )
            else:
                # Tasks VIDEO mode requires a strictly increasing millisecond timestamp.
                now_ms = int(time.monotonic_ns() // 1_000_000)
                timestamp_ms = max(last_timestamp_ms + 1, now_ms)
                last_timestamp_ms = timestamp_ms
                mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
                fresult = face_mesh.detect_for_video(mp_image, timestamp_ms)
                if run_body:
                    last_pose_tasks = pose_estimator.detect_for_video(mp_image, timestamp_ms)
                presult = last_pose_tasks if cfg.get("track_body", False) else None
                have_face = bool(getattr(fresult, "face_landmarks", None))
                lm = fresult.face_landmarks[0] if have_face else None
                pl = (
                    presult.pose_landmarks[0]
                    if presult is not None and getattr(presult, "pose_landmarks", None)
                    else None
                )

            if not have_face:
                out_queue.put({"type":"status","message":"Face capture: no face"})
                if cfg.get("preview", False):
                    cv2.putText(frame,"NO FACE",(20,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    cv2.imshow("Ref2Light face capture",frame)
                    cv2.waitKey(1)
                continue

            face_seen += 1
            h,w = frame.shape[:2]
            hp = _head_pose_cv2(cv2,lm,w,h)
            if hp is None:
                continue
            pitch_raw,yaw_raw,roll_raw = hp

            left_ear = _ear(lm,[33,160,158,133,153,144])
            right_ear = _ear(lm,[362,385,387,263,373,380])
            ear = 0.5*(left_ear+right_ear)
            mouth_ratio = _dist2(_landmark_xy(lm,13),_landmark_xy(lm,14))/max(
                _dist2(_landmark_xy(lm,61),_landmark_xy(lm,291)),1e-6
            )

            try:
                lgx,lgy = _iris_position(lm,range(468,473),(33,133),(159,145))
                rgx,rgy = _iris_position(lm,range(473,478),(362,263),(386,374))
                gx,gy = 0.5*(lgx+rgx),0.5*(lgy+rgy)
            except Exception:
                gx,gy = 0.5,0.5

            body_raw = (0.0,0.0,0.0)
            if pl is not None:
                ls,rs,lh,rh = pl[11],pl[12],pl[23],pl[24]
                sw = math.hypot(rs.x-ls.x, rs.y-ls.y)
                torso = math.hypot((0.5*(lh.x+rh.x)-0.5*(ls.x+rs.x)),
                                   (0.5*(lh.y+rh.y)-0.5*(ls.y+rs.y)))
                body_roll = math.degrees(math.atan2(rs.y-ls.y, rs.x-ls.x))
                body_yaw_proxy = (rs.z-ls.z)/max(sw,1e-4)
                shoulder_z = 0.5*(ls.z+rs.z); hip_z = 0.5*(lh.z+rh.z)
                body_pitch_proxy = (shoulder_z-hip_z)/max(torso,1e-4)
                body_raw = (body_yaw_proxy,body_pitch_proxy,body_roll)

            raw_signals = {
                "head_pitch":pitch_raw,"head_yaw":yaw_raw,"head_roll":roll_raw,
                "ear":ear,"mouth":mouth_ratio,"gaze_x":gx,"gaze_y":gy,
                "body_yaw":body_raw[0],"body_pitch":body_raw[1],"body_roll":body_raw[2],
            }

            if calibration is None or calibrate_event.is_set():
                calibration = dict(raw_signals)
                calibrate_event.clear()
                smooth_pose.clear()
                out_queue.put({"type":"status","message":"Face capture: neutral calibrated"})

            head_gain = float(cfg.get("head_gain",1.0))
            gaze_gain = float(cfg.get("gaze_gain",4.0))
            body_gain = float(cfg.get("body_gain",1.0))
            mouth_gain = float(cfg.get("mouth_gain",1.0))

            hy = (yaw_raw-calibration["head_yaw"])*head_gain
            hpitch = (pitch_raw-calibration["head_pitch"])*head_gain
            hr = (roll_raw-calibration["head_roll"])*head_gain
            if cfg.get("invert_head_x",False): hy = -hy
            if cfg.get("invert_head_y",False): hpitch = -hpitch

            gxo = (gx-calibration["gaze_x"])*gaze_gain
            gyo = -(gy-calibration["gaze_y"])*gaze_gain
            if cfg.get("invert_gaze_x",False): gxo = -gxo
            if cfg.get("invert_gaze_y",False): gyo = -gyo

            open_ear = max(calibration["ear"],1e-5)
            closed_ear = 0.42*open_ear
            eye_open = (ear-closed_ear)/max(open_ear-closed_ear,1e-5)

            mouth_delta = max(0.0,mouth_ratio-calibration["mouth"])
            mouth_open = 0.75*mouth_delta/max(0.28/max(mouth_gain,1e-3),1e-4)

            byaw = (body_raw[0]-calibration["body_yaw"])*12.0*body_gain
            bpitch = (body_raw[1]-calibration["body_pitch"])*6.0*body_gain
            broll = (body_raw[2]-calibration["body_roll"])*body_gain
            if cfg.get("invert_body_x",False): byaw=-byaw
            if cfg.get("invert_body_y",False): bpitch=-bpitch

            pose = {
                "headYawDeg":clamp(hy,*RANGES["headYawDeg"]),
                "headPitchDeg":clamp(hpitch,*RANGES["headPitchDeg"]),
                "headRollDeg":clamp(hr,*RANGES["headRollDeg"]),
                "bodyYawDeg":clamp(byaw,*RANGES["bodyYawDeg"]),
                "bodyPitchDeg":clamp(bpitch,*RANGES["bodyPitchDeg"]),
                "bodyRollDeg":clamp(broll,*RANGES["bodyRollDeg"]),
                "eyesOpen":clamp(eye_open,*RANGES["eyesOpen"]),
                "gazeX":clamp(gxo,*RANGES["gazeX"]),
                "gazeY":clamp(gyo,*RANGES["gazeY"]),
                "mouthOpen":clamp(mouth_open,*RANGES["mouthOpen"]),
            }

            alpha = clamp(cfg.get("smoothing",0.72),0.0,0.98)
            for k,v in pose.items():
                if k not in smooth_pose:
                    smooth_pose[k]=v
                else:
                    smooth_pose[k]=alpha*smooth_pose[k]+(1.0-alpha)*v

            try:
                while out_queue.qsize()>2:
                    out_queue.get_nowait()
            except Exception:
                pass
            out_queue.put({"type":"pose","pose":dict(smooth_pose)})

            if cfg.get("preview",False):
                cv2.putText(frame,f"head y/p {pose['headYawDeg']:+.1f} {pose['headPitchDeg']:+.1f}",
                            (15,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),1,cv2.LINE_AA)
                cv2.putText(frame,f"gaze {pose['gazeX']:+.2f} {pose['gazeY']:+.2f} eye {pose['eyesOpen']:.2f} mouth {pose['mouthOpen']:.2f}",
                            (15,54),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1,cv2.LINE_AA)
                cv2.putText(frame,backend_name,(15,80),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,0),1,cv2.LINE_AA)
                cv2.imshow("Ref2Light face capture",frame)
                if (cv2.waitKey(1)&0xFF) in (27,ord('q')):
                    stop_event.set()

    except Exception as exc:
        out_queue.put({"type":"error","message":"Face capture error: "+repr(exc)})
    finally:
        try:
            if face_mesh is not None: face_mesh.close()
            if pose_estimator is not None: pose_estimator.close()
        except Exception:
            pass
        cap.release()
        if cv2 is not None:
            try:
                cv2.destroyWindow("Ref2Light face capture")
            except Exception:
                pass
        out_queue.put({"type":"status","message":"Face capture stopped"})
