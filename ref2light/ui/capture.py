import math
import queue
import threading
import time
from ref2light.ui.body import BodyFollower
from ref2light.common.capture_backend import face_capture_loop, CONTEXT_QUANT
from ref2light.common.controls import RANGES


CHANNELS = {
    'head': ('headYawDeg', 'headPitchDeg', 'headRollDeg'),
    'body': ('bodyYawDeg', 'bodyPitchDeg', 'bodyRollDeg'),
    'gaze': ('gazeX', 'gazeY'),
    'blink': ('eyesOpen',),
    'mouth': ('mouthOpen',),
}

def tracked_values(pose, enabled):
    values = {}
    for channel, keys in CHANNELS.items():
        if not enabled.get(channel, False):
            continue
        for key in keys:
            if key not in pose:
                continue
            value = float(pose[key])
            if not math.isfinite(value):
                continue
            lo, hi = RANGES[key]
            value = min(hi, max(lo, value))
            # Quantize pose with the existing steps to reduce full-body updates from camera jitter.
            if key in CONTEXT_QUANT and lo < value < hi:
                step = CONTEXT_QUANT[key]
                value = min(hi, max(lo, round(value / step) * step))
            values[key] = value
    return values

class CaptureQueue(queue.Queue):
    def put(self, item, block=True, timeout=None):
        # Keep capture writes non-blocking so Tk shutdown or pauses do not delay camera release.
        while True:
            try:
                return super().put(item, block=False)
            except queue.Full:
                try:
                    self.get_nowait()
                except queue.Empty:
                    pass

def translated_status(message):
    names = {
        'Face capture: no face': 'No face detected; keeping the last pose',
        'Face capture: neutral calibrated': 'Neutral pose calibrated',
        'Face capture stopped': 'Capture stopped; manual controls active',
    }
    if message in names:
        return names[message]
    if message.startswith('Face capture backend ready:'):
        return 'Face capture ready: ' + message.split(':', 1)[1].strip()
    return 'Face capture: ' + message

class CapturePanel:
    def __init__(self, parent, apply_pose, camera=0, on_start=lambda: None, body_mode="head_follow"):
        import tkinter as tk
        from tkinter import ttk
        self.apply_pose, self.on_start = apply_pose, on_start
        self.messages = CaptureQueue(maxsize=8)
        self.stop_event, self.calibrate_event = threading.Event(), threading.Event()
        self.config_lock = threading.Lock()
        self.thread = None
        self.closed = False
        self.error = None
        self.was_alive = False
        self.config = dict(body_interval=3)
        self.frame = ttk.Frame(parent, padding=10)
        self.frame.pack(fill='both', expand=True)
        self.status = tk.StringVar(parent, value='Capture stopped; camera off')
        self.camera = tk.IntVar(parent, value=camera)
        self.drive = tk.BooleanVar(parent, value=True)
        self.body_mode = tk.StringVar(parent, value=body_mode)
        self.body_response = tk.DoubleVar(parent, value=.22)
        self.body_follower = BodyFollower()
        ttk.Label(self.frame, text='Face the camera with eyes open and mouth closed, then click Calibrate neutral. Unchecked channels remain under manual control.',
            wraplength=950).pack(fill='x', pady=(0, 8))
        camera_row = ttk.Frame(self.frame)
        camera_row.pack(fill='x')
        ttk.Label(camera_row, text='Camera index').pack(side='left')
        self.camera_box = ttk.Spinbox(camera_row, from_=0, to=9, width=5, textvariable=self.camera)
        self.camera_box.pack(side='left', padx=8)
        ttk.Label(camera_row, textvariable=self.status, wraplength=760).pack(side='left', fill='x', expand=True)
        buttons = ttk.Frame(self.frame)
        buttons.pack(fill='x', pady=8)
        self.start_button = ttk.Button(buttons, text='Start capture', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(buttons, text='Stop capture', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', padx=8)
        ttk.Button(buttons, text='Calibrate neutral', command=self.calibrate).pack(side='left')
        ttk.Checkbutton(buttons, text='Drive model', variable=self.drive, command=self.drive_changed).pack(side='left', padx=12)
        group = ttk.LabelFrame(self.frame, text='Capture channels', padding=8)
        group.pack(fill='x', pady=5)
        self.channels = {}
        for name, label in (('head', 'Head'), ('blink', 'Blink'), ('mouth', 'Mouth opening'), ('gaze', 'Gaze'), ('body', 'Body motion')):
            value = self.channels[name] = tk.BooleanVar(parent, value=name != 'body' or body_mode != 'manual')
            button = ttk.Checkbutton(group, text=label, variable=value, command=self.sync_config)
            button.pack(side='left', padx=8)
            if name == 'body':self.body_checkbox = button
        body_options = ttk.LabelFrame(self.frame, text='Body motion source', padding=6)
        body_options.pack(fill='x', pady=4)
        modes = ttk.Frame(body_options);modes.pack(fill='x')
        for mode,label in (('head_follow','Head follow (default)'),('pose','Shoulder and hip tracking'),('manual','Manual body control')):
            ttk.Radiobutton(modes,text=label,variable=self.body_mode,value=mode,
                command=self.body_mode_changed).pack(side='left',padx=8)
        response = ttk.Frame(body_options);response.pack(fill='x')
        ttk.Label(response,text='Body smoothing (s)').pack(side='left')
        tk.Scale(response,from_=0.,to=.6,resolution=.02,orient='horizontal',variable=self.body_response,
            command=lambda value:self.sync_config(),length=280,highlightthickness=0).pack(side='left',padx=8)
        ttk.Label(response,text='0 follows immediately; adjust body gain below').pack(side='left')
        tuning = ttk.LabelFrame(self.frame, text='Smoothing and motion gain', padding=8)
        tuning.pack(fill='x', pady=5)
        self.tuning = {}
        for row, (name, label, initial, lo, hi) in enumerate((
                ('smoothing', 'Smoothing', .72, 0., .95),
                ('head_gain', 'Head gain', 1., .25, 2.),
                ('gaze_gain', 'Gaze gain', 4., 1., 8.),
                ('mouth_gain', 'Mouth gain', 1., .5, 3.),
                ('body_gain', 'Body gain', 1., .25, 2.))):
            variable = self.tuning[name] = tk.DoubleVar(parent, value=initial)
            ttk.Label(tuning, text=label, width=12).grid(row=row, column=0, sticky='w')
            tk.Scale(tuning, from_=lo, to=hi, resolution=.01, orient='horizontal', variable=variable,
                command=lambda value: self.sync_config(), length=480, highlightthickness=0).grid(row=row, column=1, sticky='ew')
        tuning.columnconfigure(1, weight=1)
        flags = ttk.LabelFrame(self.frame, text='Mirror, preview, and direction', padding=8)
        flags.pack(fill='x', pady=5)
        self.flags = {}
        choices = (('mirror', 'Mirror camera', True), ('preview', 'Show camera preview', False),
            ('invert_head_x', 'Invert head yaw', False), ('invert_head_y', 'Invert head pitch', False),
            ('invert_gaze_x', 'Invert horizontal gaze', False), ('invert_gaze_y', 'Invert vertical gaze', False),
            ('invert_body_x', 'Invert body yaw', False), ('invert_body_y', 'Invert body pitch', False))
        for index, (name, label, initial) in enumerate(choices):
            variable = self.flags[name] = tk.BooleanVar(parent, value=initial)
            ttk.Checkbutton(flags, text=label, variable=variable, command=self.sync_config).grid(
                row=index//4, column=index%4, sticky='w', padx=8, pady=4)
        ttk.Label(self.frame, text='Head follow uses 25%/15%/30% of head yaw/pitch/roll, limited to +/-4, +/-1.5, and +/-2 degrees. Manual input pauses capture control; select Drive model to resume.',
            wraplength=950).pack(fill='x', pady=8)
        self.body_checkbox.configure(state="disabled" if body_mode == "manual" else "normal")
        self.sync_config()

    def sync_config(self):
        if not hasattr(self, 'flags'):
            return
        with self.config_lock:
            self.config.update({key: float(value.get()) for key, value in self.tuning.items()})
            self.config.update({key: bool(value.get()) for key, value in self.flags.items()})
            self.config['track_body'] = self.body_mode.get() == 'pose' and bool(self.channels['body'].get())

    def body_mode_changed(self):
        mode = self.body_mode.get()
        self.body_follower.reset()
        self.channels['body'].set(mode != 'manual')
        self.body_checkbox.configure(state='disabled' if mode == 'manual' else 'normal')
        self.sync_config()
        self.status.set({'head_follow':'Body follows reduced head motion; shoulders and hips need not be visible',
            'pose':'Body uses shoulder and hip tracking; keep them visible and calibrate neutral',
            'manual':'Body remains manual; capture drives the selected facial channels'}[mode])

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        try:
            camera = int(self.camera.get())
            if camera < 0:
                raise ValueError('Camera index must be non-negative')
        except Exception:
            self.status.set('Enter a valid non-negative camera index')
            return
        self.body_follower.reset()
        self.sync_config()
        while not self.messages.empty():
            self.messages.get_nowait()
        self.error = None
        self.stop_event.clear()
        self.calibrate_event.set()
        self.drive.set(True)
        self.thread = threading.Thread(target=face_capture_loop, args=(camera, self.messages,
            self.stop_event, self.calibrate_event, self.config, self.config_lock), daemon=True)
        self.thread.start()
        self.was_alive = True
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.camera_box.configure(state='disabled')
        self.status.set('Starting camera capture...')
        self.on_start()

    def stop(self):
        self.body_follower.reset()
        self.drive.set(False)
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.status.set('Stopping capture and releasing the camera...')

    def pause_drive(self):
        if self.thread is not None and self.thread.is_alive() and self.drive.get():
            self.drive.set(False)
            self.status.set('Capture control is paused; manual controls are active. Select Drive model to resume.')

    def drive_changed(self):
        self.status.set('Capture control resumed' if self.drive.get() else 'Capture control paused; manual controls active')

    def calibrate(self):
        if self.thread is None or not self.thread.is_alive():
            self.status.set('Start capture before calibrating neutral')
            return
        self.calibrate_event.set()
        self.status.set('Calibration requested; face the camera with eyes open and mouth closed')

    def poll(self):
        if self.closed:
            return
        latest_pose = None
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message.get('type')
                if kind == 'pose':
                    latest_pose = message['pose']
                elif kind == 'status':
                    if self.error is None:
                        self.status.set(translated_status(message.get('message', '')))
                    if message.get('message') == 'Face capture: neutral calibrated':
                        self.body_follower.reset()
                    if message.get('message') == 'Face capture: no face':
                        latest_pose = None
                elif kind == 'error':
                    self.error = message.get('message', 'Unknown error')
                    self.stop_event.set()
                    self.drive.set(False)
                    self.status.set('Capture failed; manual controls remain available: ' + self.error)
        except queue.Empty:
            pass
        if latest_pose is not None and self.drive.get() and not self.stop_event.is_set():
            enabled = {key: bool(value.get()) for key, value in self.channels.items()}
            if self.body_mode.get() == 'manual':enabled['body'] = False
            elif self.body_mode.get() == 'head_follow' and enabled['body']:
                body = self.body_follower.step(latest_pose,time.perf_counter(),
                    gain=float(self.tuning['body_gain'].get()),response=float(self.body_response.get()),
                    invert_x=bool(self.flags['invert_body_x'].get()),invert_y=bool(self.flags['invert_body_y'].get()))
                latest_pose = dict(latest_pose,**body)
            values = tracked_values(latest_pose, enabled)
            if values:
                self.apply_pose(values)
        alive = self.thread is not None and self.thread.is_alive()
        self.start_button.configure(state='disabled' if alive else 'normal')
        self.stop_button.configure(state='normal' if alive else 'disabled')
        self.camera_box.configure(state='disabled' if alive else 'normal')
        if self.was_alive and not alive:
            self.drive.set(False)
            if self.error is None:
                self.status.set('Capture stopped; manual controls active')
        self.was_alive = alive

    def close(self):
        self.closed = True
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
