from collections import deque
import multiprocessing as mp
import math
import os
from pathlib import Path
import queue
import time


def complete_mouth_values(amount):
    """Map the combined slider from closed to small O to wide open; mouth area is not linear."""
    amount = float(amount)
    if not math.isfinite(amount):
        raise ValueError('Combined mouth opening must be finite')
    amount = max(0., min(1., amount))
    return dict(mouthOpen=.75*min(2*amount, 1.), mouthCurve=.8*max(2*amount-1., 0.))


def capture_mouth_values(values, linked=True):
    """Change forwarded values only; mouthOpen controls opening independently when linking is disabled."""
    result = dict(values)
    if linked and 'mouthOpen' in result:
        result.update(complete_mouth_values(float(result['mouthOpen'])/.75))
    return result


def complete_mouth_position(opened, curve):
    """Invert the slider only when both mouth axes lie on the combined control path."""
    opened, curve = float(opened), float(curve)
    if abs(curve) <= 1e-7 and 0 <= opened <= .75:
        return .5*opened/.75
    if abs(opened-.75) <= 1e-7 and 0 <= curve <= .8:
        return .5+.5*curve/.8
    return None


class PendingLocalization:
    """Accept manual annotations only for the latest candidate reference."""
    def __init__(self):
        self.request = None

    def clear(self):
        previous, self.request = self.request, None
        return previous

    def begin(self, request):
        self.request = dict(request)

    def cancel(self, value):
        if self.request and value.get('request_id') == self.request['request_id']:
            self.clear()
            return True
        return False

    def take(self, value):
        request = self.request
        if (request is None or value.get('request_id') != request['request_id']
                or Path(value.get('reference', '')).resolve() != Path(request['reference']).resolve()
                or value.get('align') is not request['align'] or not value.get('output_boxes')):
            return None
        self.clear()
        return request


def renderer(args, shared, lock, running, paused, commands, status, diagnostics_factory=None, debug_enabled=None):
    from PIL import Image
    import torch
    from ref2light.ui.display import NativeDisplay
    from ref2light.ui.display_nvidia import NvidiaDisplay
    from ref2light.ui.transfer import NativeRGBTransfer
    from ref2light.ui.transfer_nvidia import NvidiaTransfer
    torch.set_num_threads(2)
    from ref2light.common.capture_backend import find_native_monitor, set_dpi_awareness
    native_frame_gpu = NvidiaTransfer() if args.display_backend=='nvidia' else NativeRGBTransfer()
    from ref2light.common.io import timestamp
    from ref2light.common.io import write_json
    from ref2light.common.controls import POSE_KEYS, NEUTRAL, limit_tk_head
    from ref2light.runtime.session import Session, ReferenceLocalizationError
    import pygame
    destination = Path(args.save_root)/timestamp()
    destination.mkdir(parents=True, exist_ok=False)
    session = last = display = None
    diagnostics = diagnostics_factory(destination) if diagnostics_factory is not None else None
    last_status = 0.
    try:
        status.put(('status', 'Loading the complete model and preparing the reference...'))
        session = Session(args.model, args.ref, args.device,
                          args.landmark_model, destination, args.align,
                          output_boxes=args.output_boxes, roi_tracking=args.roi_tracking,
                          local_graphs=args.local_graphs, body_graphs=args.body_graphs,
                          defer_reference=True, detector_recovery=getattr(args, 'detector_recovery', True))
        set_dpi_awareness()
        monitor = dict(left=0,top=0) if getattr(args,'_test_hidden',False) else find_native_monitor(*session.size)
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{monitor['left']},{monitor['top']}"
        pygame.init()
        display = (NvidiaDisplay if args.display_backend=='nvidia' else NativeDisplay)(session.size, (monitor['left'],monitor['top']), args.display_backend,
                                hidden=getattr(args,'_test_hidden',False))
        dirty, previous, save_pending = True, None, False
        next_frame = 0.
        completions = deque(maxlen=30)
        pending = PendingLocalization()
        def reference_ready(reset_pose=True):
            source_pose = session.reference_report.get('source_pose')
            with lock:
                if reset_pose:
                    raw = limit_tk_head(source_pose if source_pose is not None else [NEUTRAL[k] for k in POSE_KEYS])
                    shared[:] = raw
                else:
                    raw = list(shared[:])
            regions = session.reference_report.get('output_regions', {})
            status.put(('reference', dict(path=session.reference, source_pose=source_pose, raw_pose=raw, align=session.align,
                detector_recovered=bool(regions.get('recovery', {}).get('succeeded', False)),
                manual_regions=regions.get('source') == 'manual_output_boxes',
                brow_error=session._brow_error, brow_estimated=bool((session._brow_layout or {}).get('estimated')))))
        def clear_pending():
            previous_request = pending.clear()
            if previous_request:
                status.put(('localization_cleared', previous_request['request_id']))
        def require_localization(request):
            pending.begin(request)
            status.put(('localization_required', request))
        def is_current(path):
            return hasattr(session, 'reference') and Path(path).resolve() == Path(session.reference).resolve()
        def select_reference(path, align=None, output_boxes=None, preserve_pose=False):
            nonlocal dirty, previous, last, save_pending
            clear_pending()
            try:
                session.set_reference(path, align=align, output_boxes=output_boxes)
                if getattr(args, 'brow_boxes', None) and Path(path).resolve() == Path(args.ref).resolve():
                    session.set_brow_boxes(args.brow_boxes)
                dirty, previous, last, save_pending = True, None, None, False
                completions.clear()
                reference_ready(reset_pose=not preserve_pose)
                return True
            except ReferenceLocalizationError as exc:
                require_localization(exc.request)
                return False
            except Exception as exc:
                kept = 'keeping the current reference' if hasattr(session, 'reference') else 'waiting for a reference image'
                status.put(('status', f'Reference preparation failed, {kept}: {exc}'))
                return False
        status.put(('status', f"Loaded complete model at step {session.payload['step']}; preparing the reference."))
        select_reference(args.ref)
        while running.value:
            if diagnostics is not None:
                diagnostics.set_enabled(bool(debug_enabled.value))
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running.value = False
            try:
                while True:
                    name, value = commands.get_nowait()
                    if diagnostics is not None:
                        diagnostics.invalidate()
                    if name == 'reference':
                        if isinstance(value, dict):
                            select_reference(value['reference'], align=value['align'])
                        else:
                            select_reference(value)
                    elif name == 'reference_boxes':
                        request = pending.take(value)
                        if request is None:
                            status.put(('status', 'This region request is stale or belongs to another candidate; the reference was not changed.'))
                        else:
                            status.put(('localization_cleared', request['request_id']))
                            select_reference(request['reference'], align=request['align'], output_boxes=value['output_boxes'],
                                             preserve_pose=is_current(request['reference']))
                    elif name == 'reference_cancel':
                        if pending.cancel(value):
                            status.put(('localization_cleared', value['request_id']))
                            status.put(('status', 'Annotation cancelled; keeping the current reference.' if hasattr(session, 'reference') else 'Annotation cancelled; select a reference image.'))
                    elif name == 'edit_regions':
                        if hasattr(session, 'reference'):
                            clear_pending()
                            try:
                                require_localization(session.manual_reference_request())
                            except Exception as exc:
                                status.put(('status', 'Ordinary-view preparation failed; keeping the current reference: '+str(exc)))
                        else:
                            status.put(('status', 'No reference is loaded. Annotate the pending preview or choose another image.'))
                    elif name == 'edit_brows':
                        if hasattr(session, 'reference'):
                            status.put(('brow_localization', session.manual_brow_request()))
                    elif name == 'brow_boxes':
                        if Path(value['reference']).resolve() != Path(session.reference).resolve() or value['alignment'] != session.align:
                            status.put(('status', 'The eyebrow boxes belong to a previous reference; the current image was not changed.'))
                        else:
                            try:
                                session.set_brow_boxes(value['path'])
                                dirty, previous, last = True, None, None
                                status.put(('status', 'Eyebrow regions applied; eye and mouth regions were preserved.'))
                            except Exception as exc:
                                status.put(('status', 'Could not apply eyebrow regions: '+str(exc)))
                    elif name == 'output_boxes':
                        try:
                            if pending.request:
                                request = pending.request
                                select_reference(request['reference'], align=request['align'], output_boxes=value,
                                                 preserve_pose=is_current(request['reference']))
                            elif hasattr(session, 'reference'):
                                session.set_output_boxes(value)
                                dirty, previous, last = True, None, None
                                completions.clear()
                                status.put(('status', 'Loaded manual ordinary-view regions'))
                            else:
                                status.put(('status', 'Select a reference before loading its ordinary-view regions.'))
                        except Exception as exc:
                            status.put(('status', 'Could not load output regions: '+str(exc)))
                    elif name == 'save':
                        save_pending = True
                    elif name == 'alignment':
                        candidate = pending.request['reference'] if pending.request else getattr(session, 'reference', args.ref)
                        select_reference(candidate, align=value, preserve_pose=is_current(candidate))
                        if hasattr(session, 'reference'):
                            status.put(('alignment', session.align))
            except queue.Empty:
                pass
            with lock:
                raw = limit_tk_head(shared[:])
            if hasattr(session, 'reference') and running.value and not paused.value and (dirty or raw != previous) and time.perf_counter() >= next_frame:
                start = time.perf_counter()
                tensor = session.render(raw)
                frame = native_frame_gpu(tensor, session.size)
                ready = time.perf_counter()
                display.draw(frame)
                end = time.perf_counter()
                elapsed = (end-start)*1000
                if completions and start-completions[-1] > .5:
                    completions.clear()
                completions.append(end)
                rate = (len(completions)-1)/(completions[-1]-completions[0]) if len(completions) > 1 else None
                info = dict(session.metadata(), raw_pose=list(raw), render_and_transfer_ms=(ready-start)*1000,
                            display_ms=(end-ready)*1000, total_ms=elapsed, update_fps=rate,display_backend=args.display_backend)
                last = frame, info
                if diagnostics is not None:
                    diagnostics.record(info)
                rate_text = f'{rate:.1f} FPS' if rate is not None else 'Waiting for continuous motion'
                if end-last_status >= .2:
                    status.put(('frame', f"Frame {elapsed:.1f} ms; display update rate: {rate_text}"))
                    last_status = end
                previous, dirty = raw, False
                next_frame = max(start+1/args.fps, end)
            if save_pending and last is not None:
                saved = destination/('frame_'+timestamp())
                saved.mkdir()
                Image.fromarray(last[0]).save(saved/'native.png',compress_level=1)
                write_json(saved/'metadata.json', last[1])
                status.put(('status', 'Saved: '+str(saved)))
                save_pending = False
            time.sleep(.001)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        write_json(destination/'error.json', dict(error=repr(exc)))
        status.put(('error', str(exc)))
    finally:
        if display is not None: display.close()
        pygame.quit()
        running.value = False
        if diagnostics is not None:
            archive = diagnostics.finish(last, enabled=bool(debug_enabled.value))
            if archive is not None:
                print('Diagnostics archive: ', archive, flush=True)


def gui(args, diagnostics_factory=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from ref2light.common.controls import POSE_KEYS, NEUTRAL, PRESETS, apply_preset, limit_tk_head
    from ref2light.ui.widgets import build_controls
    from ref2light.ui.capture import CapturePanel
    from ref2light.common.capture_backend import set_dpi_awareness
    from ref2light.local.region_editor import RegionEditor
    set_dpi_awareness()
    root = tk.Tk()
    root.title('Ref2Light: Single-image light-field character control')
    root.geometry('1050x900+20+20')
    context = mp.get_context('spawn')
    shared = context.Array('d', [NEUTRAL[k] for k in POSE_KEYS], lock=False)
    lock, running, paused = context.Lock(), context.Value('b', True), context.Value('b', False)
    commands, status = context.Queue(), context.Queue()
    debug_enabled = context.Value('b', False) if diagnostics_factory is not None else None
    process = context.Process(target=renderer, args=(args, shared, lock, running, paused, commands, status,
                                                    diagnostics_factory, debug_enabled), daemon=True)
    panel = ttk.Frame(root, padding=10)
    panel.pack(fill='both', expand=True)
    text = tk.StringVar(value='Starting...')
    timing = tk.StringVar(value='Waiting for the first frame')
    reference = tk.StringVar(value=args.ref)
    aligned = tk.BooleanVar(value=args.align)
    region_editor = None
    wanted_reference = str(Path(args.ref).resolve())
    dismissed_requests = set()
    def close_editor():
        nonlocal region_editor
        if region_editor is not None:
            dismissed_requests.add(region_editor.request['request_id'])
            region_editor.dismiss()
            region_editor = None
    for variable in (text, timing, reference):
        ttk.Label(panel, textvariable=variable, wraplength=1000).pack(fill='x', pady=3)
    actions = ttk.Frame(panel)
    actions.pack(fill='x')
    def choose():
        nonlocal wanted_reference
        path = filedialog.askopenfilename(filetypes=[('Reference image', '*.png *.jpg *.jpeg *.webp')])
        if path:
            capture.pause_drive()
            close_editor()
            wanted_reference = str(Path(path).resolve())
            commands.put(('reference', dict(reference=path, align=aligned.get())))
            paused.value = False
            text.set('Loading and encoding a new reference...')
    ttk.Button(actions, text='Change reference', command=choose).pack(side='left', padx=4)
    ttk.Button(actions, text='Save native frame', command=lambda: commands.put(('save', None))).pack(side='left', padx=4)
    ttk.Button(actions, text='Pause / Resume', command=lambda: setattr(paused, 'value', not paused.value)).pack(side='left', padx=4)
    def toggle_alignment():
        nonlocal wanted_reference
        capture.pause_drive()
        wanted_reference = str(Path(region_editor.request['reference'] if region_editor is not None and not region_editor.closed else reference.get()).resolve())
        close_editor()
        commands.put(('alignment', aligned.get()))
        paused.value = False
        text.set('Changing input preprocessing while preserving the current pose...')
    ttk.Checkbutton(actions, text='Mild alignment', variable=aligned, command=toggle_alignment).pack(side='left', padx=4)
    if diagnostics_factory is not None:
        debug = tk.BooleanVar(value=False)
        ttk.Checkbutton(actions, text='Debug (ZIP on exit)', variable=debug,
                        command=lambda: setattr(debug_enabled, 'value', debug.get())).pack(side='left', padx=4)
    def choose_boxes():
        path = filedialog.askopenfilename(filetypes=[('Ordinary-view regions', '*.json')])
        if path:
            commands.put(('output_boxes', path))
    def edit_regions():
        nonlocal wanted_reference
        capture.pause_drive()
        close_editor()
        wanted_reference = str(Path(reference.get()).resolve())
        commands.put(('edit_regions', None))
        text.set('Preparing ordinary views for manual region editing...')
    edit_row = ttk.Frame(panel)
    edit_row.pack(fill='x', pady=4)
    edit_button = ttk.Button(edit_row, text='Edit facial regions', command=edit_regions, state='disabled')
    edit_button.pack(side='left', padx=4)
    ttk.Button(edit_row, text='Load boxes JSON', command=choose_boxes).pack(side='left', padx=4)
    ttk.Button(edit_row, text='Edit eyebrow regions', command=lambda: commands.put(('edit_brows', None))).pack(side='left', padx=4)
    edit_help = ttk.Label(edit_row, text='Edit regions in ordinary views; native output stays at 1:1.', wraplength=300)
    edit_help.pack(side='left', padx=4)
    pages = ttk.Notebook(panel)
    pages.pack(fill='both', expand=True)
    manual, capture_page = ttk.Frame(pages), ttk.Frame(pages)
    pages.add(manual, text='Manual controls')
    pages.add(capture_page, text='Camera capture')
    capture_scroll = ttk.Scrollbar(capture_page, orient='vertical')
    capture_scroll.pack(side='right', fill='y')
    capture_canvas = tk.Canvas(capture_page, width=1, height=1, highlightthickness=0,
                               yscrollcommand=capture_scroll.set)
    capture_canvas.pack(side='left', fill='both', expand=True)
    capture_scroll.configure(command=capture_canvas.yview)
    capture_content = ttk.Frame(capture_canvas)
    capture_window = capture_canvas.create_window(0, 0, window=capture_content, anchor='nw')
    capture_content.bind('<Configure>', lambda event: capture_canvas.configure(
        scrollregion=capture_canvas.bbox('all')))
    capture_canvas.bind('<Configure>', lambda event: capture_canvas.itemconfigure(
        capture_window, width=event.width))
    variables = {key: tk.DoubleVar(root, value=NEUTRAL[key]) for key in POSE_KEYS}
    complete_amount = tk.DoubleVar(root, value=0.)
    complete_label = tk.StringVar(root, value='0.00: closed -> 0.50: small O -> 1.00: wide open')
    linked_capture = tk.BooleanVar(root, value=True)
    synchronizing_complete = False
    controls = None
    last_controls_refresh=0.
    controls_dirty=False
    def update_complete_indicator():
        nonlocal synchronizing_complete
        amount = complete_mouth_position(variables['mouthOpen'].get(), variables['mouthCurve'].get())
        if amount is None:
            complete_label.set('The two mouth axes are independent; the combined slider is inactive')
        else:
            synchronizing_complete = True
            complete_amount.set(amount)
            synchronizing_complete = False
            complete_label.set(f'{amount:.2f}: 0 closed -> 0.50 small O -> 1.00 wide open')
    def refresh_controls():
        nonlocal last_controls_refresh,controls_dirty
        now=time.perf_counter()
        if controls_dirty and now-last_controls_refresh>=.05:
            if controls and manual.winfo_ismapped():controls['refresh']()
            update_complete_indicator()
            last_controls_refresh=now;controls_dirty=False
    def push():
        nonlocal controls_dirty
        raw = limit_tk_head([variables[key].get() for key in POSE_KEYS])
        with lock:
            shared[:] = raw
        for key, value in zip(POSE_KEYS[:3], raw[:3]):
            if variables[key].get() != value:
                variables[key].set(value)
        # Write poses immediately, throttle UI redraws to 20 Hz, and flush the final state during polling.
        controls_dirty=True
        refresh_controls()
    def apply_complete(value):
        if synchronizing_complete:
            return
        capture.pause_drive()
        for key, amount in complete_mouth_values(value).items():
            variables[key].set(amount)
        push()
        paused.value = False
    complete_bar = ttk.LabelFrame(manual, text='Mouth opening (combined control; area is nonlinear)', padding=6)
    complete_bar.pack(fill='x', pady=4)
    ttk.Scale(complete_bar, from_=0., to=1., variable=complete_amount, command=apply_complete).pack(fill='x')
    ttk.Label(complete_bar, textvariable=complete_label).pack(side='left', padx=4)
    for label, amount in (('Close mouth', 0.), ('Small O', .5), ('Wide open', 1.)):
        ttk.Button(complete_bar, text=label, command=lambda amount=amount: apply_complete(amount)).pack(side='left', padx=3)
    controls = build_controls(manual, variables, push)
    def captured(values):
        for key, value in capture_mouth_values(values, linked_capture.get()).items():
            variables[key].set(value)
        push()
    def toggle_mouth_link():
        if not linked_capture.get():
            variables['mouthCurve'].set(0.)
            push()
        text.set('Capture now links closed, small-O, and wide-open mouth controls' if linked_capture.get() else 'Capture uses mouthOpen only; rad is zero')
    ttk.Checkbutton(capture_content, text='Link full mouth opening during capture (off: mouthOpen only)',
                    variable=linked_capture, command=toggle_mouth_link).pack(fill='x', padx=8, pady=6)
    capture = CapturePanel(capture_content, captured, camera=args.camera, body_mode=args.body_mode,
                           on_start=lambda: setattr(paused, 'value', False))
    for pad in controls['pads']:
        pad.canvas.bind('<Button-1>', lambda event: capture.pause_drive(), add='+')
    for scale in controls['scales'].values():
        scale.bind('<Button-1>', lambda event: capture.pause_drive(), add='+')
    row = ttk.Frame(manual)
    row.pack(fill='x', pady=4)
    def preset(name):
        capture.pause_drive()
        values = apply_preset([variables[k].get() for k in POSE_KEYS], name)
        for key, value in zip(POSE_KEYS, values):
            variables[key].set(value)
        push()
        paused.value = False
    for index, name in enumerate(PRESETS):
        label = 'Dataset neutral pose' if name == 'Neutral reset' else name
        ttk.Button(row, text=label, command=lambda name=name: preset(name)).grid(row=index//6, column=index%6, padx=2, pady=3)
    known_source = None
    def source_reset():
        if known_source is not None:
            capture.pause_drive()
            for key, value in zip(POSE_KEYS, known_source):
                variables[key].set(value)
            push()
            paused.value = False
    source_button = ttk.Button(edit_row, text='Reset source pose', command=source_reset, state='disabled')
    source_button.pack(side='left', padx=4, before=edit_help)
    ttk.Label(panel, text='Head yaw, pitch, and roll are limited to +/-10 degrees for manual controls, capture, and source-pose reset. Controls use absolute teacher-pose parameters. Reset may change an unseen reference pose. Alignment preserves visible content; it does not complete missing body parts. Check the result on the target display.', wraplength=1000).pack(fill='x')
    def poll():
        nonlocal known_source, region_editor
        capture.poll()
        refresh_controls()
        try:
            while True:
                kind, value = status.get_nowait()
                if kind == 'frame':
                    timing.set(value)
                elif kind == 'reference':
                    reference.set(value['path'])
                    aligned.set(value['align'])
                    known_source = value['source_pose']
                    source_button.configure(state='normal' if known_source is not None else 'disabled')
                    edit_button.configure(state='normal')
                    for key, position in zip(POSE_KEYS, value['raw_pose']):
                        variables[key].set(position)
                    push()
                    method = 'manual regions applied' if value.get('manual_regions') else 'CPU recovery succeeded after regular detection failed' if value.get('detector_recovered') else 'facial regions detected'
                    pose_note = 'using the recorded source pose' if known_source is not None else 'no source-pose metadata; using the dataset default pose'
                    brow_note = ('eyebrow detection failed; mark both brows manually; eye and mouth controls remain available' if value.get('brow_error') else
                                 'eyebrows estimated from reference eye-brow geometry; manual editing is available' if value.get('brow_estimated') else 'eyebrows detected')
                    text.set(f'Reference ready; {method}; {brow_note}; {pose_note}. Check the target display.')
                elif kind == 'brow_localization':
                    from ref2light.brows.brow_editor import BrowEditor
                    if str(Path(value['reference']).resolve()) != str(Path(reference.get()).resolve()):
                        continue
                    capture.pause_drive()
                    BrowEditor(root, value, lambda message: commands.put(('brow_boxes', message)))
                elif kind == 'localization_required':
                    if value['request_id'] in dismissed_requests or str(Path(value['reference']).resolve()) != wanted_reference:
                        continue
                    close_editor()
                    capture.pause_drive()
                    def apply_regions(message):
                        commands.put(('reference_boxes', message))
                        dismissed_requests.add(message['request_id'])
                        paused.value = False
                        text.set('Applying manual regions to the candidate reference...')
                    def cancel_regions(message):
                        commands.put(('reference_cancel', message))
                        dismissed_requests.add(message['request_id'])
                        text.set('Annotation cancelled; the current reference was preserved.')
                    try:
                        region_editor = RegionEditor(root, value, apply_regions, cancel_regions)
                        text.set('Mark both eyes and lips in the ordinary-view window. Save to apply the candidate reference.')
                    except Exception as exc:
                        commands.put(('reference_cancel', dict(request_id=value['request_id'])))
                        text.set('Could not open the region editor: '+str(exc))
                        messagebox.showerror('Ordinary-view region editor', str(exc), parent=root)
                elif kind == 'localization_cleared':
                    if region_editor is not None and region_editor.request['request_id'] == value:
                        close_editor()
                elif kind == 'error':
                    messagebox.showerror('Ref2Light', value, parent=root)
                    text.set(value)
                elif kind == 'alignment':
                    aligned.set(value)
                    text.set('Mild alignment enabled; weights and pose preserved' if value else 'Original framing restored; weights and pose preserved')
                else:
                    text.set(value)
        except queue.Empty:
            pass
        if not running.value:
            capture.stop()
        root.after(10, poll)
    def close():
        close_editor()
        capture.stop()
        running.value = False
        root.withdraw()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
        root.destroy()
    root.protocol('WM_DELETE_WINDOW', close)
    process.start()
    root.after(10, poll)
    root.mainloop()
