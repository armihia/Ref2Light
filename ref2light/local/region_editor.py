from datetime import datetime, timezone
import json
import math
from pathlib import Path


REGIONS = ('left_eye', 'right_eye', 'mouth')
REGION_NAMES = {'left_eye': 'Screen-left eye', 'right_eye': 'Screen-right eye', 'mouth': 'Lips'}
REGION_COLORS = {'left_eye': '#28e0ff', 'right_eye': '#ffe052', 'mouth': '#ff6dbe'}


def canvas_box_to_normalized(start, end, image_size, scale, origin=(0., 0.)):
    """Map scroll-adjusted Canvas coordinates back to the image, allowing reverse drags."""
    width, height = image_size
    if min(width, height) <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Image dimensions and display scale must be positive')
    if not all(math.isfinite(float(x)) for point in (start, end, origin) for x in point):
        raise ValueError('Drag coordinates must be finite')
    points = [[max(0., min(1., (point[axis]-origin[axis])/scale/size))
               for axis, size in enumerate((width, height))] for point in (start, end)]
    box = [min(points[0][0], points[1][0]), min(points[0][1], points[1][1]),
           max(points[0][0], points[1][0]), max(points[0][1], points[1][1])]
    if (box[2]-box[0])*width < 1 or (box[3]-box[1])*height < 1:
        raise ValueError('The box is too small; include the entire eyelid or lips')
    return box


def _validate_tight_box(value):
    if len(value) != 4 or not all(math.isfinite(float(x)) and 0 <= x <= 1 for x in value):
        raise ValueError('Facial boxes must contain four finite normalized coordinates within the image')
    if value[2] <= value[0] or value[3] <= value[1]:
        raise ValueError('Facial boxes must have positive area')
    return [float(x) for x in value]


def detection_boxes(tight_boxes, image_size):
    """Match detector padding in aspect-correct coordinates to preserve eye height."""
    import numpy as np
    from ref2light.common.landmarks import region_box
    width, height = image_size
    if min(width, height) <= 0:
        raise ValueError('Ordinary-view dimensions must be positive')
    side = max(width, height)
    result = []
    for index, box in enumerate(tight_boxes):
        box = _validate_tight_box(box)
        points = np.asarray(box, dtype=np.float64).reshape(2, 2)*[width/side, height/side]
        padded = np.asarray(region_box(points, 1.5, .45 if index < 2 else .35))
        result.append(np.clip(padded*[side/width, side/height, side/width, side/height], 0, 1).tolist())
    if len(result) != 3:
        raise ValueError('Each view requires boxes for the screen-left eye, screen-right eye, and lips')
    return result


def build_manual_payload(request, annotations, mode='center'):
    """Copy center-view boxes to all views, or require all three views to be fully annotated."""
    from ref2light.common.render.output_regions import interpolate_boxes
    from ref2light.local.output_regions import BOX_SPACE, boxes_from_detection
    if mode not in ('center', 'three'):
        raise ValueError('The annotation mode must be center or three')
    count = int(request['view_count'])
    if count < 1:
        raise ValueError('The number of ordinary views must be positive')
    views = {int(view['index']): view for view in request['views']}
    required = [count//2] if mode == 'center' else sorted(set((0, count//2, count-1)))
    detected, annotated = {}, []
    for index in required:
        if index not in views:
            raise ValueError(f'Missing the ordinary image for view {index}')
        regions = annotations.get(index, {})
        if any(name not in regions for name in REGIONS):
            raise ValueError(f'View {index} needs complete screen-left eye, screen-right eye, and lip boxes')
        tight = [_validate_tight_box(regions[name]) for name in REGIONS]
        if tight[0][0]+tight[0][2] >= tight[1][0]+tight[1][2]:
            raise ValueError(f'The eye order is reversed in view {index}; left and right refer to the screen')
        detected[index] = detection_boxes(tight, (int(request['width']), int(request['height'])))
        annotated.append(dict(index=index, path=views[index]['path'], tight_boxes=tight, boxes=detected[index]))
    boxes = boxes_from_detection(interpolate_boxes(detected, count, 'cpu'))
    return dict(coordinate_space=BOX_SPACE, reference=str(Path(request['reference']).resolve()),
                alignment_enabled=bool(request['align']), boxes=boxes, source='manual_ordinary_views',
                request_id=request['request_id'], annotated_views=annotated,
                annotation_mode=mode, region_order=list(REGIONS),
                tight_box_padding=dict(factor=1.5, eye_min_height_width_ratio=.45, mouth_min_height_width_ratio=.35),
                view_assumption='Center-view boxes copied to all views; side views have not been verified' if mode == 'center' else 'Three views annotated; remaining views are linearly interpolated')


def save_manual_payload(request, annotations, mode='center'):
    payload = build_manual_payload(request, annotations, mode)
    destination = Path(request['destination'])
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = destination/f'manual_regions_{stamp}.json'
    with path.open('x', encoding='utf-8') as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
    return path


class RegionEditor:
    """Show ordinary views in a non-blocking Toplevel without modifying the native output window."""
    def __init__(self, parent, request, on_submit, on_cancel):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image
        self.request, self.on_submit, self.on_cancel = request, on_submit, on_cancel
        self.annotations, self.images = {}, {}
        self.scale, self.photo, self.drag = 1., None, None
        self.closed = False
        self.mode = tk.StringVar(parent, value='center')
        self.region = tk.StringVar(parent, value='left_eye')
        self.zoom = tk.StringVar(parent, value='100%')
        self.progress = tk.StringVar(parent)
        self.views = {int(view['index']): view for view in request['views']}
        self.center = int(request['view_count'])//2
        self.index = self.center
        self.view_labels = {i: f'{"Center" if i == self.center else "Leftmost" if i == 0 else "Rightmost"} view {i}' for i in self.views}
        self.selected_view = tk.StringVar(parent, value=self.view_labels[self.center])
        # Validate all images before creating the editor window.
        for index, view in self.views.items():
            with Image.open(view['path']) as image:
                if image.size != (request['width'], request['height']):
                    raise ValueError('Ordinary-view dimensions do not match the localization request; padded-image coordinates are invalid')
                self.images[index] = image.convert('RGB')
        self.window = tk.Toplevel(parent)
        self.window.title('Edit facial regions: '+Path(request['reference']).name)
        self.window.geometry(f'900x{min(900, max(600, parent.winfo_screenheight()-100))}')
        self.window.protocol('WM_DELETE_WINDOW', self.cancel)
        panel = ttk.Frame(self.window, padding=10)
        panel.pack(fill='both', expand=True)
        reason = 'Edit the current reference manually. ' if request.get('reason') == 'manual_adjustment' or not request.get('error') else 'Automatic localization failed. '
        ttk.Label(panel, text=reason+'Draw boxes for the screen-left eye, screen-right eye, and lips, in that order.\nInclude the full eyelids and lips without extra padding. Detector padding is applied automatically. Zoom and scroll as needed.', wraplength=860).pack(fill='x')
        ttk.Label(panel, text='Candidate: '+str(request['reference'])+f'; mild alignment: {"On" if request["align"] else "Off"}', wraplength=860).pack(fill='x', pady=4)
        choices = ttk.Frame(panel)
        choices.pack(fill='x')
        ttk.Radiobutton(choices, text='Annotate center view', variable=self.mode, value='center', command=self._change_mode).pack(side='left')
        ttk.Radiobutton(choices, text='Annotate three views', variable=self.mode, value='three', command=self._change_mode).pack(side='left')
        self.view_selector = ttk.Combobox(choices, textvariable=self.selected_view, values=list(self.view_labels.values()), state='disabled', width=18)
        self.view_selector.pack(side='left', padx=8)
        self.view_selector.bind('<<ComboboxSelected>>', self._change_view)
        tools = ttk.Frame(panel)
        tools.pack(fill='x', pady=4)
        for name in REGIONS:
            ttk.Radiobutton(tools, text=REGION_NAMES[name], variable=self.region, value=name).pack(side='left')
        ttk.Button(tools, text='Clear selected box', command=self._clear).pack(side='left', padx=6)
        ttk.Button(tools, text='Fit to window', command=self.fit).pack(side='left')
        zoom_selector = ttk.Combobox(tools, textvariable=self.zoom, values=('50%', '75%', '100%', '150%', '200%', '300%'), width=7, state='readonly')
        zoom_selector.pack(side='left', padx=6)
        zoom_selector.bind('<<ComboboxSelected>>', lambda event: self._set_scale(float(self.zoom.get().rstrip('%'))/100))
        area = ttk.Frame(panel)
        area.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(area, background='#242424', highlightthickness=0, width=700, height=600)
        horizontal = ttk.Scrollbar(area, orient='horizontal', command=self.canvas.xview)
        vertical = ttk.Scrollbar(area, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        area.rowconfigure(0, weight=1)
        area.columnconfigure(0, weight=1)
        self.canvas.bind('<ButtonPress-1>', self._start)
        self.canvas.bind('<B1-Motion>', self._drag)
        self.canvas.bind('<ButtonRelease-1>', self._end)
        self.canvas.bind('<MouseWheel>', lambda event: self.canvas.yview_scroll(-int(event.delta/120), 'units'))
        ttk.Label(panel, textvariable=self.progress, wraplength=860).pack(fill='x', pady=4)
        ttk.Label(panel, text='Center-only annotations are copied to all views. Annotate both side views when possible. Cancel keeps the current reference.', wraplength=860).pack(fill='x')
        buttons = ttk.Frame(panel)
        buttons.pack(fill='x', pady=4)
        ttk.Button(buttons, text='Save and apply reference', command=self.submit).pack(side='left')
        ttk.Button(buttons, text='Cancel and keep reference', command=self.cancel).pack(side='left', padx=8)
        self.window.update_idletasks()
        self.fit()

    def _set_scale(self, scale):
        self.scale = max(.05, min(4., scale))
        self.zoom.set(f'{self.scale*100:.0f}%')
        self._redraw()

    def fit(self):
        width, height = self.images[self.index].size
        self._set_scale(min(max(20, self.canvas.winfo_width())/width, max(20, self.canvas.winfo_height())/height))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _redraw(self):
        from PIL import Image, ImageTk
        image = self.images[self.index]
        size = tuple(max(1, round(value*self.scale)) for value in image.size)
        self.photo = ImageTk.PhotoImage(image.resize(size, Image.Resampling.LANCZOS), master=self.window)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, *size))
        for name, box in self.annotations.get(self.index, {}).items():
            coords = [box[i]*image.size[i % 2]*self.scale for i in range(4)]
            self.canvas.create_rectangle(*coords, outline=REGION_COLORS[name], width=2)
            self.canvas.create_text(coords[0]+2, coords[1]-2, text=REGION_NAMES[name], anchor='sw', fill=REGION_COLORS[name])
        self.progress.set('; '.join(f'{self.view_labels[index]}: {len(self.annotations.get(index, {}))}/3 boxes' for index in sorted(self.views)))

    def _change_mode(self):
        self.view_selector.configure(state='readonly' if self.mode.get() == 'three' else 'disabled')
        if self.mode.get() == 'center':
            self.index = self.center
            self.selected_view.set(self.view_labels[self.center])
        self.drag = None
        self._redraw()

    def _change_view(self, event=None):
        self.index = next(index for index, label in self.view_labels.items() if label == self.selected_view.get())
        self.drag = None
        self._redraw()

    def _point(self, event):
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _start(self, event):
        point = self._point(event)
        width, height = self.images[self.index].size
        if 0 <= point[0] <= width*self.scale and 0 <= point[1] <= height*self.scale:
            self.drag = point

    def _drag(self, event):
        if self.drag is not None:
            self.canvas.delete('drag')
            self.canvas.create_rectangle(*self.drag, *self._point(event), outline=REGION_COLORS[self.region.get()], width=2, dash=(4, 2), tags='drag')

    def _end(self, event):
        if self.drag is None:
            return
        try:
            box = canvas_box_to_normalized(self.drag, self._point(event), self.images[self.index].size, self.scale)
            name = self.region.get()
            self.annotations.setdefault(self.index, {})[name] = box
            if name != 'mouth':
                self.region.set(REGIONS[REGIONS.index(name)+1])
            self._redraw()
        except ValueError as exc:
            self.progress.set(str(exc))
            self.canvas.delete('drag')
        finally:
            self.drag = None

    def _clear(self):
        self.annotations.get(self.index, {}).pop(self.region.get(), None)
        self._redraw()

    def submit(self):
        from tkinter import messagebox
        try:
            path = save_manual_payload(self.request, self.annotations, self.mode.get())
            self.on_submit(dict(request_id=self.request['request_id'], reference=self.request['reference'],
                                align=self.request['align'], output_boxes=str(path.resolve())))
            self.dismiss()
        except Exception as exc:
            messagebox.showerror('Cannot apply facial regions', str(exc), parent=self.window)

    def cancel(self):
        if not self.closed:
            self.on_cancel(dict(request_id=self.request['request_id']))
            self.dismiss()

    def dismiss(self):
        """Close a superseded editor without cancelling the newer renderer request."""
        if not self.closed:
            self.closed = True
            self.window.destroy()
