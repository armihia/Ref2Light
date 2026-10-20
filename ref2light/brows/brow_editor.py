from pathlib import Path
from ref2light.common.io import write_json
from ref2light.local.region_editor import canvas_box_to_normalized


class BrowEditor:
    def __init__(self, parent, request, on_submit):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from PIL import Image, ImageTk
        self.tk, self.messagebox, self.ImageTk = tk, messagebox, ImageTk
        self.request, self.on_submit = request, on_submit
        view = next(v for v in request['views'] if v['index'] == request['view_count']//2)
        self.image = Image.open(view['path']).convert('RGB')
        self.window = tk.Toplevel(parent)
        self.window.title('Eyebrow regions in ordinary views')
        self.window.geometry('780x850')
        self.group = tk.IntVar(value=0)
        self.boxes = {}
        panel = ttk.Frame(self.window, padding=8); panel.pack(fill='both', expand=True)
        ttk.Label(panel, text='Mark the screen-left and screen-right eyebrows, including room for vertical movement. Exclude the eyes. The center-view boxes are copied to all views.', wraplength=720).pack(fill='x')
        row = ttk.Frame(panel); row.pack(fill='x')
        for i, name in enumerate(('Screen-left brow', 'Screen-right brow')):
            ttk.Radiobutton(row, text=name, variable=self.group, value=i).pack(side='left')
        self.scale = min(700/self.image.width, 680/self.image.height)
        self.photo = ImageTk.PhotoImage(self.image.resize((round(self.image.width*self.scale), round(self.image.height*self.scale))))
        self.canvas = tk.Canvas(panel, width=self.photo.width(), height=self.photo.height(), background='#222222')
        self.canvas.pack()
        self.canvas.bind('<ButtonPress-1>', self.start)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.end)
        ttk.Button(panel, text='Save and apply eyebrow regions', command=self.submit).pack(side='left', pady=8)
        ttk.Button(panel, text='Cancel', command=self.window.destroy).pack(side='left', padx=8)
        self.redraw()

    def redraw(self):
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        for i, box in self.boxes.items():
            coords = [box[0]*self.photo.width(), box[1]*self.photo.height(), box[2]*self.photo.width(), box[3]*self.photo.height()]
            self.canvas.create_rectangle(*coords, outline=('#28e0ff', '#ffe052')[i], width=2)

    def start(self, event):
        self.begin = (event.x, event.y)
        self.temp = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='white', width=2)

    def drag(self, event):
        if hasattr(self, 'temp'): self.canvas.coords(self.temp, *self.begin, event.x, event.y)

    def end(self, event):
        try:
            self.boxes[self.group.get()] = canvas_box_to_normalized(self.begin, (event.x, event.y), self.image.size, self.scale)
        except ValueError as exc:
            self.messagebox.showerror('Eyebrow regions', str(exc), parent=self.window)
        self.redraw()

    def submit(self):
        if set(self.boxes) != {0, 1}:
            self.messagebox.showerror('Eyebrow regions', 'Mark both eyebrows first', parent=self.window); return
        if self.boxes[0][0]+self.boxes[0][2] >= self.boxes[1][0]+self.boxes[1][2]:
            self.messagebox.showerror('Eyebrow regions', 'The eyebrow order is reversed; left and right refer to the screen', parent=self.window); return
        path = Path(self.request['destination'])/'manual_brows.json'
        write_json(path, dict(reference=self.request['reference'], alignment_enabled=self.request['align'],
                             view_boxes=[[self.boxes[0], self.boxes[1]]]*self.request['view_count'],
                             source='manual_center_brows', estimated=True))
        self.on_submit(dict(reference=self.request['reference'], alignment=self.request['align'], path=str(path)))
        self.window.destroy()
