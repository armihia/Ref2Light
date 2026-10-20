from ref2light.common.controls import RANGES, LABELS

def build_controls(parent, variables, callback):
    """Build manual controls independently of model initialization."""
    import tkinter as tk
    from tkinter import ttk
    pads, scales, labels = [], {}, {}
    class XYPad(ttk.Frame):
        def __init__(self, owner, title, xkey, ykey):
            super().__init__(owner)
            self.xkey, self.ykey = xkey, ykey
            self.width, self.height, self.padding = 270, 145, 10
            ttk.Label(self, text=title).pack(anchor='w')
            self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg='#f7f7f7',
                                    highlightthickness=1, highlightbackground='#aaa')
            self.canvas.pack(pady=3)
            self.value = ttk.Label(self)
            self.value.pack()
            for event in ('<Button-1>', '<B1-Motion>'):
                self.canvas.bind(event, self.drag)
            self.canvas.bind('<Double-Button-1>', lambda event: self.set(0., 0.))
            self.redraw()
        def set(self, x, y):
            variables[self.xkey].set(x)
            variables[self.ykey].set(y)
            callback()
        def drag(self, event):
            p, w, h = self.padding, self.width, self.height
            x, y = max(p, min(w-p, event.x)), max(p, min(h-p, event.y))
            lo, hi = RANGES[self.xkey]
            xv = lo+(x-p)/(w-2*p)*(hi-lo)
            lo, hi = RANGES[self.ykey]
            self.set(xv, hi-(y-p)/(h-2*p)*(hi-lo))
        def redraw(self):
            p, w, h = self.padding, self.width, self.height
            xv, yv = variables[self.xkey].get(), variables[self.ykey].get()
            lo, hi = RANGES[self.xkey]
            x = p+(xv-lo)/(hi-lo)*(w-2*p)
            lo, hi = RANGES[self.ykey]
            y = p+(hi-yv)/(hi-lo)*(h-2*p)
            self.canvas.delete('all')
            self.canvas.create_line(p, h/2, w-p, h/2, fill='#bbb')
            self.canvas.create_line(w/2, p, w/2, h-p, fill='#bbb')
            self.canvas.create_oval(x-6, y-6, x+6, y+6, fill='#333', outline='')
            self.value.configure(text=f'X {xv:+.2f}   Y {yv:+.2f}')
    panel = ttk.Frame(parent)
    panel.pack(fill='x', pady=(0, 9))
    for column, (title, xkey, ykey) in enumerate((
        ('Head: yaw / pitch (+/-10 deg)', 'headYawDeg', 'headPitchDeg'),
        ('Body: yaw / pitch', 'bodyYawDeg', 'bodyPitchDeg'),
        ('Gaze: horizontal / vertical', 'gazeX', 'gazeY'))):
        pad = XYPad(panel, title, xkey, ykey)
        pad.grid(row=0, column=column, padx=7)
        pads.append(pad)
    scalar = ttk.LabelFrame(parent, text='Roll and expression controls', padding=8)
    scalar.pack(fill='x')
    keys = ('headRollDeg', 'bodyRollDeg', 'eyesOpen', 'mouthOpen',
            'mouthCurve', 'browHeight', 'browDistance', 'browCurve')
    for column in (1, 4):
        scalar.columnconfigure(column, weight=1)
    for index, key in enumerate(keys):
        row, column = index//2, (index%2)*3
        ttk.Label(scalar, text=LABELS[key], width=12).grid(row=row, column=column, sticky='w')
        scale = tk.Scale(scalar, from_=RANGES[key][0], to=RANGES[key][1],
            resolution=.1 if key.endswith('Deg') else .01, orient='horizontal',
            variable=variables[key], showvalue=False, length=270, highlightthickness=0,
            command=lambda value: callback())
        scale.grid(row=row, column=column+1, sticky='ew', pady=3)
        label = ttk.Label(scalar, width=7, anchor='e')
        label.grid(row=row, column=column+2, padx=(3, 10))
        scales[key], labels[key] = scale, label
    def refresh():
        for pad in pads:
            pad.redraw()
        for key, label in labels.items():
            label.configure(text=f'{variables[key].get():+.2f}')
    refresh()
    return dict(pads=pads, scales=scales, refresh=refresh)
