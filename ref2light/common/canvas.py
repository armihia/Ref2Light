import numpy as np
from PIL import Image

def canonical_ref(arr, out_w=288, out_h=512):
    # Scale the square reference uniformly and center it on a portrait black canvas.
    im = Image.fromarray(arr, "RGB").resize((out_w, out_w), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    top = (out_h - out_w) // 2
    canvas.paste(im, (0, top))
    return np.asarray(canvas, dtype=np.uint8).copy()
