from functools import lru_cache


import numpy as np


import torch


from ref2light.reference.reference import square_rgb, canvas_array


@lru_cache(maxsize=8)
def reference_image(path):
    return square_rgb(path)


def ref_tensor(image):
    return torch.from_numpy(canvas_array(image)).permute(2, 0, 1).contiguous()


