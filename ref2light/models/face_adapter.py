from __future__ import annotations


import torch


def amp(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)

