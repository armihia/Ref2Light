import torch
import torch.nn as nn
import torch.nn.functional as F

class DisplayInterleaver(nn.Module):
    """Sample K anchors using the calibration equations from CubeVi MultiView.shader to produce a 1440 x 2560 interlaced image."""
    def __init__(self, calibration, anchors=12):
        super().__init__()
        self.anchors = int(anchors)
        self.native_w = int(calibration["native_width"])
        self.native_h = int(calibration["native_height"])
        self.view_count = int(calibration["view_count"])
        self.slope = float(calibration["slope"])
        self.interval = float(calibration["interval"])
        self.x0 = float(calibration["x0"])

        grid = self._make_grid()
        # Large deterministic buffer; do not store it in checkpoints.
        self.register_buffer("sample_grid", grid, persistent=False)

    def _make_grid(self):
        H, W = self.native_h, self.native_w
        device = torch.device("cpu")
        yy = torch.arange(H, dtype=torch.float32, device=device)[:, None]
        xx = torch.arange(W, dtype=torch.float32, device=device)[None, :]

        # Fragment UVs are evaluated at pixel centers. The Unity shader then
        # multiplies by output size and adds 0.5, yielding 1-based x/y here.
        x_shader = xx + 1.0
        y_shader = yy + 1.0

        # Spatial normalized coordinates for grid_sample(align_corners=False).
        gx = 2.0 * ((xx + 0.5) / W) - 1.0
        gy = 2.0 * ((yy + 0.5) / H) - 1.0
        gx = gx.expand(H, W)
        gy = gy.expand(H, W)

        grids = []
        for bias in (0.0, 1.0, 2.0):
            x1 = (x_shader + y_shader * self.slope) * 3.0 + bias
            x_local = torch.remainder(x1 + self.x0, self.interval)
            choice_float = x_local / self.interval
            view = torch.floor(choice_float * self.view_count)
            view = view.clamp(0, self.view_count - 1)

            # Map 40 discrete views onto K continuous anchor centers.
            a = view / max(self.view_count - 1, 1) * (self.anchors - 1)
            gz = 2.0 * ((a + 0.5) / self.anchors) - 1.0
            grids.append(torch.stack([gx, gy, gz], dim=-1))

        # N=3 channels, D_out=1, H, W, xyz
        return torch.stack(grids, dim=0).unsqueeze(1)

    def forward(self, anchors_rgb):
        """
        anchors_rgb: B,K,3,960,540
        returns:     B,3,2560,1440
        """
        B, K, C, H, W = anchors_rgb.shape
        assert K == self.anchors and C == 3

        # Each RGB subpixel has a different angular choice. Treat RGB channels
        # as separate grid_sample batch items so each can have its own z grid.
        volume = anchors_rgb.permute(0, 2, 1, 3, 4).reshape(B * 3, 1, K, H, W)
        grid = self.sample_grid.to(device=anchors_rgb.device, dtype=anchors_rgb.dtype)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1, -1)
        grid = grid.reshape(B * 3, 1, self.native_h, self.native_w, 3)

        out = F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        out = out.reshape(B, 3, 1, self.native_h, self.native_w).squeeze(2)
        return out
