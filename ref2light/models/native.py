from __future__ import annotations
import torch
import torch.nn.functional as F
from ref2light.models.swizzle import DisplayInterleaver
from ref2light.models.conditioned import ConditionedGenerator

PRECISION_CONTRACT = "v3_2r_fp32_head_geometry_swizzle_v1"

class FP32DisplayInterleaver(DisplayInterleaver):
    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Rebuild grids from calibration to avoid carrying half-precision rounding into FP32.
        device = self.sample_grid.device
        self.sample_grid = self._make_grid().to(device=device, dtype=torch.float32)
        return self

    def forward(self, anchors_rgb):
        b, k, c, h, w = anchors_rgb.shape
        if k != self.anchors or c != 3:
            raise ValueError(f"Invalid anchors: {tuple(anchors_rgb.shape)}")
        if self.sample_grid.dtype != torch.float32:
            raise RuntimeError("Swizzle calibration grid must stay FP32.")
        with torch.autocast(device_type=anchors_rgb.device.type, enabled=False):
            volume = anchors_rgb.float().permute(0, 2, 1, 3, 4).reshape(b * 3, 1, k, h, w)
            grid = self.sample_grid.to(device=anchors_rgb.device)
            grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1, -1)
            grid = grid.reshape(b * 3, 1, self.native_h, self.native_w, 3)
            out = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border",
                                align_corners=False)
            return out.reshape(b, 3, 1, self.native_h, self.native_w).squeeze(2)

class NativeGenerator(ConditionedGenerator):
    precision_contract = PRECISION_CONTRACT

    def __init__(self, calibration, anchors=12, flow_scale=0.35):
        super().__init__(calibration, anchors=anchors, flow_scale=flow_scale)
        self.swizzle = FP32DisplayInterleaver(calibration, anchors=anchors)
        self._fp32_grid_cache = {}

    def _apply(self, fn, recurse=True):
        self._fp32_grid_cache = {}
        return super()._apply(fn, recurse=recurse)

    @staticmethod
    def _base_grid(B, K, H, W, device, dtype=None):
        # Keep sampling grids in FP32.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            yy = torch.arange(H, device=device, dtype=torch.float32)
            xx = torch.arange(W, device=device, dtype=torch.float32)
            gy = 2.0 * ((yy + 0.5) / H) - 1.0
            gx = 2.0 * ((xx + 0.5) / W) - 1.0
            gy, gx = torch.meshgrid(gy, gx, indexing="ij")
            grid = torch.stack((gx, gy), dim=-1)
            return grid[None, None].expand(B, K, H, W, 2)

    def _cached_grid(self, B, H, W, device):
        key = (H, W, str(device))
        if key not in self._fp32_grid_cache:
            self._fp32_grid_cache = {key: self._base_grid(1, 1, H, W, device)}
        return self._fp32_grid_cache[key].expand(B, self.anchors, H, W, 2)

    def compose_raw_anchors(self, ref, raw):
        """raw has shape B x K x 6 x H x W; compose images in FP32 with gradients."""
        B, K, C, H, W = raw.shape
        if K != self.anchors or C != 6:
            raise ValueError(f"Bad raw shape {tuple(raw.shape)}")
        with torch.autocast(device_type=ref.device.type, enabled=False):
            raw = raw.float()
            flow = torch.tanh(raw[:, :, :2]) * self.flow_scale
            direct = torch.sigmoid(raw[:, :, 2:5])
            alpha = torch.sigmoid(raw[:, :, 5:6])
            base = self._cached_grid(B, H, W, ref.device)
            grid = (base + flow.permute(0, 1, 3, 4, 2)).clamp(-1.2, 1.2)
            ref_rep = ref.float()[:, None].expand(-1, K, -1, -1, -1)
            ref_rep = ref_rep.reshape(B * K, 3, ref.shape[-2], ref.shape[-1])
            warped = F.grid_sample(ref_rep, grid.reshape(B * K, H, W, 2),
                                   mode="bilinear", padding_mode="zeros",
                                   align_corners=False).view(B, K, 3, H, W)
            anchors = warped * (1.0 - alpha) + direct * alpha
            return anchors, {"flow": flow, "direct": direct, "alpha": alpha,
                             "warped": warped}

    def render_anchors_from_features(self, ref, feats, pose):
        p = self.pose(pose)
        s0, s1, s2, s3, s4, s5 = feats
        x = self.bfilm(self.bottleneck(s5), p)
        x = self.u4(x, s4, p)
        x = self.u3(x, s3, p)
        x = self.u2(x, s2, p)
        x = self.u1(x, s1, p)
        x = self.u0(x, s0, p)
        x = self._crop_480x270(x)
        if self.head.weight.dtype != torch.float32:
            raise RuntimeError("Keep model parameters FP32; use autocast, not model.half()/bfloat16().")
        with torch.autocast(device_type=x.device.type, enabled=False):
            raw = self.head(x.float())
            raw = F.pixel_shuffle(raw, 2)
            B, _, H, W = raw.shape
            raw = raw.view(B, self.anchors, 6, H, W)
            return self.compose_raw_anchors(ref, raw)
