import torch
import torch.nn as nn
import torch.nn.functional as F
from ref2light.models.swizzle import DisplayInterleaver

def groups_for(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(groups_for(c), c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(groups_for(c), c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        y = self.c1(F.silu(self.n1(x), inplace=True))
        y = self.c2(F.silu(self.n2(y), inplace=True))
        return x + y

class DownStage(nn.Module):
    def __init__(self, cin, cout, blocks=2):
        super().__init__()
        self.down = nn.Conv2d(cin, cout, 4, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(cout) for _ in range(blocks)])

    def forward(self, x):
        return self.blocks(self.down(x))

class PoseFiLM(nn.Module):
    def __init__(self, pose_dim, channels):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Linear(pose_dim, channels * 2),
        )

    def forward(self, x, p):
        gamma, beta = self.affine(p).chunk(2, dim=1)
        gamma = 0.25 * torch.tanh(gamma)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

class UpStage(nn.Module):
    def __init__(self, cin, skip_c, cout, pose_dim, blocks=2):
        super().__init__()
        self.proj = nn.Conv2d(cin + skip_c, cout, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(cout) for _ in range(blocks)])
        self.film = PoseFiLM(pose_dim, cout)

    def forward(self, x, skip, pose):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.blocks(x)
        return self.film(x, pose)

class SpatialReferenceEncoder(nn.Module):
    # Input: B x 3 x 512 x 288
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            ResBlock(64),
            ResBlock(64),
        )
        self.d1 = DownStage(64, 96)    # 256x144
        self.d2 = DownStage(96, 160)   # 128x72
        self.d3 = DownStage(160, 256)  # 64x36
        self.d4 = DownStage(256, 384)  # 32x18
        self.d5 = DownStage(384, 512)  # 16x9

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        s5 = self.d5(s4)
        return (s0, s1, s2, s3, s4, s5)

class ReferenceGenerator(nn.Module):

    def __init__(self, calibration, pose_in_dim=12, anchors=12, flow_scale=0.35):
        super().__init__()
        self.anchors = anchors
        self.flow_scale = flow_scale
        self.encoder = SpatialReferenceEncoder()

        self.pose = nn.Sequential(
            nn.Linear(pose_in_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(ResBlock(512), ResBlock(512), ResBlock(512))
        self.bfilm = PoseFiLM(256, 512)

        self.u4 = UpStage(512, 384, 384, 256)
        self.u3 = UpStage(384, 256, 256, 256)
        self.u2 = UpStage(256, 160, 192, 256)
        self.u1 = UpStage(192, 96, 128, 256)
        self.u0 = UpStage(128, 64, 96, 256)

        # Center-crop 512 x 288 to 480 x 270, then upsample by two with PixelShuffle.
        # Each anchor predicts six channels: displacement (2), colour (3), and blending weight (1).
        self.head = nn.Conv2d(96, anchors * 6 * 4, 1)

        # A negative bias initially favours reference sampling.
        with torch.no_grad():
            b = self.head.bias.view(anchors, 6, 4)
            b[:, 5, :].fill_(-2.0)

        self.swizzle = DisplayInterleaver(calibration, anchors=anchors)

    @staticmethod
    def _crop_480x270(x):
        H, W = x.shape[-2:]
        top = (H - 480) // 2
        left = (W - 270) // 2
        return x[..., top:top+480, left:left+270]

    @staticmethod
    def _base_grid(B, K, H, W, device, dtype):
        yy = torch.arange(H, device=device, dtype=dtype)
        xx = torch.arange(W, device=device, dtype=dtype)
        gy = 2.0 * ((yy + 0.5) / H) - 1.0
        gx = 2.0 * ((xx + 0.5) / W) - 1.0
        gy, gx = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1)
        return grid[None, None].expand(B, K, H, W, 2)

    def encode_reference(self, ref):
        return self.encoder(ref)

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
        raw = self.head(x)
        raw = F.pixel_shuffle(raw, 2)  # B, anchors*6, 960,540

        B, _, H, W = raw.shape
        raw = raw.view(B, self.anchors, 6, H, W)
        flow = torch.tanh(raw[:, :, 0:2]) * self.flow_scale
        direct = torch.sigmoid(raw[:, :, 2:5])
        alpha = torch.sigmoid(raw[:, :, 5:6])

        base = self._base_grid(B, self.anchors, H, W, raw.device, raw.dtype)
        grid = base + flow.permute(0,1,3,4,2)
        grid = grid.clamp(-1.2, 1.2)

        # Warp the canonical portrait reference for each angular anchor.
        ref_rep = ref[:, None].expand(-1, self.anchors, -1, -1, -1)
        ref_rep = ref_rep.reshape(B * self.anchors, 3, ref.shape[-2], ref.shape[-1])
        warped = F.grid_sample(
            ref_rep,
            grid.reshape(B * self.anchors, H, W, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(B, self.anchors, 3, H, W)

        anchors = warped * (1.0 - alpha) + direct * alpha
        aux = {"flow": flow, "direct": direct, "alpha": alpha, "warped": warped}
        return anchors, aux

    def render_native_from_features(self, ref, feats, pose, return_aux=False):
        anchors, aux = self.render_anchors_from_features(ref, feats, pose)
        native = self.swizzle(anchors)
        if return_aux:
            aux["anchors"] = anchors
            return native, aux
        return native

    def forward(self, ref, pose, return_aux=False):
        feats = self.encode_reference(ref)
        return self.render_native_from_features(ref, feats, pose, return_aux=return_aux)
