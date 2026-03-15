"""Model components for LIFT/CamEncode."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class FullResDepthContextHead(nn.Module):
    """Refine concatenated depth/context logits to full image resolution."""

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        hidden = max(hidden_channels, in_channels)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.refine(x)


class CamEncode(nn.Module):
    """Exact NVLabs CamEncode with a lightweight CNN head for full-resolution outputs."""

    def __init__(self, D, C, downsample=16):
        super().__init__()
        self.D = D
        self.C = C
        self.downsample = downsample

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)
        self.full_res_head = FullResDepthContextHead(self.D + self.C)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def _encoder_forward(self, x):
        endpoints = {}
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f"reduction_{len(endpoints) + 1}"] = prev_x
            prev_x = x
        endpoints[f"reduction_{len(endpoints) + 1}"] = x
        return endpoints

    def get_eff_depth(self, x):
        endpoints = self._encoder_forward(x)
        x = self.up1(endpoints["reduction_5"], endpoints["reduction_4"])
        return x

    def forward_depth_head(self, x, full_res: bool = True):
        input_hw = x.shape[-2:]
        x = self.get_eff_depth(x)
        x = self.depthnet(x)
        depth_logits = x[:, :self.D]
        ctx_map = x[:, self.D : (self.D + self.C)]

        if full_res:
            combined = torch.cat([depth_logits, ctx_map], dim=1)
            refined = self.full_res_head(combined, input_hw)
            depth_logits = refined[:, :self.D]
            ctx_map = refined[:, self.D : (self.D + self.C)]

        return depth_logits, ctx_map

    def get_depth_feat(self, x):
        depth_logits, ctx_map = self.forward_depth_head(x, full_res=True)
        depth = self.get_depth_dist(depth_logits)
        lifted = depth.unsqueeze(1) * ctx_map.unsqueeze(2)  # [B, C, D, H, W]
        return depth, lifted

    @staticmethod
    def coords_to_grid(coords: torch.Tensor, image_hw):
        if coords.numel() == 0:
            return coords.new_zeros((0, 2))
        H, W = image_hw
        coords = coords.to(dtype=torch.float32)
        x = coords[:, 0]
        y = coords[:, 1]
        if W > 1:
            x_norm = 2.0 * (x / (W - 1.0)) - 1.0
        else:
            x_norm = torch.zeros_like(x)
        if H > 1:
            y_norm = 2.0 * (y / (H - 1.0)) - 1.0
        else:
            y_norm = torch.zeros_like(y)
        return torch.stack([x_norm, y_norm], dim=-1)

    @staticmethod
    def sample_depth_logits(depth_logits: torch.Tensor, coords: torch.Tensor, image_hw):
        if coords is None:
            return depth_logits.new_zeros((0, depth_logits.shape[0]))
        if depth_logits.dim() != 3:
            raise ValueError("depth_logits must have shape [D, H, W]")
        grid = CamEncode.coords_to_grid(coords, image_hw)
        if grid.numel() == 0:
            return depth_logits.new_zeros((0, depth_logits.shape[0]))
        grid = grid.view(1, 1, grid.shape[0], 2)
        sampled = F.grid_sample(
            depth_logits.unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=True,
        )
        sampled = sampled[0, :, 0, :].transpose(0, 1).contiguous()
        return sampled

    def forward(self, x):
        depth, lifted = self.get_depth_feat(x)
        return depth, lifted
