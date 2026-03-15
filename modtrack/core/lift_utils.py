"""Minimal Lift utility functions used by ModTrack evaluation and finetuning."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modtrack.core.model import CamEncode


def make_bins(D: int, zmin: float, zmax: float, scheme: str, device: torch.device) -> torch.Tensor:
    """Create depth bin centers for the configured discretization scheme."""
    if scheme == "linear":
        return torch.linspace(zmin, zmax, D, device=device)
    if scheme == "log":
        return torch.logspace(math.log10(zmin), math.log10(zmax), D, device=device)
    if scheme == "inv":
        inv = torch.linspace(1.0 / zmax, 1.0 / zmin, D, device=device)
        return 1.0 / inv
    raise ValueError(f"Unsupported bin scheme: {scheme}")


def load_camencode_from_lss_ckpt(camencode: nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load CamEncode weights from either full LSS checkpoints or camencode-only checkpoints."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    if not isinstance(sd, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {ckpt_path!r}")

    if "camencode" in sd and isinstance(sd["camencode"], dict):
        ce_sd = sd["camencode"]
    else:
        ce_sd = {
            k[len("camencode.") :]: v
            for k, v in sd.items()
            if isinstance(k, str) and k.startswith("camencode.")
        }
        if not ce_sd:
            ce_sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    missing, unexpected = camencode.load_state_dict(ce_sd, strict=False)
    print(f"[ckpt] CamEncode loaded. missing={len(missing)} unexpected={len(unexpected)}")


def frame_to_tensor(frame_rgb: np.ndarray, device: torch.device, imagenet_norm: bool = True) -> torch.Tensor:
    """Convert an RGB uint8 frame to BCHW float tensor for CamEncode inference."""
    x = torch.from_numpy(frame_rgb).float() / 255.0
    if imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        x = (x - mean) / std
    return x.permute(2, 0, 1).unsqueeze(0).to(device)


def expected_depth_from_probs(depth_probs: torch.Tensor, depth_bins: torch.Tensor) -> torch.Tensor:
    """Compute expected depth map from per-bin depth probabilities."""
    return (depth_probs * depth_bins.view(1, -1, 1, 1).to(depth_probs.device)).sum(dim=1)


def gaussian_sample_points_with_weights(
    box: torch.Tensor,
    num_samples: int,
    sigma_scale: float,
    device: torch.device,
    center_u_frac: float = 0.5,
    center_v_frac: float = 0.5,
    min_sigma_px: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample UV points inside one box and return normalized Gaussian weights."""
    b = box.to(device=device, dtype=torch.float32)
    x1, y1, x2, y2 = b
    w = (x2 - x1).clamp(min=1.0)
    h = (y2 - y1).clamp(min=1.0)

    cx = x1 + w * float(center_u_frac)
    cy = y1 + h * float(center_v_frac)
    sx = torch.clamp(w * sigma_scale, min=min_sigma_px)
    sy = torch.clamp(h * sigma_scale, min=min_sigma_px)

    u = torch.normal(mean=cx, std=sx, size=(num_samples,), device=device)
    v = torch.normal(mean=cy, std=sy, size=(num_samples,), device=device)
    u = u.clamp(x1, x2 - 1e-6)
    v = v.clamp(y1, y2 - 1e-6)

    du = (u - cx) / sx
    dv = (v - cy) / sy
    w_unnorm = torch.exp(-0.5 * (du * du + dv * dv))
    w_norm = w_unnorm / w_unnorm.sum().clamp_min(1e-12)
    return torch.stack([u, v], dim=-1), w_norm


def gather_sample_points_from_boxes(
    boxes: torch.Tensor,
    num_samples: int,
    sigma_scale: float,
    image_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample weighted points for each valid box and concatenate results."""
    if boxes is None or boxes.numel() == 0 or num_samples <= 0:
        empty = torch.zeros((0,), device=device, dtype=torch.float32)
        return empty.view(0, 2), empty

    H, W = image_hw
    coords_list: List[torch.Tensor] = []
    weights_list: List[torch.Tensor] = []

    for box in boxes:
        b = box.to(device=device, dtype=torch.float32).clone()
        b[0::2] = b[0::2].clamp(0.0, float(W - 1))
        b[1::2] = b[1::2].clamp(0.0, float(H - 1))
        if (b[2] - b[0]) < 1.0 or (b[3] - b[1]) < 1.0:
            continue

        coords, w_norm = gaussian_sample_points_with_weights(b, num_samples, sigma_scale, device)
        if coords.numel():
            coords_list.append(coords)
            weights_list.append(w_norm)

    if not coords_list:
        empty = torch.zeros((0,), device=device, dtype=torch.float32)
        return empty.view(0, 2), empty

    return torch.cat(coords_list, dim=0), torch.cat(weights_list, dim=0)


def sample_expected_depth_at_uv(depth_map: torch.Tensor, coords: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinearly sample expected depth values at UV coordinates."""
    if coords is None or coords.numel() == 0:
        return depth_map.new_zeros((0,))
    H, W = image_hw
    grid = CamEncode.coords_to_grid(coords, (H, W))
    if grid.numel() == 0:
        return depth_map.new_zeros((0,))
    grid = grid.view(1, grid.shape[0], 1, 2)
    depth_in = depth_map.view(1, 1, H, W)
    sampled = F.grid_sample(depth_in, grid, mode="bilinear", align_corners=True)
    return sampled.view(-1)


def sample_depth_probs_at_uv(depth_probs: torch.Tensor, coords: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinearly sample per-bin depth probabilities at UV coordinates."""
    if coords is None or coords.numel() == 0:
        return depth_probs.new_zeros((0, depth_probs.shape[1]))
    if depth_probs.dim() != 4:
        raise ValueError("depth_probs must have shape [B, D, H, W]")
    B, D, _H, _W = depth_probs.shape
    if B != 1:
        raise ValueError("sample_depth_probs_at_uv assumes batch size of 1")

    grid = CamEncode.coords_to_grid(coords, image_hw)
    if grid.numel() == 0:
        return depth_probs.new_zeros((0, D))

    grid = grid.view(1, grid.shape[0], 1, 2)
    sampled = F.grid_sample(depth_probs, grid, mode="bilinear", align_corners=True)
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()
