"""Shared image loading and resize/calibration utilities for dataset adapters."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except Exception:  # pragma: no cover - optional import
    cv2 = None
    HAS_CV2 = False

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - optional import
    PILImage = None


def load_rgb_image(path: str) -> np.ndarray:
    """Load an image as RGB using OpenCV when available, otherwise Pillow."""
    if HAS_CV2:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if PILImage is not None:
        return np.array(PILImage.open(path).convert("RGB"))
    raise RuntimeError("Could not read image; install opencv-python or pillow")


def resize_keep_aspect_and_update_K(
    rgb: np.ndarray,
    K: np.ndarray,
    target_hw: Tuple[int, int],
    mode: str = "crop",
    return_transform: bool = False,
):
    """Resize with aspect preservation and update camera intrinsics accordingly."""
    Ht, Wt = target_hw
    H0, W0 = rgb.shape[:2]
    s = max(Wt / float(W0), Ht / float(H0)) if mode == "crop" else min(Wt / float(W0), Ht / float(H0))
    Wr, Hr = int(round(W0 * s)), int(round(H0 * s))
    if HAS_CV2:
        rgb_r = cv2.resize(rgb, (Wr, Hr), interpolation=cv2.INTER_AREA)
    else:
        rgb_r = np.array(PILImage.fromarray(rgb).resize((Wr, Hr)))
    K2 = K.copy()
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] *= s
    K2[1, 2] *= s
    offset_x = 0.0
    offset_y = 0.0
    if mode == "crop":
        x0 = max(0, (Wr - Wt) // 2)
        y0 = max(0, (Hr - Ht) // 2)
        rgb_out = rgb_r[y0 : y0 + Ht, x0 : x0 + Wt]
        K2[0, 2] -= x0
        K2[1, 2] -= y0
        offset_x = -float(x0)
        offset_y = -float(y0)
    else:
        pad_x = max(0, (Wt - Wr) // 2)
        pad_y = max(0, (Ht - Hr) // 2)
        rgb_out = np.zeros((Ht, Wt, 3), dtype=rgb_r.dtype)
        rgb_out[pad_y : pad_y + Hr, pad_x : pad_x + Wr] = rgb_r
        K2[0, 2] += pad_x
        K2[1, 2] += pad_y
        offset_x = float(pad_x)
        offset_y = float(pad_y)
    if return_transform:
        return rgb_out, K2, float(s), offset_x, offset_y
    return rgb_out, K2
