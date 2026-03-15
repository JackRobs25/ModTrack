"""Dataset registry and adapter specification definitions for ModTrack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np


FrameIterator = Callable[..., Iterator[Tuple]]
CalibrationLoader = Callable[[str], Dict[str, Tuple[np.ndarray, np.ndarray]]]


@dataclass(frozen=True)
class DatasetSpec:
    """Static adapter configuration used by evaluation and training pipelines."""

    name: str
    cameras: List[str]
    load_calibration: CalibrationLoader
    iter_frames: FrameIterator
    native_hw: Tuple[int, int]
    ground_plane_x_range: Tuple[float, float]
    ground_plane_y_range: Tuple[float, float]


def list_datasets() -> List[str]:
    """List supported dataset identifiers."""
    return ["wildtrack", "multiviewx", "radarscenes"]


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the adapter spec for a dataset name."""
    key = name.lower().strip()
    if key == "wildtrack":
        from .wildtrack import WILDTRACK_CAMERAS, iter_wildtrack_frames, _load_calibration

        return DatasetSpec(
            name="wildtrack",
            cameras=WILDTRACK_CAMERAS,
            load_calibration=lambda root: _load_calibration(Path(root)),
            iter_frames=iter_wildtrack_frames,
            native_hw=(1080, 1920),
            ground_plane_x_range=(-3.0, 9.0),
            ground_plane_y_range=(-9.0, 27.0),
        )

    if key == "multiviewx":
        from .multiviewx import MULTIVIEWX_CAMERAS, MULTIVIEWX_NATIVE_HW, _load_calibration, iter_multiviewx_frames

        return DatasetSpec(
            name="multiviewx",
            cameras=MULTIVIEWX_CAMERAS,
            load_calibration=lambda root: _load_calibration(Path(root)),
            iter_frames=iter_multiviewx_frames,
            native_hw=MULTIVIEWX_NATIVE_HW,
            ground_plane_x_range=(0.0, 25.0),
            ground_plane_y_range=(0.0, 25.0),
        )

    if key == "radarscenes":
        from .radarscenes import RADARSCENES_SENSORS, RADARSCENES_NATIVE_HW, _load_calibration, iter_radarscenes_frames

        return DatasetSpec(
            name="radarscenes",
            cameras=RADARSCENES_SENSORS,
            load_calibration=lambda root: _load_calibration(Path(root)),
            iter_frames=iter_radarscenes_frames,
            native_hw=RADARSCENES_NATIVE_HW,
            ground_plane_x_range=(-100.0, 100.0),
            ground_plane_y_range=(-100.0, 100.0),
        )

    raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(list_datasets())}")
