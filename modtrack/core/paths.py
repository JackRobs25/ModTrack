"""Default path and asset resolution for ModTrack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the dataset root directory (environment-overridable)."""
    env = os.environ.get("MODTRACK_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else project_root() / "datasets"


def weights_root() -> Path:
    """Return the weights root directory (environment-overridable)."""
    env = os.environ.get("MODTRACK_WEIGHTS_ROOT")
    return Path(env).expanduser().resolve() if env else project_root() / "weights"


def dataset_root(dataset: str) -> Path:
    """Return the canonical dataset path under the configured data root."""
    return data_root() / dataset.lower().strip()


def weight_layout(dataset: str) -> Dict[str, Path]:
    """Return default checkpoint paths for a dataset."""
    d = dataset.lower().strip()
    base = weights_root() / d
    return {
        "base": base,
        "detector": base / "yolo.pt",
        "lift": base / "lift.pt",
        "semantic": base / "osnet.pt",
    }


def ensure_file(path: Path, desc: str) -> None:
    """Raise FileNotFoundError with guidance when a required file is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {desc}: {path}\n"
            f"Set MODTRACK_WEIGHTS_ROOT or place files in the default weights layout."
        )


def ensure_dir(path: Path, desc: str) -> None:
    """Raise FileNotFoundError with guidance when a required directory is missing."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing {desc}: {path}\n"
            f"Set MODTRACK_DATA_ROOT or place datasets in the default dataset layout."
        )
