"""Dataset helpers used by semantics crop extraction and OSNet training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset-specific conventions for semantic crop pipelines."""

    name: str
    crops_subdir: str = "person_crops"
    frame_token_widths: Tuple[int, ...] = ()
    image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")


_DATASETS = {
    "wildtrack": DatasetSpec(name="wildtrack", frame_token_widths=(8,)),
    "multiviewx": DatasetSpec(name="multiviewx", frame_token_widths=(4, 5)),
}


def dataset_choices() -> Tuple[str, ...]:
    """Return supported dataset names for semantic workflows."""
    return tuple(_DATASETS.keys())


def get_dataset_spec(name: str) -> DatasetSpec:
    """Resolve a semantic dataset spec by name."""
    if name not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return _DATASETS[name]


def resolve_crops_root(
    dataset: Optional[str],
    dataset_root: Optional[str],
    data_root: Optional[str],
) -> Path:
    """Resolve the person-crops directory from explicit or dataset-derived paths."""
    if data_root:
        return Path(data_root)
    if not dataset or not dataset_root:
        raise ValueError("Provide --data_root or both --dataset and --dataset_root.")
    spec = get_dataset_spec(dataset)
    return Path(dataset_root) / spec.crops_subdir


def frame_token_candidates(token: str, widths: Sequence[int]) -> Tuple[str, ...]:
    """Generate common frame-token variants to handle zero-padding inconsistencies."""
    candidates = []
    token = str(token)
    bases = [token]
    if token:
        stripped = token.lstrip("0")
        bases.append(stripped if stripped else "0")
    for base in bases:
        candidates.append(base)
        for width in widths:
            if len(base) < width:
                candidates.append(base.zfill(width))
    deduped = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)
