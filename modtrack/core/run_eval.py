"""High-level ModTrack evaluation runner with minimal public interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from modtrack.core.paths import dataset_root, ensure_dir, ensure_file, weight_layout


def _resolve_eval_assets(dataset: str, mode: str) -> Tuple[Path, Path, Path, Path | None]:
    """Resolve required dataset/checkpoint paths for a given eval configuration."""
    d = dataset.lower().strip()
    m = mode.lower().strip()

    droot = dataset_root(d)
    ensure_dir(droot, f"dataset root for '{d}'")

    layout = weight_layout(d)
    if d != "radarscenes":
        ensure_file(layout["detector"], f"YOLO checkpoint for '{d}'")
        ensure_file(layout["lift"], f"Lift checkpoint for '{d}'")
        if m in {"semantic", "joint"}:
            ensure_file(layout["semantic"], f"semantic OSNet checkpoint for '{d}'")
            semantic = layout["semantic"]
        else:
            semantic = None
    else:
        semantic = None

    return droot, layout["detector"], layout["lift"], semantic


def run_eval(dataset: str, mode: str) -> Path:
    """Run ModTrack evaluation for a dataset/mode pair and return the results file path."""
    # Import lazily so CLI `--help` does not trigger evaluator dependency initialization.
    from modtrack.core.evaluate_modtrack import ModTrackEvaluator

    d = dataset.lower().strip()
    m = mode.lower().strip()
    if d not in {"wildtrack", "multiviewx", "radarscenes"}:
        raise ValueError(f"Unsupported dataset '{dataset}'")
    if m not in {"spatial", "semantic", "joint"}:
        raise ValueError(f"Unsupported mode '{mode}'")

    droot, yolo_ckpt, lift_ckpt, semantic_ckpt = _resolve_eval_assets(d, m)

    effective_mode = "spatial" if d == "radarscenes" else m
    if d == "radarscenes" and m != "spatial":
        print(f"[ModTrack] RadarScenes supports spatial evaluation. Requested mode '{m}' will run spatial internally.")

    evaluator_kwargs = {
        "dataset": d,
        "dataset_root": str(droot),
        "yolo_checkpoint": str(yolo_ckpt) if d != "radarscenes" else "",
        "lift_checkpoint": str(lift_ckpt) if d != "radarscenes" else None,
        "semantic_checkpoint": str(semantic_ckpt) if semantic_ckpt is not None else None,
        "matching_mode": effective_mode,
    }
    if d != "radarscenes":
        print("[ModTrack] Camera datasets use fixed eval window [360, 400).")

    evaluator = ModTrackEvaluator(**evaluator_kwargs)

    if d == "radarscenes":
        results = evaluator.evaluate_radarscenes_benchmark()
        out_dir = evaluator.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "radarscenes_lstq_results.json"
        out_file.write_text(json.dumps(results, indent=2))
        print(f"[ModTrack] RadarScenes results saved to {out_file}")
        return out_file

    results = evaluator.evaluate_on_sequence()
    evaluator.print_results(results)
    out_file = evaluator.save_results(results)
    if out_file is None:
        raise RuntimeError("Evaluation completed but results path was not returned by save_results().")
    print(f"[ModTrack] Evaluation results saved to {out_file}")
    return Path(out_file)
