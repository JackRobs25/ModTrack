"""Minimal finetuning workflows for ModTrack."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from modtrack.core.paths import ensure_dir, ensure_file, weights_root, weight_layout


def _copy_to_default_weight(dataset: str, component: str, source: Path) -> Path:
    """Copy a produced checkpoint into the default ModTrack weights layout."""
    target = weight_layout(dataset)[component]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _resolve_person_crops_root(output_dir: Path) -> Path:
    """Resolve canonical person-crop directory under a workflow output directory."""
    # Accept either `<output-dir>` as the workspace root or direct person-crops directory.
    if output_dir.name == "person_crops":
        return output_dir
    nested = output_dir / "person_crops"
    if nested.is_dir():
        return nested
    return nested


def run_finetune_yolo(dataset: str, data_root: Path, output_dir: Path) -> Path:
    """Train/fine-tune YOLO for the selected dataset and register the resulting checkpoint."""
    # Import lazily to avoid ultralytics initialization on CLI help paths.
    from modtrack.finetune.finetune_yolo import (
        convert_multiviewx_to_yolo,
        convert_wildtrack_to_yolo,
        finetune_yolo,
    )

    dataset = dataset.lower().strip()
    if dataset not in {"wildtrack", "multiviewx"}:
        raise ValueError("YOLO finetuning supports only wildtrack and multiviewx")

    ensure_dir(data_root, f"dataset root for '{dataset}'")
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_data_root = output_dir / "yolo_data" / dataset
    run_project = output_dir / "runs" / "yolo"
    save_dir = output_dir / "checkpoints" / "yolo"

    if dataset == "wildtrack":
        data_config = convert_wildtrack_to_yolo(
            wildtrack_root=data_root,
            output_root=yolo_data_root,
            train_ratio=0.9,
            rebuild=True,
            target_hw=(360, 640),
        )
    else:
        data_config = convert_multiviewx_to_yolo(
            multiviewx_root=data_root,
            output_root=yolo_data_root,
            train_ratio=0.9,
            rebuild=True,
            target_hw=(360, 640),
        )

    ckpt = finetune_yolo(
        model_name="yolo11x.pt",
        data_config=data_config,
        epochs=50,
        batch_size=8,
        imgsz=640,
        device="cuda" if shutil.which("nvidia-smi") else "cpu",
        save_dir=save_dir,
        project=str(run_project),
        run_name=f"{dataset}_modtrack_finetune",
    )
    target = _copy_to_default_weight(dataset, "detector", Path(ckpt))
    print(f"[ModTrack] YOLO checkpoint copied to {target}")
    return target


def run_finetune_lift(dataset: str, data_root: Path, output_dir: Path) -> Path:
    """Run Lift finetuning and copy the best checkpoint into the default weights path."""
    dataset = dataset.lower().strip()
    if dataset not in {"wildtrack", "multiviewx"}:
        raise ValueError("Lift finetuning supports only wildtrack and multiviewx")

    ensure_dir(data_root, f"dataset root for '{dataset}'")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "checkpoints" / "lift"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Initialize from the LSS baseline checkpoint.
    lss_default_ckpt = weights_root() / "lss" / "model525000.pt"
    ensure_file(
        lss_default_ckpt,
        "Lift finetune base checkpoint (expected at weights/lss/model525000.pt by default)",
    )

    cmd = [
        sys.executable,
        "-m",
        "modtrack.finetune.finetune_lift",
        "--dataset",
        dataset,
        "--out_dir",
        str(ckpt_dir),
        "--camera",
        "all",
        "--imagenet_norm",
        "--epochs",
        "50",
    ]
    if dataset == "wildtrack":
        cmd.extend(["--wildtrack_root", str(data_root)])
    else:
        cmd.extend(["--multiviewx_root", str(data_root)])

    subprocess.run(cmd, check=True)

    best = ckpt_dir / f"best_{dataset}.pt"
    last = ckpt_dir / f"last_{dataset}.pt"
    source = best if best.is_file() else last
    if not source.is_file():
        raise FileNotFoundError(f"Lift finetune output not found in {ckpt_dir}")

    target = _copy_to_default_weight(dataset, "lift", source)
    print(f"[ModTrack] Lift checkpoint copied to {target}")
    return target


def run_semantics_prepare(dataset: str, data_root: Path, output_dir: Path) -> Path:
    """Extract person crops for downstream semantic ReID training."""
    from modtrack.finetune.semantics_prepare import extract_person_crops

    dataset = dataset.lower().strip()
    if dataset not in {"wildtrack", "multiviewx"}:
        raise ValueError("Semantics prepare supports only wildtrack and multiviewx")

    ensure_dir(data_root, f"dataset root for '{dataset}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_root = _resolve_person_crops_root(output_dir)
    crops_root.mkdir(parents=True, exist_ok=True)
    extract_person_crops(dataset, str(data_root), str(crops_root))
    print(f"[ModTrack] Person crops written to {crops_root}")
    return crops_root


def run_finetune_semantics(dataset: str, data_root: Path, output_dir: Path) -> Path:
    """Run OSNet semantic finetuning and register the resulting checkpoint."""
    dataset = dataset.lower().strip()
    if dataset not in {"wildtrack", "multiviewx"}:
        raise ValueError("Semantics finetuning supports only wildtrack and multiviewx")

    ensure_dir(data_root, f"dataset root for '{dataset}'")
    output_dir.mkdir(parents=True, exist_ok=True)

    crops_root = _resolve_person_crops_root(output_dir)
    if not crops_root.is_dir():
        raise FileNotFoundError(
            f"Missing person crops at {crops_root}. Run 'modtrack finetune semantics-prepare' first."
        )
    if not any(crops_root.rglob("*.jpg")):
        raise FileNotFoundError(
            f"No crop images found at {crops_root}. Run 'modtrack finetune semantics-prepare' first."
        )

    ckpt_dir = output_dir / "checkpoints" / "semantic"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "modtrack.finetune.train_osnet",
        "--dataset",
        dataset,
        "--dataset_root",
        str(data_root),
        "--data_root",
        str(crops_root),
        "--checkpoint_dir",
        str(ckpt_dir),
        "--pretrained",
        "--arch",
        "osnet_ain_x1_0",
        "--method",
        "triplet",
        "--num_epochs",
        "150",
        "--batch_size",
        "64",
        "--instances_per_pid",
        "4",
        "--steps_per_epoch",
        "350",
        "--learning_rate",
        "3e-4",
        "--infonce_temperature",
        "0.07",
        "--cross_camera_only",
        "--cross_camera_negatives",
        "--eval_cross_camera_only",
        "--eval_thresholds",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "--eval_pairs",
        "3000",
        "--num_workers",
        "1",
    ]
    subprocess.run(cmd, check=True)

    best = ckpt_dir / f"best_osnet_{dataset}.pt"
    last = ckpt_dir / f"osnet_last_{dataset}.pt"
    source = best if best.is_file() else last
    if not source.is_file():
        raise FileNotFoundError(f"Semantic finetune output not found in {ckpt_dir}")

    target = _copy_to_default_weight(dataset, "semantic", source)
    print(f"[ModTrack] Semantic checkpoint copied to {target}")
    return target
