#!/usr/bin/env python3
"""
Fine-tune a YOLO detector on the WildTrack or MultiviewX dataset.

This script converts the ground-truth annotations into YOLO format,
fine-tunes a pretrained YOLO model, evaluates it, and saves the adapted weights
to a reusable checkpoint for downstream tasks.
"""

from __future__ import annotations

import torch
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import yaml

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - gives a clear error when deps missing
    raise SystemExit(
        "The 'ultralytics' package is required. Install it with `pip install ultralytics`."
    ) from exc

from PIL import Image


WILDTRACK_CAMERAS: Sequence[str] = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
MULTIVIEWX_CAMERAS: Sequence[str] = ("C1", "C2", "C3", "C4", "C5", "C6")
IMAGE_EXTS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_SAVE_NAMES: Dict[str, str] = {
    "wildtrack": "yolo_wildtrack_finetuned.pt",
    "multiviewx": "yolo_multiviewx_finetuned.pt",
}




def resize_with_center_crop(image: Image.Image, target_hw: Tuple[int, int]) -> Tuple[Image.Image, float, int, int]:
    """Resize with preserved aspect ratio, then center-crop to target (H, W)."""
    target_h, target_w = target_hw
    src_w, src_h = image.size
    scale = max(target_w / float(src_w), target_h / float(src_h))
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    resized = image.resize((resized_w, resized_h), resample=Image.BILINEAR)
    x0 = max(0, (resized_w - target_w) // 2)
    y0 = max(0, (resized_h - target_h) // 2)
    cropped = resized.crop((x0, y0, x0 + target_w, y0 + target_h))
    return cropped, scale, x0, y0


def transform_boxes_after_resize(
    boxes: Sequence[Tuple[float, float, float, float]],
    scale: float,
    x_offset: int,
    y_offset: int,
    target_w: int,
    target_h: int,
) -> List[Tuple[float, float, float, float]]:
    """Scale and crop axis-aligned boxes to match a resize_with_center_crop call."""
    adjusted: List[Tuple[float, float, float, float]] = []
    for xmin, ymin, xmax, ymax in boxes:
        xmin_s = xmin * scale - x_offset
        xmax_s = xmax * scale - x_offset
        ymin_s = ymin * scale - y_offset
        ymax_s = ymax * scale - y_offset

        xmin_c = max(0.0, min(float(target_w), xmin_s))
        xmax_c = max(0.0, min(float(target_w), xmax_s))
        ymin_c = max(0.0, min(float(target_h), ymin_s))
        ymax_c = max(0.0, min(float(target_h), ymax_s))

        if xmax_c <= xmin_c or ymax_c <= ymin_c:
            continue
        adjusted.append((xmin_c, ymin_c, xmax_c, ymax_c))
    return adjusted


def _token_variants(frame_token: str) -> List[str]:
    candidates = [frame_token]
    if frame_token.isdigit():
        stripped = frame_token.lstrip("0") or "0"
        # WildTrack and MultiviewX are inconsistent about zero-padding (e.g. 00000.json vs 0000.png),
        # so try a small set of common widths.
        candidates.extend(
            [
                stripped,
                frame_token.zfill(4),
                frame_token.zfill(5),
                frame_token.zfill(8),
                stripped.zfill(4),
                stripped.zfill(5),
                stripped.zfill(8),
            ]
        )
    # Keep stable order but de-dupe.
    seen = set()
    unique: List[str] = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        unique.append(cand)
    return unique


def discover_image(path_root: Path, camera: str, frame_token: str) -> Path | None:
    """Find the image file for a given camera and frame token."""
    camera_dir = path_root / "Image_subsets" / camera
    if not camera_dir.is_dir():
        camera_dir = path_root / "Image_subsets" / camera.upper()
        if not camera_dir.is_dir():
            return None
    for token in _token_variants(frame_token):
        for ext in IMAGE_EXTS:
            candidate = camera_dir / f"{token}{ext}"
            if candidate.is_file():
                return candidate
    return None


def boxes_for_camera(records: Iterable[dict], cam_idx: int) -> List[Tuple[float, float, float, float]]:
    """Extract (xmin, ymin, xmax, ymax) boxes for a given camera index."""
    boxes: List[Tuple[float, float, float, float]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        views = record.get("views", [])
        if not isinstance(views, list):
            continue
        for view in views:
            if not isinstance(view, dict):
                continue
            if view.get("viewNum") != cam_idx:
                continue
            xmin, xmax = float(view.get("xmin", -1)), float(view.get("xmax", -1))
            ymin, ymax = float(view.get("ymin", -1)), float(view.get("ymax", -1))
            if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
                continue
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append((xmin, ymin, xmax, ymax))
            break
    return boxes


def normalize_annotation_records(raw: object, ann_path: Path) -> List[dict]:
    """Normalize annotation payloads into a list of person records."""
    records_obj: object = raw
    if isinstance(records_obj, dict):
        for key in ("annotations", "records", "objects", "labels", "data", "persons"):
            value = records_obj.get(key)
            if isinstance(value, list):
                records_obj = value
                break
        else:
            if isinstance(records_obj.get("views"), list):
                # Single-record dict payloads are still valid.
                records_obj = [records_obj]
            else:
                raise ValueError(
                    f"Unsupported annotation dict in {ann_path}: expected one of "
                    "['annotations','records','objects','labels','data','persons'] or a 'views' list."
                )
    if not isinstance(records_obj, list):
        raise ValueError(f"Unsupported annotation payload in {ann_path}: expected list/dict, got {type(raw).__name__}")

    records: List[dict] = [record for record in records_obj if isinstance(record, dict)]
    if len(records) != len(records_obj):
        raise ValueError(f"Invalid annotation entries in {ann_path}: expected dict records only.")
    if not records:
        return []
    return records


def ensure_symlink_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink pointing to src; copy the file if the symlink fails."""
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo_labels(label_path: Path, boxes: Sequence[Tuple[float, float, float, float]], width: int, height: int) -> None:
    """Write YOLO-format labels (class_id x_center y_center width height) to disk."""
    lines: List[str] = []
    for xmin, ymin, xmax, ymax in boxes:
        w = max(xmax - xmin, 1.0)
        h = max(ymax - ymin, 1.0)
        x_center = xmin + w / 2.0
        y_center = ymin + h / 2.0
        x_center /= width
        y_center /= height
        w /= width
        h /= height
        # Clamp to [0, 1] to avoid training crashes due to rounding errors.
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        w = min(max(w, 1e-6), 1.0)
        h = min(max(h, 1e-6), 1.0)
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines))


def infer_camera_sizes(dataset_root: Path, cameras: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """Enumerate cameras and grab width/height for each once (assumes uniform sizing)."""
    sizes: Dict[str, Tuple[int, int]] = {}
    for camera in cameras:
        img_path = discover_image(dataset_root, camera, "00000000")
        if img_path is None:
            # Fall back to any file inside the camera directory.
            camera_dir = dataset_root / "Image_subsets" / camera
            if not camera_dir.exists():
                camera_dir = dataset_root / "Image_subsets" / camera.upper()
            if camera_dir.is_dir():
                for path in camera_dir.iterdir():
                    if path.suffix.lower() in IMAGE_EXTS:
                        img_path = path
                        break
        if img_path is None:
            raise FileNotFoundError(f"Unable to determine resolution for camera '{camera}'.")
        with Image.open(img_path) as im:
            sizes[camera] = im.size  # (width, height)
    return sizes


def _convert_multiview_dataset_to_yolo(
    dataset: str,
    dataset_root: Path,
    output_root: Path,
    cameras: Sequence[str],
    train_ratio: float,
    rebuild: bool,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Path:
    """
    Convert multiview dataset annotations into YOLO format and return the path to the data YAML.
    """
    output_root = output_root.resolve()
    labels_root = output_root / "labels"
    images_root = output_root / "images"
    data_yaml_path = output_root / f"{dataset}.yaml"

    if not rebuild and data_yaml_path.is_file():
        return data_yaml_path

    if rebuild and output_root.exists():
        shutil.rmtree(output_root)

    labels_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    ann_dir = dataset_root / "annotations_positions"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    all_tokens = sorted(p.stem for p in ann_dir.glob("*.json"))
    if not all_tokens:
        raise FileNotFoundError(f"No {dataset} annotations found in {ann_dir}")

    # Ordered split: use first train_ratio frames for training, remainder for testing.
    split_idx = max(1, int(len(all_tokens) * train_ratio))
    train_tokens = set(all_tokens[:split_idx])

    camera_sizes = infer_camera_sizes(dataset_root, cameras)

    total_images_written = 0
    for split in ("train", "test"):
        (labels_root / split).mkdir(parents=True, exist_ok=True)
        (images_root / split).mkdir(parents=True, exist_ok=True)

    for token in all_tokens:
        split = "train" if token in train_tokens else "test"
        ann_path = ann_dir / f"{token}.json"
        with ann_path.open("r") as f:
            records = normalize_annotation_records(json.load(f), ann_path)
        for cam_idx, camera in enumerate(cameras):
            image_path = discover_image(dataset_root, camera, token)
            if image_path is None:
                continue

            dest_name = f"{camera}_{token}{image_path.suffix.lower()}"
            dest_image = images_root / split / dest_name
            boxes = boxes_for_camera(records, cam_idx)

            if target_hw is not None:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    resized_img, scale, x_off, y_off = resize_with_center_crop(img, target_hw)
                    resized_img.save(dest_image)
                boxes = transform_boxes_after_resize(
                    boxes,
                    scale=scale,
                    x_offset=x_off,
                    y_offset=y_off,
                    target_w=target_hw[1],
                    target_h=target_hw[0],
                )
                label_w, label_h = target_hw[1], target_hw[0]
            else:
                ensure_symlink_or_copy(image_path, dest_image)
                label_w, label_h = camera_sizes[camera]

            label_path = labels_root / split / f"{camera}_{token}.txt"
            write_yolo_labels(label_path, boxes, label_w, label_h)
            total_images_written += 1

    if total_images_written == 0:
        example_ann = next(iter(all_tokens), "<none>")
        raise FileNotFoundError(
            "\n".join(
                [
                    f"No images were discovered for dataset '{dataset}' under {dataset_root / 'Image_subsets'}.",
                    f"Found {len(all_tokens)} annotation files under {ann_dir} (example token: {example_ann}).",
                    "This usually means the annotation frame tokens are padded differently than the image filenames.",
                    "Try passing the correct --dataset-root, or adjust the token padding logic in discover_image().",
                ]
            )
        )

    yaml_content = "\n".join(
        [
            f"path: {output_root}",
            "train: images/train",
            "val: images/test",
            "test: images/test",
            "names:",
            "  0: person",
            "nc: 1",
        ]
    )
    data_yaml_path.write_text(yaml_content)
    return data_yaml_path


def convert_wildtrack_to_yolo(
    wildtrack_root: Path,
    output_root: Path,
    train_ratio: float,
    rebuild: bool,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Path:
    """Convert WildTrack annotations/images into a YOLO training dataset layout."""
    return _convert_multiview_dataset_to_yolo(
        dataset="wildtrack",
        dataset_root=wildtrack_root,
        output_root=output_root,
        cameras=WILDTRACK_CAMERAS,
        train_ratio=train_ratio,
        rebuild=rebuild,
        target_hw=target_hw,
    )


def convert_multiviewx_to_yolo(
    multiviewx_root: Path,
    output_root: Path,
    train_ratio: float,
    rebuild: bool,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Path:
    """Convert MultiviewX annotations/images into a YOLO training dataset layout."""
    return _convert_multiview_dataset_to_yolo(
        dataset="multiviewx",
        dataset_root=multiviewx_root,
        output_root=output_root,
        cameras=MULTIVIEWX_CAMERAS,
        train_ratio=train_ratio,
        rebuild=rebuild,
        target_hw=target_hw,
    )


def finetune_yolo(
    model_name: str,
    data_config: Path,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    save_dir: Path,
    project: str,
    run_name: str,
) -> Path:
    """Load a pretrained YOLO model, fine-tune it, evaluate, and persist the best weights."""
    model = YOLO(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train(
        data=str(data_config),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=run_name,
        pretrained=True,
        workers=1,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics model did not expose a trainer with a save directory.")
    run_save_dir = Path(trainer.save_dir)
    best_weights = run_save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Best weights not found at {best_weights}")

    eval_split = "test"
    with data_config.open("r") as f:
        data_cfg = yaml.safe_load(f) or {}
    if not isinstance(data_cfg, dict) or not data_cfg.get("test"):
        eval_split = "val"
    model.val(data=str(data_config), imgsz=imgsz, batch=batch_size, device=device, split=eval_split, workers=1)

    final_ckpt = save_dir / DEFAULT_SAVE_NAMES.get(data_config.stem, DEFAULT_SAVE_NAMES["wildtrack"])
    shutil.copy2(best_weights, final_ckpt)
    return final_ckpt


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for YOLO dataset conversion and finetuning."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on WildTrack or MultiviewX.")
    finetune_root = Path(__file__).resolve().parent
    project_root = finetune_root.parent
    parser.add_argument(
        "--dataset",
        type=str,
        default="wildtrack",
        choices=("wildtrack", "multiviewx"),
        help="Dataset to use for fine-tuning.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to the dataset root directory (overrides dataset-specific root flags).",
    )
    parser.add_argument(
        "--wildtrack-root",
        type=Path,
        default=None,
        help="Path to the WildTrack dataset root directory.",
    )
    parser.add_argument(
        "--multiviewx-root",
        type=Path,
        default=None,
        help="Path to the MultiviewX dataset root directory.",
    )
    parser.add_argument(
        "--yolo-data-root",
        type=Path,
        default=None,
        help="Path to store the intermediate YOLO-format dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11x.pt",
        help="Name or path of the pretrained YOLO checkpoint to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and eval.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square image size used for training/eval (defaults to max(resize width/height)).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device to use (e.g. '0', 'cpu', 'auto').")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Fraction of frames used for training (ordered split).")
    parser.add_argument(
        "--rebuild-yolo-dataset",
        action="store_true",
        help="Force regeneration of the YOLO-format dataset even if it already exists.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("weights"),
        help="Directory to store the finetuned YOLO checkpoint.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Ultralytics project directory for run artifacts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name used by Ultralytics for logging.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Checkpoint filename to write under --save-dir (defaults to dataset-specific name).",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=360,
        help="Height to resize images before YOLO training (set <=0 to keep original).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="Width to resize images before YOLO training (set <=0 to keep original).",
    )
    return parser.parse_args()


def main() -> None:
    """Run YOLO conversion + finetuning workflow from command-line arguments."""
    args = parse_args()

    dataset = args.dataset.lower().strip()
    if args.dataset_root is not None:
        dataset_root = args.dataset_root.expanduser().resolve()
    elif dataset == "wildtrack":
        if args.wildtrack_root is None:
            raise SystemExit("--dataset-root or --wildtrack-root is required for wildtrack")
        dataset_root = args.wildtrack_root.expanduser().resolve()
    else:
        if args.multiviewx_root is None:
            raise SystemExit("--dataset-root or --multiviewx-root is required for multiviewx")
        dataset_root = args.multiviewx_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"{dataset} dataset not found at {dataset_root}")

    resize_hw = None
    if args.resize_height > 0 and args.resize_width > 0:
        resize_hw = (args.resize_height, args.resize_width)

    if args.yolo_data_root is None:
        args.yolo_data_root = dataset_root / "data" / f"{dataset}_yolo"
    if args.project is None:
        args.project = str(dataset_root / "runs" / "yolo")
    if args.run_name is None:
        args.run_name = f"{dataset}_finetune"

    if dataset == "wildtrack":
        data_config = convert_wildtrack_to_yolo(
            wildtrack_root=dataset_root,
            output_root=args.yolo_data_root,
            train_ratio=args.train_ratio,
            rebuild=args.rebuild_yolo_dataset,
            target_hw=resize_hw,
        )
    else:
        data_config = convert_multiviewx_to_yolo(
            multiviewx_root=dataset_root,
            output_root=args.yolo_data_root,
            train_ratio=args.train_ratio,
            rebuild=args.rebuild_yolo_dataset,
            target_hw=resize_hw,
        )

    checkpoint_path = finetune_yolo(
        model_name=args.model,
        data_config=data_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        save_dir=args.save_dir,
        project=args.project,
        run_name=args.run_name,
    )

    if args.save_name is not None:
        checkpoint_target = args.save_dir / args.save_name
        if checkpoint_target != checkpoint_path:
            shutil.copy2(checkpoint_path, checkpoint_target)
            checkpoint_path = checkpoint_target

    print(f"Finetuned weights saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
