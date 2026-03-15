"""Crop extraction utility for semantic ReID finetuning datasets."""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from tqdm import tqdm

from modtrack.finetune.semantics_dataset_registry import DatasetSpec, dataset_choices, frame_token_candidates, get_dataset_spec


def _load_annotation_records(ann_file: Path) -> list[dict[str, Any]]:
    """Load annotation records while supporting multiple JSON container formats."""
    with ann_file.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            return data["annotations"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
    if isinstance(data, list):
        return data
    return []


def _format_person_id(person_id: Any) -> str:
    """Format identity values into stable output directory names."""
    try:
        return f"{int(person_id):03d}"
    except (TypeError, ValueError):
        return str(person_id)


def _find_image_path(
    images_dir: Path, camera_num: int, frame_token: str, spec: DatasetSpec
) -> Optional[Path]:
    """Resolve an image file for a camera/frame token pair."""
    camera_dir = images_dir / f"C{camera_num + 1}"
    if not camera_dir.is_dir():
        camera_dir = images_dir / f"c{camera_num + 1}"
    if not camera_dir.is_dir():
        return None
    for token in frame_token_candidates(frame_token, spec.frame_token_widths):
        for ext in spec.image_exts:
            img_path = camera_dir / f"{token}{ext}"
            if img_path.is_file():
                return img_path
    return None


def _extract_crop(
    img_path: Path,
    output_file: Path,
    crop_box: tuple[float, float, float, float],
) -> bool:
    """Extract and save a crop; return False when processing fails."""
    try:
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            crop = img.crop(crop_box)
            crop.save(output_file, quality=95)
        return True
    except (OSError, ValueError) as exc:
        print(f"Error processing {img_path}: {exc}")
        return False


def extract_person_crops(dataset: str, dataset_root: str, output_root: str) -> None:
    """
    Extract person crops organized by person ID and camera.

    Output structure: output_root/personID/c{camera_num}/frame_{frame_num}.jpg
    """
    spec = get_dataset_spec(dataset)
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    annotations_dir = dataset_path / "annotations_positions"
    images_dir = dataset_path / "Image_subsets"

    annotation_files = sorted(annotations_dir.glob("*.json"))
    print(f"Found {len(annotation_files)} annotation files")

    for ann_file in tqdm(annotation_files, desc="Processing frames"):
        frame_token = ann_file.stem
        records = _load_annotation_records(ann_file)

        for record in records:
            if not isinstance(record, dict):
                continue
            person_id = record.get("personID", record.get("positionID"))
            if person_id is None:
                continue

            for view in record.get("views", []):
                camera_num = view.get("viewNum")
                if camera_num is None:
                    continue
                try:
                    camera_num = int(camera_num)
                except (TypeError, ValueError):
                    continue

                xmin, ymin = view.get("xmin", -1), view.get("ymin", -1)
                xmax, ymax = view.get("xmax", -1), view.get("ymax", -1)

                if xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1:
                    continue
                if xmax <= xmin or ymax <= ymin:
                    continue

                person_camera_dir = output_path / _format_person_id(person_id) / f"c{camera_num}"
                person_camera_dir.mkdir(parents=True, exist_ok=True)

                img_path = _find_image_path(images_dir, camera_num, frame_token, spec)
                if img_path is None:
                    print(f"Warning: Image not found for {frame_token} camera {camera_num}")
                    continue

                output_file = person_camera_dir / f"frame_{frame_token}.jpg"
                if not _extract_crop(img_path, output_file, (xmin, ymin, xmax, ymax)):
                    print(f"  Context: person={person_id} camera={camera_num} frame={frame_token}")
                    continue

    print(f"\nCrop extraction complete! Output saved to: {output_path}")

    person_dirs = list(output_path.glob("*"))
    print(f"Total unique persons: {len(person_dirs)}")

    for person_dir in sorted(person_dirs)[:5]:
        cameras = list(person_dir.glob("c*"))
        total_crops = sum(len(list(cam.glob("*.jpg"))) for cam in cameras)
        print(f"  Person {person_dir.name}: {len(cameras)} cameras, {total_crops} total crops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract person crops for SimCLR training")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=dataset_choices(),
        help="Dataset name (e.g., wildtrack, multiviewx)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="Root directory of the selected dataset",
    )
    parser.add_argument(
        "--wildtrack_root",
        type=str,
        default=None,
        help="Legacy WildTrack root (deprecated; use --dataset wildtrack instead)",
    )
    parser.add_argument("--output_root", type=str, required=True, help="Output directory for crops")
    args = parser.parse_args()

    if args.dataset:
        if not args.dataset_root:
            parser.error("--dataset_root is required when using --dataset")
        dataset = args.dataset
        dataset_root = args.dataset_root
    elif args.wildtrack_root:
        dataset = "wildtrack"
        dataset_root = args.wildtrack_root
    else:
        parser.error("Provide --dataset and --dataset_root (or --wildtrack_root for legacy usage)")

    extract_person_crops(dataset, dataset_root, args.output_root)
