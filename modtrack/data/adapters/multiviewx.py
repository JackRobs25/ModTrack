"""MultiviewX dataset adapter used by the ModTrack evaluation pipeline."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .common import load_rgb_image, resize_keep_aspect_and_update_K

MULTIVIEWX_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MULTIVIEWX_CAMERAS = ["C1", "C2", "C3", "C4", "C5", "C6"]
MULTIVIEWX_NATIVE_HW = (1080, 1920)
_GRID_WIDTH = 1000
_GRID_SPACING = 0.025  # 2.5 cm in metres
_GRID_ORIGIN_X = 0.0
_GRID_ORIGIN_Y = 0.0


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)
    k = rvec / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R.astype(np.float32)


def _parse_opencv_matrix(elem: ET.Element) -> np.ndarray:
    rows = int(elem.findtext("rows"))
    cols = int(elem.findtext("cols"))
    data = elem.findtext("data") or ""
    values = [float(x) for x in data.strip().split()]
    if len(values) != rows * cols:
        raise ValueError(f"Unexpected matrix size ({rows}x{cols}) vs data length {len(values)}")
    return np.asarray(values, dtype=np.float32).reshape(rows, cols)


def _load_intrinsics(intrinsic_dir: Path) -> Dict[str, np.ndarray]:
    intrinsics: Dict[str, np.ndarray] = {}
    for file in intrinsic_dir.glob("intr_*.xml"):
        if not file.is_file():
            continue
        tree = ET.parse(str(file))
        root = tree.getroot()
        cam_matrix_elem = root.find("camera_matrix")
        if cam_matrix_elem is None:
            continue
        K = _parse_opencv_matrix(cam_matrix_elem)
        key = file.stem.replace("intr_", "").upper()
        intrinsics[key] = K
    return intrinsics


def _load_extrinsics(extrinsic_dir: Path) -> Dict[str, np.ndarray]:
    extrinsics: Dict[str, np.ndarray] = {}
    for file in extrinsic_dir.glob("extr_*.xml"):
        if not file.is_file():
            continue
        tree = ET.parse(str(file))
        root = tree.getroot()
        rvec_elem = root.find("rvec")
        tvec_elem = root.find("tvec")
        if rvec_elem is None or tvec_elem is None:
            continue
        rvec = _parse_opencv_matrix(rvec_elem).reshape(3)
        tvec = _parse_opencv_matrix(tvec_elem).reshape(3)
        R_wc = _rodrigues(rvec.astype(np.float64))
        t_wc = tvec.astype(np.float32)
        T_cw = np.eye(4, dtype=np.float32)
        T_cw[:3, :3] = R_wc.T
        T_cw[:3, 3] = (-R_wc.T @ t_wc).astype(np.float32)
        key = file.stem.replace("extr_", "").upper()
        extrinsics[key] = T_cw
    return extrinsics


def _load_calibration(root: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    calib_dir = root / "calibrations"
    intr_dir = calib_dir / "intrinsic"
    ext_dir = calib_dir / "extrinsic"
    if not intr_dir.is_dir() or not ext_dir.is_dir():
        raise FileNotFoundError("MultiviewX calibration directories not found under 'calibrations'.")
    intrinsics = _load_intrinsics(intr_dir)
    extrinsics = _load_extrinsics(ext_dir)
    calibs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, cam_key in enumerate(MULTIVIEWX_CAMERAS, start=1):
        key = f"CAMERA{i}"
        K = intrinsics.get(key)
        T = extrinsics.get(key)
        if K is None or T is None:
            raise FileNotFoundError(f"Calibration missing for camera '{cam_key}'. Expected intr/extr '{key}'.")
        calibs[cam_key] = (K.astype(np.float32), T.astype(np.float32))
    return calibs


def _camera_directory(root: Path, camera: str) -> Path:
    cam_dir = root / "Image_subsets" / camera
    if not cam_dir.is_dir():
        cam_dir = root / "Image_subsets" / camera.upper()
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"MultiviewX camera directory not found for '{camera}'.")
    return cam_dir


def _position_id_to_world(position_id: int) -> np.ndarray:
    x = _GRID_ORIGIN_X + _GRID_SPACING * (position_id % _GRID_WIDTH)
    y = _GRID_ORIGIN_Y + _GRID_SPACING * (position_id // _GRID_WIDTH)
    return np.array([x, y, 0.0], dtype=np.float32)


def _annotation_token_for_image_stem(stem: str) -> str:
    if len(stem) >= 5:
        return stem
    return stem.zfill(5)


def _load_frame_annotations(ann_dir: Path, frame_token: str) -> List[dict]:
    ann_path = ann_dir / f"{frame_token}.json"
    if not ann_path.is_file() and frame_token.isdigit():
        ann_path = ann_dir / f"{frame_token.zfill(5)}.json"
    if not ann_path.is_file():
        return []
    with ann_path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            return data["annotations"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
    if isinstance(data, list):
        return data
    return []


def _world_points_for_camera(records: List[dict], cam_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: List[np.ndarray] = []
    boxes: List[np.ndarray] = []
    person_ids: List[int] = []
    for record in records:
        pos_id = record.get("positionID")
        if pos_id is None:
            continue
        identity = record.get("personID", pos_id)
        for view in record.get("views", []):
            if view.get("viewNum") != cam_index:
                continue
            xmin = view.get("xmin", -1)
            xmax = view.get("xmax", -1)
            ymin = view.get("ymin", -1)
            ymax = view.get("ymax", -1)
            if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
                continue
            points.append(_position_id_to_world(int(pos_id)))
            boxes.append(np.array([xmin, ymin, xmax, ymax], dtype=np.float32))
            person_ids.append(int(identity))
            break
    if not points:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.stack(points, axis=0),
        np.stack(boxes, axis=0),
        np.asarray(person_ids, dtype=np.int32),
    )


def iter_multiviewx_frames(
    multiviewx_root: str,
    camera: str,
    target_hw: Tuple[int, int],
    every_n: int = 1,
    max_frames: int = 0,
    return_token: bool = False,
    frame_start: int = 0,
    frame_end: int = 0,
) -> Iterator[Tuple]:
    """
    Iterate over MultiviewX frames for a given camera.

    Args:
        frame_start: Start yielding from this frame index (0-based, inclusive).
                     Frames before this are skipped without decoding.
        frame_end: Stop yielding at this frame index (exclusive, 0 = no limit).
    """
    root = Path(multiviewx_root)
    calibs = _load_calibration(root)
    cam_key = camera.upper()
    if cam_key not in calibs:
        available = ", ".join(sorted(calibs.keys()))
        raise ValueError(f"Camera '{camera}' not found. Available: {available}")
    K0, T_cam_world = calibs[cam_key]
    H, W = target_hw
    cam_dir = _camera_directory(root, cam_key)
    ann_dir = root / "annotations_positions"
    step = max(1, every_n)
    emitted = 0
    images = sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in MULTIVIEWX_IMAGE_EXTS)
    cam_index = MULTIVIEWX_CAMERAS.index(cam_key)

    for idx, img_path in enumerate(images):
        # Skip frames before frame_start WITHOUT decoding (performance optimization)
        if idx < frame_start:
            continue
        # Stop at frame_end
        if frame_end > 0 and idx >= frame_end:
            break
        if (idx % step) != 0:
            continue
        frame_token = _annotation_token_for_image_stem(img_path.stem)
        if max_frames and emitted >= max_frames:
            break

        rgb0 = load_rgb_image(str(img_path))
        rgb, K_img, scale, offset_x, offset_y = resize_keep_aspect_and_update_K(
            rgb0, K0, target_hw=(H, W), mode="crop", return_transform=True
        )
        records = _load_frame_annotations(ann_dir, frame_token)
        world_pts, boxes_px, person_ids = _world_points_for_camera(records, cam_index)

        if boxes_px.size:
            boxes_scaled = boxes_px.copy()
            boxes_scaled[:, [0, 2]] = boxes_scaled[:, [0, 2]] * scale + offset_x
            boxes_scaled[:, [1, 3]] = boxes_scaled[:, [1, 3]] * scale + offset_y
            boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0.0, float(W - 1))
            boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0.0, float(H - 1))
            valid_sizes = (boxes_scaled[:, 2] - boxes_scaled[:, 0] >= 1.0) & (
                boxes_scaled[:, 3] - boxes_scaled[:, 1] >= 1.0
            )
            if not np.all(valid_sizes):
                world_pts = world_pts[valid_sizes] if world_pts.size else world_pts
                person_ids = person_ids[valid_sizes] if person_ids.size else person_ids
                boxes_scaled = boxes_scaled[valid_sizes]
        else:
            boxes_scaled = boxes_px

        payload: Tuple = (
            img_path,
            rgb,
            (H, W),
            K_img,
            T_cam_world.copy(),
            np.eye(4, dtype=np.float32),
            world_pts if world_pts.size else None,
            boxes_scaled if boxes_scaled.size else None,
            person_ids if person_ids.size else None,
        )
        if return_token:
            payload = payload + (f"{cam_key}:{frame_token}",)
        yield payload
        emitted += 1
