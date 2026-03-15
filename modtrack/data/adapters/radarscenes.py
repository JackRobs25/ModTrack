"""RadarScenes Dataset Adapter for ModTrack.

Converts radar point-cloud detections from 4 automotive radars into the
detection-dict format consumed by ModTrack's tracker:
    {cam_id, conf, cls, z_bev, R_bev, vec, yolo_class, bbox}

No YOLO, no Lift depth, no camera features -- the radar gives direct
BEV (x_seq, y_seq) positions. Covariance is derived analytically from
range/azimuth measurement uncertainty via Jacobian propagation.

Dataset: https://radar-scenes.com/
Paper:   RadarScenes (Schumann et al., 2021)
License: CC BY-NC-SA 4.0
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import h5py
import numpy as np

# ======================== MOS Classifier ========================

_MOS_CLF = None      # RF model (sklearn)
_MOS_NN = None       # NN model (PyTorch)
_MOS_NN_STATS = None  # normalization stats dict {mean, std, k_neighbors, threshold}


def load_mos_classifier(path: str):
    """Load a pretrained MOS classifier — RF (.pkl) or NN (.pt).

    The classifier replaces the hard Doppler threshold with a trained
    binary classifier. Analogous to YOLO for cameras — a pretrained
    perception module that plugs into the unchanged tracker.
    """
    if path.endswith(".pt"):
        _load_nn_model(path)
    else:
        _load_rf_model(path)


def _load_rf_model(path: str):
    """Load sklearn Random Forest model."""
    global _MOS_CLF
    import joblib
    _MOS_CLF = joblib.load(path)
    print(f"[MOS] Loaded RF classifier from {path}")


def _load_nn_model(path: str):
    """Load PyTorch NN model with normalization stats.

    Supports two checkpoint formats:
    - Legacy binary: nn.Sequential MOS-only model
    - Multi-task: RadarMultiTaskMLP with MOS + class heads (has "multitask": True)
    """
    global _MOS_NN, _MOS_NN_STATS
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arch = checkpoint.get("architecture", {"hidden1": 64, "hidden2": 32, "dropout": 0.3})
    input_dim = checkpoint["input_dim"]
    is_multitask = checkpoint.get("multitask", False)

    if is_multitask:
        try:
            from modtrack.finetune.train_radar_mos_nn import get_multitask_model_class
        except ImportError as exc:
            raise RuntimeError(
                "RadarScenes multi-task MOS checkpoints require modtrack.finetune.train_radar_mos_nn."
            ) from exc
        ModelClass = get_multitask_model_class()
        model = ModelClass(
            input_dim=input_dim,
            hidden1=arch["hidden1"],
            hidden2=arch["hidden2"],
            num_classes=checkpoint.get("num_classes", 5),
            dropout=arch["dropout"],
        )
    else:
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(input_dim, arch["hidden1"]),
            nn.BatchNorm1d(arch["hidden1"]),
            nn.ReLU(),
            nn.Dropout(arch["dropout"]),
            nn.Linear(arch["hidden1"], arch["hidden2"]),
            nn.BatchNorm1d(arch["hidden2"]),
            nn.ReLU(),
            nn.Dropout(arch["dropout"]),
            nn.Linear(arch["hidden2"], 1),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _MOS_NN = model
    _MOS_NN_STATS = {
        "mean": np.array(checkpoint["mean"], dtype=np.float32),
        "std": np.array(checkpoint["std"], dtype=np.float32),
        "k_neighbors": checkpoint.get("k_neighbors", 16),
        "threshold": checkpoint.get("best_threshold", 0.5),
        "multitask": is_multitask,
        "num_classes": checkpoint.get("num_classes", 5),
        "grouped_class_names": checkpoint.get("grouped_class_names"),
    }
    mode_str = f"multi-task ({_MOS_NN_STATS['num_classes']}-class)" if is_multitask else "binary MOS"
    print(f"[MOS] Loaded NN classifier from {path} "
          f"({mode_str}, k={_MOS_NN_STATS['k_neighbors']}, threshold={_MOS_NN_STATS['threshold']})")


_MOS_THRESHOLD = 0.5  # probability threshold for predict_proba


def set_mos_threshold(threshold: float):
    """Set the probability threshold for the MOS classifier."""
    global _MOS_THRESHOLD
    _MOS_THRESHOLD = threshold
    if _MOS_NN_STATS is not None:
        _MOS_NN_STATS["threshold"] = threshold
    print(f"[MOS] Probability threshold set to {threshold}")


def predict_moving(
    vr_compensated: np.ndarray,
    rcs: np.ndarray,
    range_sc: np.ndarray,
    azimuth_sc: np.ndarray,
    min_doppler: float = 0.5,
    x_seq: np.ndarray = None,
    y_seq: np.ndarray = None,
    timestamps: np.ndarray = None,
    return_probs: bool = False,
) -> np.ndarray:
    """Per-point moving/static prediction.

    If a trained NN classifier is loaded and spatial data (x_seq, y_seq,
    timestamps) is provided, uses k-NN neighborhood features for context-aware
    prediction.

    If a trained RF classifier is loaded, uses predict_proba with a tunable
    threshold.

    Otherwise falls back to the simple Doppler threshold (|vr| >= min_doppler).

    Returns:
        If return_probs=False: boolean mask, True = moving.
        If return_probs=True:
          - Binary NN/RF/Doppler: (mask, proba)
          - Multi-task NN: (mask, proba, class_ids, class_confs)
    """
    # Path 1: Neural network with spatial context
    if _MOS_NN is not None and x_seq is not None and y_seq is not None:
        result = _predict_moving_nn(vr_compensated, rcs, range_sc, azimuth_sc,
                                    x_seq, y_seq, timestamps)
        if return_probs:
            return result  # 2-tuple (binary) or 4-tuple (multi-task)
        return result[0]  # mask only
    # Path 2: Random Forest (per-point features only)
    if _MOS_CLF is not None:
        X = np.column_stack([
            np.abs(vr_compensated),
            vr_compensated,
            rcs,
            range_sc,
            azimuth_sc,
        ]).astype(np.float32)
        proba = _MOS_CLF.predict_proba(X)[:, 1]
        mask = proba >= _MOS_THRESHOLD
        return (mask, proba) if return_probs else mask
    # Path 3: Doppler threshold (default)
    speeds = np.abs(vr_compensated)
    mask = speeds >= min_doppler
    if return_probs:
        # Soft confidence: ramp from 0 at 0 m/s to 1.0 at 2.0 m/s
        # Points below min_doppler still get low proba (used for S_cls)
        proba = np.clip(speeds / 2.0, 0.0, 1.0).astype(np.float32)
        return (mask, proba)
    return mask


def _predict_moving_nn(
    vr_compensated: np.ndarray,
    rcs: np.ndarray,
    range_sc: np.ndarray,
    azimuth_sc: np.ndarray,
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    timestamps: np.ndarray,
) -> tuple:
    """NN prediction with k-NN neighborhood features.

    Returns:
        Binary model: (mask, proba)
        Multi-task model: (mask, proba, class_ids, class_confs)
            class_ids: [N] int array of grouped class IDs (0-4)
            class_confs: [N] float array of class confidence (softmax max)
    """
    import torch
    try:
        from modtrack.finetune.train_radar_mos_nn import compute_knn_features
    except ImportError as exc:
        raise RuntimeError(
            "RadarScenes NN MOS inference requires modtrack.finetune.train_radar_mos_nn."
        ) from exc

    stats = _MOS_NN_STATS
    k = stats["k_neighbors"]
    is_multitask = stats.get("multitask", False)

    # Per-point features
    X_perpoint = np.column_stack([
        np.abs(vr_compensated),
        vr_compensated,
        rcs,
        range_sc,
        azimuth_sc,
    ]).astype(np.float32)

    # k-NN features
    ts = timestamps if timestamps is not None else np.zeros(len(vr_compensated))
    X_knn = compute_knn_features(x_seq, y_seq, vr_compensated, rcs, range_sc, ts, k=k)

    X = np.concatenate([X_perpoint, X_knn], axis=1)

    # Normalize
    X = (X - stats["mean"]) / stats["std"]

    # Inference
    X_t = torch.from_numpy(X).float()
    chunk_size = 100_000

    with torch.no_grad():
        if is_multitask:
            proba_parts, cls_id_parts, cls_conf_parts = [], [], []
            for start in range(0, len(X_t), chunk_size):
                end = min(start + chunk_size, len(X_t))
                mos_logits, class_logits = _MOS_NN(X_t[start:end])
                proba_parts.append(torch.sigmoid(mos_logits).numpy())
                cls_probs = torch.softmax(class_logits, dim=-1)
                cls_id_parts.append(cls_probs.argmax(dim=-1).numpy())
                cls_conf_parts.append(cls_probs.max(dim=-1).values.numpy())
            proba = np.concatenate(proba_parts)
            class_ids = np.concatenate(cls_id_parts)
            class_confs = np.concatenate(cls_conf_parts)
            mask = proba >= stats["threshold"]
            return mask, proba, class_ids, class_confs
        else:
            proba_parts = []
            for start in range(0, len(X_t), chunk_size):
                end = min(start + chunk_size, len(X_t))
                logits = _MOS_NN(X_t[start:end]).squeeze(-1)
                proba_parts.append(torch.sigmoid(logits).numpy())
            proba = np.concatenate(proba_parts)
            mask = proba >= stats["threshold"]
            return mask, proba

# ======================== Constants ========================

RADARSCENES_SENSORS = ["sensor_1", "sensor_2", "sensor_3", "sensor_4"]

RADARSCENES_SENSOR_NAMES = {
    1: "side_left",
    2: "front_left",
    3: "front_right",
    4: "side_right",
}

# No images — native_hw is unused; set to dummy (0, 0)
RADARSCENES_NATIVE_HW = (0, 0)

# 77 GHz automotive radar measurement uncertainty
RADAR_SIGMA_RANGE_M = 0.10        # Range std (meters)
RADAR_SIGMA_AZIMUTH_RAD = 0.031   # Azimuth std (~1.8 degrees)
RADAR_EGO_NOISE_M2 = 0.10         # Odometry drift covariance (m^2)

RADAR_FEATURE_DIM = 8

# ======================== Class Mapping ========================
# RadarScenes label_id → COCO/YOLO class ID
# label_id: 0=Car, 1=LargeVehicle, 2=Truck, 3=Bus, 4=Train,
#           5=Bicycle, 6=MotorizedTwoWheeler, 7=Pedestrian,
#           8=PedestrianGroup, 9=Animal, 10=OtherDynamic, 11=Static

RADARSCENES_LABEL_TO_YOLO_CLASS = {
    0: 2,    # Car → COCO car
    1: 7,    # Large Vehicle → COCO truck
    2: 7,    # Truck → COCO truck
    3: 5,    # Bus → COCO bus
    4: 5,    # Train → COCO bus (closest dynamics)
    5: 1,    # Bicycle → COCO bicycle
    6: 3,    # Motorized Two-Wheeler → COCO motorcycle
    7: 0,    # Pedestrian → COCO person
    8: 0,    # Pedestrian Group → COCO person
    9: 0,    # Animal → COCO person (similar dynamics)
    10: 2,   # Other Dynamic → COCO car (default)
    11: -1,  # Static → skip
}

RADARSCENES_LABEL_NAMES = {
    0: "car", 1: "large_vehicle", 2: "truck", 3: "bus", 4: "train",
    5: "bicycle", 6: "motorcycle", 7: "pedestrian", 8: "pedestrian_group",
    9: "animal", 10: "other_dynamic", 11: "static",
}

MOVING_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

# ======================== GT label_id → Motion Profile (Oracle) ========================
# Maps raw RadarScenes label_id to tracker profile name for oracle experiment.
LABEL_ID_TO_PROFILE_NAME = {
    0: "radar_vehicle",    # Car
    1: "truck",            # Large Vehicle
    2: "truck",            # Truck
    3: "truck",            # Bus (truck dynamics)
    4: "truck",            # Train (truck dynamics)
    5: "motorcycle",       # Bicycle
    6: "motorcycle",       # Motorized Two-Wheeler
    7: "pedestrian",       # Pedestrian
    8: "pedestrian",       # Pedestrian Group
    9: "pedestrian",       # Animal (pedestrian dynamics)
    10: "radar_vehicle",   # Other Dynamic
    11: "static",          # Static (filtered by oracle MOS)
}

# ======================== NN Grouped Class → Motion Profile ========================
# Maps multi-task NN grouped class ID → tracker profile class_name.
# Used when multi-task NN is loaded to assign per-detection motion profiles,
# mirroring how YOLO class → profile works in the camera pipeline.
GROUPED_CLASS_TO_PROFILE_NAME = {
    0: "radar_vehicle",    # car → RADAR_VEHICLE_PROFILE
    1: "truck",            # large_vehicle → TRUCK_PROFILE
    2: "motorcycle",       # two_wheeler → MOTORCYCLE_PROFILE
    3: "pedestrian",       # pedestrian → PEDESTRIAN_PROFILE
    4: "static",           # static → not tracked (filtered by MOS)
}
GROUPED_CLASS_NAMES = ["car", "large_vehicle", "two_wheeler", "pedestrian", "static"]
NUM_CLASSES = 5


# ======================== Calibration ========================

def _find_sensors_json(root: Path) -> Path:
    """Locate sensors.json, handling nested extraction layouts."""
    for candidate in [
        root / "sensors.json",
        root / "data" / "sensors.json",
        root / "RadarScenes" / "data" / "sensors.json",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find sensors.json under {root}")


def _find_data_dir(root: Path) -> Path:
    """Locate the directory containing sequence_* folders."""
    for candidate in [
        root / "data",
        root / "RadarScenes" / "data",
        root,
    ]:
        if candidate.is_dir() and any(candidate.glob("sequence_*")):
            return candidate
    raise FileNotFoundError(f"Cannot find sequence directories under {root}")


def _load_sensor_calibration(root: Path) -> Dict[str, Dict]:
    """Load sensor mounting positions from sensors.json.

    Returns: Dict mapping sensor_id str (e.g. "1") to {"x", "y", "yaw", "id"}.
    Keys in the file are "radar_1" etc. — we normalize to numeric string keys.
    """
    sensors_path = _find_sensors_json(root)
    with open(sensors_path) as f:
        raw = json.load(f)
    # Normalize: "radar_1" -> "1"
    result = {}
    for key, cal in raw.items():
        sid = str(cal.get("id", key.replace("radar_", "")))
        result[sid] = cal
    return result


def _load_calibration(root: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load radar sensor calibrations for registry compatibility.

    Returns: {sensor_name: (K_dummy_3x3, T_ego_sensor_4x4)}
    K is identity (no image intrinsics).
    T is the sensor→ego transform from mounting position.
    """
    sensors = _load_sensor_calibration(root)
    result = {}
    for sid_str, cal in sensors.items():
        sensor_name = f"sensor_{sid_str}"
        K_dummy = np.eye(3, dtype=np.float32)
        yaw = float(cal["yaw"])
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4, dtype=np.float32)
        T[0, 0] = c; T[0, 1] = -s
        T[1, 0] = s; T[1, 1] = c
        T[0, 3] = float(cal["x"])
        T[1, 3] = float(cal["y"])
        result[sensor_name] = (K_dummy, T)
    return result


# ======================== BEV Covariance ========================

def _radar_bev_covariance(
    range_m: float,
    azimuth_rad: float,
    yaw_global: float,
    sigma_r: float = RADAR_SIGMA_RANGE_M,
    sigma_az: float = RADAR_SIGMA_AZIMUTH_RAD,
    ego_noise: float = RADAR_EGO_NOISE_M2,
    min_eigenvalue: float = 0.21,
) -> np.ndarray:
    """Compute 2x2 BEV covariance from radar polar measurement uncertainty.

    Jacobian of polar→Cartesian:
        x = r sin(θ),  y = r cos(θ)
        J = [[sin(θ), r cos(θ)],
             [cos(θ), -r sin(θ)]]
    R_cart = J diag(σ_r², σ_θ²) Jᵀ
    Then rotate to global frame and add ego-motion noise floor.
    """
    sa, ca = np.sin(azimuth_rad), np.cos(azimuth_rad)
    J = np.array([[sa, range_m * ca],
                  [ca, -range_m * sa]], dtype=np.float64)
    R_polar = np.diag([sigma_r ** 2, sigma_az ** 2])
    R_sensor = J @ R_polar @ J.T

    # Rotate to global frame
    c, s = np.cos(yaw_global), np.sin(yaw_global)
    R_rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    R_global = R_rot @ R_sensor @ R_rot.T

    # Add ego-motion uncertainty
    R_pose = np.eye(2, dtype=np.float64) * ego_noise
    R_global += R_pose

    # Floor eigenvalue to prevent Mahalanobis explosion
    eigvals, eigvecs = np.linalg.eigh(R_global)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    R_global = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return R_global.astype(np.float32), R_pose.astype(np.float32)


# ======================== Radar Feature Vector ========================

def _build_radar_feature(
    rcs: float,
    vr_compensated: float,
    range_sc: float,
    azimuth_sc: float,
) -> np.ndarray:
    """Build 8-dim pseudo-feature vector from radar metadata.

    L2-normalized so cosine similarity can be used for semantic matching.
    No GT labels used — only measured radar quantities.
    """
    feat = np.array([
        rcs / 30.0,                                          # RCS normalized
        vr_compensated / 30.0,                                # Radial velocity
        abs(vr_compensated) / 15.0,                           # Speed magnitude
        range_sc / 100.0,                                     # Range normalized
        azimuth_sc / np.pi,                                   # Azimuth normalized
        np.clip(rcs / 20.0, -1.0, 1.0),                     # RCS strength proxy
        1.0 if abs(vr_compensated) > 1.0 else 0.0,           # Dynamic indicator (Doppler)
        0.5,                                                   # Padding
    ], dtype=np.float32)
    norm = np.linalg.norm(feat)
    if norm > 1e-8:
        feat /= norm
    return feat


# ======================== Per-Sensor DBSCAN Pre-Clustering ========================

def _dbscan_precluster(
    points_xy: np.ndarray,
    eps: float = 2.0,
    min_samples: int = 2,
) -> np.ndarray:
    """DBSCAN on 2D points. Returns cluster labels (-1 = noise).

    Uses sklearn's ball-tree DBSCAN for O(n log n) efficiency.
    Noise points (label=-1) are NOT assigned to clusters — they are
    discarded by the caller. This prevents isolated clutter from
    inflating detection count.
    """
    from sklearn.cluster import DBSCAN as SkDBSCAN
    n = len(points_xy)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        # Single point can't form a cluster with min_samples >= 2
        return np.array([-1], dtype=int)

    labels = SkDBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree").fit_predict(points_xy)

    # Noise points stay as -1 (caller skips them)
    return labels


# ======================== Doppler/RCS Heuristic Classification ========================

def _classify_radar_point(speed: float, rcs: float) -> str:
    """Classify a single radar point by Doppler speed and RCS into a motion profile.

    Thresholds derived from RadarScenes class statistics:
      - truck/bus:   high speed + high RCS (large metal cross-section)
      - car:         moderate-to-high speed + moderate RCS
      - motorcycle:  moderate speed + low RCS (small cross-section)
      - pedestrian:  low speed + low RCS
    """
    if speed > 3.0:
        if rcs > 12.0:
            return "truck"          # large vehicle: fast + strong reflector
        elif rcs > 0.0:
            return "radar_vehicle"  # car: fast + moderate reflector
        else:
            return "motorcycle"     # two-wheeler: fast + weak reflector
    elif speed > 0.5:
        if rcs > 5.0:
            return "radar_vehicle"  # slow-moving car (parking, turning)
        elif rcs > -5.0:
            return "motorcycle"     # slow two-wheeler / cyclist
        else:
            return "pedestrian"     # slow + very weak reflector
    else:
        return "pedestrian"         # near-stationary → pedestrian dynamics


def _classify_radar_cluster(speed: float, rcs: float, n_pts: int) -> str:
    """Classify a DBSCAN cluster by mean Doppler speed, mean RCS, and point count.

    Cluster-level features provide more robust classification than single points:
      - n_pts correlates with physical size (trucks > cars > motorcycles > pedestrians)
    """
    if speed > 3.0:
        if rcs > 12.0 or n_pts >= 8:
            return "truck"          # large vehicle: fast + (strong reflector or many points)
        elif rcs > 0.0:
            return "radar_vehicle"  # car
        else:
            return "motorcycle"     # two-wheeler
    elif speed > 0.5:
        if n_pts >= 5 and rcs > 5.0:
            return "radar_vehicle"  # slow car
        elif rcs > -5.0:
            return "motorcycle"     # slow two-wheeler
        else:
            return "pedestrian"
    else:
        return "pedestrian"


# ======================== HDF5 → Detection Dicts ========================

def _hdf5_to_frames(
    h5_path: str,
    sensor_calibs: Dict,
    dt_frame: float = 0.05,
    min_doppler: float = 0.5,
    min_rcs: float = -10.0,
    min_eigenvalue: float = 0.21,
    active_sensors: Optional[Set[int]] = None,
    dbscan_eps: float = 2.0,
    use_classifier: bool = False,
    skip_dbscan: bool = False,
    oracle: bool = False,
) -> Tuple[Dict[int, List[Dict]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read radar HDF5, filter/cluster, return {frame_idx: [detection_dict, ...]}.

    Two modes:
    - DBSCAN mode (default): pre-cluster per (frame, sensor) group, each cluster
      becomes one detection. Analogous to YOLO producing one bbox per object.
    - Point mode (skip_dbscan=True): each classified-moving point becomes its own
      detection. Graph clustering downstream handles grouping via Mahalanobis distance,
      making the pipeline structurally identical to the camera pipeline.

    When use_classifier=True, uses the loaded MOS classifier instead of Doppler
    threshold for moving/static filtering.

    Each detection dict includes 'raw_point_indices' — indices into the HDF5 arrays
    for LSTQ point-level evaluation.

    Also returns odometry arrays for ego-pose interpolation.
    """
    raw = load_sequence_raw(h5_path)
    timestamps = raw["timestamps"]
    x_seq = raw["x_seq"]
    y_seq = raw["y_seq"]
    range_sc = raw["range_sc"]
    azimuth_sc = raw["azimuth_sc"]
    rcs = raw["rcs"]
    vr_compensated = raw["vr_compensated"]
    sensor_id = raw["sensor_id"]
    label_id = raw["label_id"]
    track_id = raw["track_id"]
    odo_ts = raw["odo_ts"]
    odo_x = raw["odo_x"]
    odo_y = raw["odo_y"]
    odo_yaw = raw["odo_yaw"]

    # Quantize timestamps into frames
    t0 = timestamps.min()
    dt_us = dt_frame * 1e6
    frame_indices = ((timestamps - t0) / dt_us).astype(int)

    # Interpolate ego yaw for each radar measurement
    yaw_interp = np.interp(timestamps.astype(np.float64),
                           odo_ts.astype(np.float64),
                           odo_yaw.astype(np.float64))

    # Build mask for valid (dynamic) points
    valid = np.ones(len(timestamps), dtype=bool)
    valid &= (rcs >= min_rcs)
    if active_sensors is not None:
        sensor_mask = np.zeros(len(timestamps), dtype=bool)
        for s in active_sensors:
            sensor_mask |= (sensor_id == s)
        valid &= sensor_mask

    # Moving/static classification: oracle (GT), trained classifier, or Doppler threshold
    # Multi-task NN also returns per-point class predictions + confidence
    class_ids_all = None   # [N] grouped class IDs (multi-task only)
    class_confs_all = None  # [N] class confidence (multi-task only)
    if oracle:
        # Oracle: GT label_id defines moving/static (same criterion as LSTQ eval)
        moving_mask = np.isin(label_id, list(MOVING_CLASSES))
        moving_proba = np.where(moving_mask, 1.0, 0.0).astype(np.float32)
    elif use_classifier:
        result = predict_moving(
            vr_compensated, rcs, range_sc, azimuth_sc,
            min_doppler=min_doppler,
            x_seq=x_seq, y_seq=y_seq,
            timestamps=timestamps,
            return_probs=True,
        )
        if len(result) == 4:
            # Multi-task NN: (mask, mos_proba, class_ids, class_confs)
            moving_mask, moving_proba, class_ids_all, class_confs_all = result
        else:
            # Binary NN/RF: (mask, mos_proba)
            moving_mask, moving_proba = result
    else:
        speeds = np.abs(vr_compensated)
        moving_mask = speeds >= min_doppler if min_doppler > 0 else np.ones(len(timestamps), dtype=bool)
        # Soft Doppler confidence: ramp from 0 at 0 m/s to 1.0 at 2.0 m/s
        moving_proba = np.clip(speeds / 2.0, 0.0, 1.0).astype(np.float32)
    valid &= moving_mask

    # Group valid points by (frame_idx, sensor_id)
    valid_indices = np.where(valid)[0]
    frame_sensor_groups: Dict[Tuple[int, int], List[int]] = {}
    for idx in valid_indices:
        key = (int(frame_indices[idx]), int(sensor_id[idx]))
        frame_sensor_groups.setdefault(key, []).append(idx)

    frames: Dict[int, List[Dict]] = {}

    for (fi, sid), point_indices in frame_sensor_groups.items():
        pts = np.array(point_indices)

        sensor_key = str(sid)
        if sensor_key not in sensor_calibs:
            continue
        sensor_yaw = float(sensor_calibs[sensor_key]["yaw"])

        if skip_dbscan:
            # Point mode: each radar point becomes its own detection.
            # Graph clustering downstream groups them via Mahalanobis distance.
            for pt_idx in pts:
                pt_x = float(x_seq[pt_idx])
                pt_y = float(y_seq[pt_idx])
                pt_range = float(range_sc[pt_idx])
                pt_azimuth = float(azimuth_sc[pt_idx])
                pt_rcs = float(rcs[pt_idx])
                pt_vr = float(vr_compensated[pt_idx])
                pt_yaw = float(yaw_interp[pt_idx])
                yaw_global = pt_yaw + sensor_yaw

                # Class: oracle GT, multi-task NN, or Doppler/RCS heuristic
                if oracle:
                    lid = int(label_id[pt_idx])
                    class_name_pred = LABEL_ID_TO_PROFILE_NAME.get(lid, "radar_vehicle")
                elif class_ids_all is not None:
                    grouped_cls = int(class_ids_all[pt_idx])
                    class_name_pred = GROUPED_CLASS_TO_PROFILE_NAME.get(grouped_cls, "radar_vehicle")
                else:
                    class_name_pred = _classify_radar_point(abs(pt_vr), pt_rcs)
                # Confidence: oracle=1.0, otherwise MOS probability
                conf = 1.0 if oracle else max(0.10, float(moving_proba[pt_idx]))

                # R_bev from sensor geometry only (no confidence scaling — per paper)
                R_bev, R_pose = _radar_bev_covariance(
                    pt_range, pt_azimuth, yaw_global,
                    min_eigenvalue=min_eigenvalue,
                )

                feat = _build_radar_feature(pt_rcs, pt_vr, pt_range, pt_azimuth)

                az_global = pt_azimuth + yaw_global
                vx_bev = pt_vr * np.sin(az_global)
                vy_bev = pt_vr * np.cos(az_global)

                det = {
                    "cam_id": sid,
                    "bbox": None,
                    "conf": conf,
                    "z_bev": np.array([pt_x, pt_y], dtype=np.float32),
                    "R_bev": R_bev,
                    "R_pose": R_pose,
                    "vec": feat,
                    "feature": feat,
                    "class_name": class_name_pred,
                    "radar_track_id": int(track_id[pt_idx]),
                    "radar_label_id": int(label_id[pt_idx]),
                    "rcs": pt_rcs,
                    "vr_compensated": pt_vr,
                    "azimuth_global": az_global,
                    "velocity_bev": np.array([vx_bev, vy_bev], dtype=np.float32),
                    "n_points": 1,
                    "raw_point_indices": [int(pt_idx)],
                }
                frames.setdefault(fi, []).append(det)
        else:
            # DBSCAN mode: cluster nearby points into object-level detections
            xy = np.column_stack([x_seq[pts], y_seq[pts]])
            cluster_labels = _dbscan_precluster(xy, eps=dbscan_eps, min_samples=2)

            unique_labels = set(cluster_labels)
            unique_labels.discard(-1)  # skip noise
            for cl_id in sorted(unique_labels):
                cl_mask = cluster_labels == cl_id
                cl_pts = pts[cl_mask]
                if len(cl_pts) == 0:
                    continue

                cl_x = float(x_seq[cl_pts].mean())
                cl_y = float(y_seq[cl_pts].mean())
                cl_range = float(range_sc[cl_pts].mean())
                cl_azimuth = float(azimuth_sc[cl_pts].mean())
                cl_rcs = float(rcs[cl_pts].max())
                cl_vr = float(vr_compensated[cl_pts].mean())
                cl_yaw = float(yaw_interp[cl_pts[0]])
                yaw_global = cl_yaw + sensor_yaw

                # Cluster-level Doppler filter (only when not using classifier or oracle)
                cl_speed = abs(cl_vr)
                if not use_classifier and not oracle and cl_speed < min_doppler:
                    continue

                # Class: oracle GT, multi-task NN (majority vote), or Doppler/RCS heuristic
                n_pts_cl = len(cl_pts)
                if oracle:
                    cl_labels_raw = label_id[cl_pts]
                    majority_lid = int(np.bincount(cl_labels_raw, minlength=12).argmax())
                    class_name_pred = LABEL_ID_TO_PROFILE_NAME.get(majority_lid, "radar_vehicle")
                elif class_ids_all is not None:
                    cl_classes = class_ids_all[cl_pts]
                    grouped_cls = int(np.bincount(cl_classes, minlength=NUM_CLASSES).argmax())
                    class_name_pred = GROUPED_CLASS_TO_PROFILE_NAME.get(grouped_cls, "radar_vehicle")
                else:
                    class_name_pred = _classify_radar_cluster(cl_speed, cl_rcs, n_pts_cl)
                # Confidence: oracle=1.0, otherwise MOS probability
                conf = 1.0 if oracle else max(0.10, float(moving_proba[cl_pts].mean()))

                cl_labels = label_id[cl_pts]
                cl_lid = int(np.bincount(cl_labels).argmax())

                cl_tids = track_id[cl_pts]
                nonzero_tids = cl_tids[cl_tids > 0]
                cl_track_id = int(np.bincount(nonzero_tids).argmax()) if len(nonzero_tids) > 0 else 0

                # R_bev from sensor geometry only (no confidence scaling — per paper)
                R_bev, R_pose = _radar_bev_covariance(
                    cl_range, cl_azimuth, yaw_global,
                    min_eigenvalue=min_eigenvalue,
                )
                feat = _build_radar_feature(cl_rcs, cl_vr, cl_range, cl_azimuth)

                az_global = cl_azimuth + yaw_global
                vx_bev = cl_vr * np.sin(az_global)
                vy_bev = cl_vr * np.cos(az_global)

                det = {
                    "cam_id": sid,
                    "bbox": None,
                    "conf": conf,
                    "z_bev": np.array([cl_x, cl_y], dtype=np.float32),
                    "R_bev": R_bev,
                    "R_pose": R_pose,
                    "vec": feat,
                    "feature": feat,
                    "class_name": class_name_pred,
                    "radar_track_id": cl_track_id,
                    "radar_label_id": cl_lid,
                    "rcs": cl_rcs,
                    "vr_compensated": cl_vr,
                    "azimuth_global": az_global,
                    "velocity_bev": np.array([vx_bev, vy_bev], dtype=np.float32),
                    "n_points": n_pts_cl,
                    "raw_point_indices": cl_pts.tolist(),
                }
                frames.setdefault(fi, []).append(det)

    return frames, odo_ts, odo_x, odo_y, odo_yaw


def load_sequence_raw(h5_path: str) -> Dict[str, np.ndarray]:
    """Load all raw radar point arrays from an HDF5 sequence file.

    Returns a dict of arrays indexed by point index (consistent across calls).
    Used by LSTQ evaluation to access per-point GT labels and track IDs.
    """
    with h5py.File(h5_path, "r") as f:
        rd = f["radar_data"]
        timestamps = rd["timestamp"][:]
        x_seq = rd["x_seq"][:]
        y_seq = rd["y_seq"][:]
        range_sc = rd["range_sc"][:]
        azimuth_sc = rd["azimuth_sc"][:]
        rcs_arr = rd["rcs"][:]
        vr_compensated = rd["vr_compensated"][:]
        sensor_id = rd["sensor_id"][:]
        label_id = rd["label_id"][:]
        track_id_raw = rd["track_id"][:]

        odo = f["odometry"]
        odo_ts = odo["timestamp"][:]
        odo_x = odo["x_seq"][:]
        odo_y = odo["y_seq"][:]
        odo_yaw = odo["yaw_seq"][:]

    # Convert track_id UUIDs to stable integer IDs
    _tid_map: Dict[bytes, int] = {b"": 0}
    _tid_counter = 1
    track_id = np.zeros(len(track_id_raw), dtype=np.int64)
    for i, raw_tid in enumerate(track_id_raw):
        tid_bytes = bytes(raw_tid)
        if tid_bytes not in _tid_map:
            _tid_map[tid_bytes] = _tid_counter
            _tid_counter += 1
        track_id[i] = _tid_map[tid_bytes]

    return {
        "timestamps": timestamps,
        "x_seq": x_seq,
        "y_seq": y_seq,
        "range_sc": range_sc,
        "azimuth_sc": azimuth_sc,
        "rcs": rcs_arr,
        "vr_compensated": vr_compensated,
        "sensor_id": sensor_id,
        "label_id": label_id,
        "track_id": track_id,
        "odo_ts": odo_ts,
        "odo_x": odo_x,
        "odo_y": odo_y,
        "odo_yaw": odo_yaw,
    }


# ======================== Scene Iterator ========================

def iter_radarscenes_scenes(
    radarscenes_root: str,
    dt_frame: float = 0.05,
    max_scenes: int = 0,
    min_doppler: float = 0.5,
    min_rcs: float = -10.0,
    active_sensors: Optional[Set[int]] = None,
    scene_ids: Optional[List[str]] = None,
    use_classifier: bool = False,
    skip_dbscan: bool = False,
) -> Iterator[Tuple[str, str, Iterator[Dict]]]:
    """Iterate over RadarScenes sequences.

    Yields: (scene_id, scene_name, frame_iterator)
    where frame_iterator yields dicts:
        {frame_idx, detections, timestamp, ego_pose}
    """
    root = Path(radarscenes_root)

    # Find the actual data directory (handles nested extraction layouts)
    data_dir = _find_data_dir(root)

    sensor_calibs = _load_sensor_calibration(root)

    sequence_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "radar_data.h5").exists()
    ])

    scene_count = 0
    for seq_dir in sequence_dirs:
        scene_name = seq_dir.name
        if scene_ids is not None and scene_name not in scene_ids:
            continue
        if max_scenes > 0 and scene_count >= max_scenes:
            break

        h5_path = str(seq_dir / "radar_data.h5")

        frames, odo_ts, odo_x, odo_y, odo_yaw = _hdf5_to_frames(
            h5_path, sensor_calibs,
            dt_frame=dt_frame,
            min_doppler=min_doppler,
            min_rcs=min_rcs,
            active_sensors=active_sensors,
            use_classifier=use_classifier,
            skip_dbscan=skip_dbscan,
        )

        if not frames:
            continue

        def _make_frame_iter(frames_dict, odo_ts, odo_x, odo_y, odo_yaw, dt_frame, t0_us):
            for frame_idx in sorted(frames_dict.keys()):
                dets = frames_dict[frame_idx]
                t_frame = t0_us + frame_idx * dt_frame * 1e6
                ego_x = float(np.interp(t_frame, odo_ts.astype(np.float64), odo_x.astype(np.float64)))
                ego_y = float(np.interp(t_frame, odo_ts.astype(np.float64), odo_y.astype(np.float64)))
                ego_yaw = float(np.interp(t_frame, odo_ts.astype(np.float64), odo_yaw.astype(np.float64)))
                yield {
                    "frame_idx": frame_idx,
                    "detections": dets,
                    "timestamp": t_frame,
                    "ego_pose": {"x": ego_x, "y": ego_y, "yaw": ego_yaw},
                }

        # Compute t0 for this sequence
        with h5py.File(h5_path, "r") as f:
            t0_us = float(f["radar_data"]["timestamp"][:].min())

        frame_iter = _make_frame_iter(frames, odo_ts, odo_x, odo_y, odo_yaw, dt_frame, t0_us)
        yield (scene_name, scene_name, frame_iter)
        scene_count += 1


def iter_radarscenes_scenes_lstq(
    radarscenes_root: str,
    dt_frame: float = 0.05,
    max_scenes: int = 0,
    min_doppler: float = 0.5,
    min_rcs: float = -10.0,
    active_sensors: Optional[Set[int]] = None,
    scene_ids: Optional[List[str]] = None,
    dbscan_eps: float = 2.0,
    use_classifier: bool = False,
    skip_dbscan: bool = False,
    oracle: bool = False,
) -> Iterator[Tuple[str, str, Iterator[Dict], Dict[str, np.ndarray]]]:
    """Iterate scenes with raw point arrays for LSTQ evaluation.

    Like iter_radarscenes_scenes but also yields the raw point data dict
    so the runner can propagate object-level track IDs back to individual
    radar points for LSTQ scoring.

    Yields: (scene_id, scene_name, frame_iterator, raw_data)
    where raw_data is the dict from load_sequence_raw() containing all
    per-point arrays (label_id, track_id, timestamps, etc.).
    """
    root = Path(radarscenes_root)
    data_dir = _find_data_dir(root)
    sensor_calibs = _load_sensor_calibration(root)

    sequence_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "radar_data.h5").exists()
    ])

    scene_count = 0
    for seq_dir in sequence_dirs:
        scene_name = seq_dir.name
        if scene_ids is not None and scene_name not in scene_ids:
            continue
        if max_scenes > 0 and scene_count >= max_scenes:
            break

        h5_path = str(seq_dir / "radar_data.h5")

        # Load raw data once (shared between frame building and LSTQ eval)
        raw_data = load_sequence_raw(h5_path)

        frames, odo_ts, odo_x, odo_y, odo_yaw = _hdf5_to_frames(
            h5_path, sensor_calibs,
            dt_frame=dt_frame,
            min_doppler=min_doppler,
            min_rcs=min_rcs,
            active_sensors=active_sensors,
            dbscan_eps=dbscan_eps,
            use_classifier=use_classifier,
            skip_dbscan=skip_dbscan,
            oracle=oracle,
        )

        if not frames:
            continue

        # Build frame→raw_point_indices mapping for ALL points (including static)
        # so LSTQ can evaluate classification of every point
        t0 = float(raw_data["timestamps"].min())
        dt_us = dt_frame * 1e6
        all_frame_indices = ((raw_data["timestamps"] - t0) / dt_us).astype(int)
        frame_to_all_points: Dict[int, List[int]] = {}
        for pi in range(len(raw_data["timestamps"])):
            fi = int(all_frame_indices[pi])
            frame_to_all_points.setdefault(fi, []).append(pi)
        raw_data["frame_to_all_points"] = frame_to_all_points
        raw_data["all_frame_indices"] = all_frame_indices

        def _make_frame_iter(frames_dict, odo_ts, odo_x, odo_y, odo_yaw, dt_frame, t0_us):
            for frame_idx in sorted(frames_dict.keys()):
                dets = frames_dict[frame_idx]
                t_frame = t0_us + frame_idx * dt_frame * 1e6
                ego_x = float(np.interp(t_frame, odo_ts.astype(np.float64), odo_x.astype(np.float64)))
                ego_y = float(np.interp(t_frame, odo_ts.astype(np.float64), odo_y.astype(np.float64)))
                ego_yaw = float(np.interp(t_frame, odo_ts.astype(np.float64), odo_yaw.astype(np.float64)))
                yield {
                    "frame_idx": frame_idx,
                    "detections": dets,
                    "timestamp": t_frame,
                    "ego_pose": {"x": ego_x, "y": ego_y, "yaw": ego_yaw},
                }

        t0_us = t0
        frame_iter = _make_frame_iter(frames, odo_ts, odo_x, odo_y, odo_yaw, dt_frame, t0_us)
        yield (scene_name, scene_name, frame_iter, raw_data)
        scene_count += 1


# ======================== Flat Frame Iterator (Registry) ========================

def iter_radarscenes_frames(
    root: str,
    camera: str = "",
    target_hw: Tuple[int, int] = (0, 0),
    every_n: int = 1,
    max_frames: int = 0,
    **kwargs,
) -> Iterator[Tuple]:
    """Flat frame iterator for DatasetSpec.iter_frames compatibility.

    Yields 7-element tuples (camera-dataset compatibility format):
        (rgb, hw, K_img, T_wc, T_ec, reserved_0, reserved_1)
    For radar: rgb=None, K=identity, T_wc=ego_pose_4x4.

    NOTE: For RadarScenes, prefer iter_radarscenes_scenes() in the runner.
    This is provided only for registry API compatibility.
    """
    frame_count = 0
    for _sid, _sname, frame_iter in iter_radarscenes_scenes(root):
        for frame in frame_iter:
            if frame_count % every_n != 0:
                frame_count += 1
                continue
            if max_frames > 0 and frame_count >= max_frames:
                return

            ego = frame["ego_pose"]
            c, s = np.cos(ego["yaw"]), np.sin(ego["yaw"])
            T_wc = np.eye(4, dtype=np.float32)
            T_wc[0, 0] = c; T_wc[0, 1] = -s
            T_wc[1, 0] = s; T_wc[1, 1] = c
            T_wc[0, 3] = ego["x"]
            T_wc[1, 3] = ego["y"]

            yield (
                None,                               # rgb (no image)
                (0, 0),                              # hw
                np.eye(3, dtype=np.float32),         # K_img
                T_wc,                                # T_wc
                np.eye(4, dtype=np.float32),         # T_ec
                None,                                # reserved_0
                None,                                # reserved_1
            )
            frame_count += 1
