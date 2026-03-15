"""
ModTrack End-to-End Evaluation Script with Ablation Support
"""

import json
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# ModTrack modules
from modtrack.core.tracker.config import (
    get_bev_min_variance,
    get_config_for_mode,
    get_depth_max,
    get_depth_prior_min_range_m,
    get_depth_var_cutoff,
    get_phd_process_noise_scale,
    get_pedestrian_height_m,
    get_footpoint_base_sigma_scale,
    get_footpoint_min_depth_var,
    get_footpoint_lift_disagree_m,
    get_footpoint_bbox_disagree_m,
    get_footpoint_lift_confident_var_max,
    get_footpoint_lift_inflate,
    get_footpoint_bbox_agree_shrink,
    get_r_pose_variance,
)
from modtrack.core.tracker.semantic_encoder import SemanticEncoder
from modtrack.core.tracker.identity_matching import DualModalMatcher
from modtrack.core.tracker.gm_phd_hmm import (
    IdentityInformedGMPHDFilter,
)
from modtrack.core.tracker.motion_models import get_profile_for_yolo_class, get_profile_by_name_for_dataset
import modtrack.core.tracker.gm_phd_hmm as gm_phd_hmm_module
from modtrack.core.tracker.bev_tracker import (
    graph_clustering,
    kalman_fusion,
    semantic_similarity,
)
from modtrack.core.lstq import LSTQEvaluator, LSTQ_CLASS_MOVING, LSTQ_CLASS_STATIC
from modtrack.data.adapters.registry import get_dataset_spec

try:
    from modtrack.core.ground_plane_projection import bev_project, footpoint_ray_plane
    from modtrack.core.depth_estimator import CamEncodeDepthEstimator
    from ultralytics import YOLO
    _HAS_CAMERA_MODULES = True
except ImportError:
    _HAS_CAMERA_MODULES = False

try:
    from modtrack.metrics.trackeval_backend import MOTEvaluator as TrackEvalMOTEvaluator
    from modtrack.metrics.pymotmetrics_backend import MOTEvaluator as PyMotMetricsEvaluator
    from modtrack.metrics.gospa_backend import MOTEvaluator as GOSPAEvaluator
    _HAS_MOT_EVAL = True
except Exception:
    _HAS_MOT_EVAL = False

def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModTrackEvaluator:
    """
    End-to-end ModTrack evaluation pipeline with MOT metrics.
    """

    def __init__(
        self,
        dataset: str,
        dataset_root: str,
        yolo_checkpoint: str,
        lift_checkpoint: Optional[str] = None,
        semantic_checkpoint: Optional[str] = None,
        matching_mode: str = "joint",
        device: Optional[str] = None,
        max_distance_mot: float = 1.0,
        cameras: Optional[str] = None,
        seed: int = 42,
        track_classes: Optional[str] = None,
        yolo_imgsz: int = 0,
        det_min_conf: Optional[float] = None,
        high_conf_thresh: Optional[float] = None,
        low_conf_min: Optional[float] = None,
    ):
        """
        Initialize evaluator.

        Args:
            dataset: Dataset name (e.g., wildtrack, multiviewx)
            dataset_root: Path to dataset root directory
            yolo_checkpoint: Path to finetuned YOLO model
            lift_checkpoint: Path to finetuned Lift model checkpoint
            semantic_checkpoint: Path to semantic feature model checkpoint (OSNet)
            matching_mode: "spatial_only", "semantic_only", or "joint"
            device: torch device
            max_distance_mot: MOT evaluation distance in meters
            cameras: Comma-separated camera indices to use (e.g. "0,1,3,5"). None = all.
            seed: Random seed for reproducible sampling/clustering
            track_classes: Comma-separated COCO YOLO class IDs to track
            yolo_imgsz: YOLO inference image size (0 = auto: 640)
            det_min_conf: Minimum detector confidence to keep a raw YOLO detection.
            high_conf_thresh: High-confidence threshold for ByteTrack stage-1 matching.
            low_conf_min: Minimum confidence for ByteTrack stage-2 matching.
        """
        _set_global_seed(seed)
        self.dataset = dataset.lower().strip()
        self.dataset_root = Path(dataset_root)
        self.dataset_spec = get_dataset_spec(self.dataset)
        self.target_hw = (360, 640)
        self._is_radar = self.dataset == "radarscenes"
        if self._is_radar:
            matching_mode = "spatial"
        self.matching_mode = matching_mode
        self.pre_fusion_mode = "spatial" if self.matching_mode == "joint" else self.matching_mode
        self.camera_names = list(self.dataset_spec.cameras)
        self.output_dir = Path("./results") / self.dataset / self.matching_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_distance_mot = max_distance_mot
        self.active_camera_indices: Optional[List[int]] = None
        if cameras is not None:
            self.active_camera_indices = [int(c.strip()) for c in cameras.split(",")]

        # Track class filtering
        self.track_class_ids: Optional[set] = None
        if track_classes:
            self.track_class_ids = {int(c.strip()) for c in track_classes.split(",")}

        # YOLO imgsz
        if yolo_imgsz <= 0:
            yolo_imgsz = 640
        self.yolo_imgsz = yolo_imgsz
        if self.dataset == "wildtrack":
            default_det_min_conf = 0.1
            default_high_conf = 0.5
            default_low_conf = 0.2
        else:
            default_det_min_conf = 0.63
            default_high_conf = 0.5
            default_low_conf = 0.3
        self.det_min_conf = float(
            default_det_min_conf if det_min_conf is None else det_min_conf
        )
        self.high_conf_thresh = float(
            default_high_conf if high_conf_thresh is None else high_conf_thresh
        )
        self.low_conf_min = float(
            default_low_conf if low_conf_min is None else low_conf_min
        )
        if not (0.0 <= self.det_min_conf <= 1.0):
            raise ValueError(
                f"det_min_conf must be in [0,1], got {self.det_min_conf}"
            )
        if not (0.0 <= self.low_conf_min <= 1.0):
            raise ValueError(
                f"low_conf_min must be in [0,1], got {self.low_conf_min}"
            )
        if not (0.0 <= self.high_conf_thresh <= 1.0):
            raise ValueError(
                f"high_conf_thresh must be in [0,1], got {self.high_conf_thresh}"
            )
        if self.low_conf_min >= self.high_conf_thresh:
            raise ValueError(
                f"low_conf_min ({self.low_conf_min}) must be < "
                f"high_conf_thresh ({self.high_conf_thresh})"
            )
        if det_min_conf is not None and self.det_min_conf > self.high_conf_thresh:
            print(
                "  [Config Warning] det_min_conf > high_conf_thresh; "
                "low-confidence stage will be effectively disabled."
            )
        print(
            "  Detection confidence: "
            f"det_min={self.det_min_conf:.2f}, "
            f"high_conf={self.high_conf_thresh:.2f}, "
            f"low_conf_min={self.low_conf_min:.2f}"
        )

        self.bev_min_variance = get_bev_min_variance(self.dataset)
        self.r_pose_variance = get_r_pose_variance(self.dataset)
        self.r_pose_bev = np.eye(2, dtype=np.float32) * self.r_pose_variance

        self.config = get_config_for_mode(matching_mode, dataset=self.dataset)
        self.clustering_max_euclidean_dist = (
            10.0 if self._is_radar else 0.5
        )
        self.new_id_cost = 12.0
        print(f"\n[ModTrackEvaluator] Dataset: {self.dataset} | Mode: {matching_mode} | sem_thresh: {self.config.sem_thresh}")
        if self.pre_fusion_mode != self.matching_mode:
            print(f"  Pre-fusion mode override: {self.pre_fusion_mode} (clustering + fusion)")
        if self.active_camera_indices is not None:
            print(f"  Active cameras: {self.active_camera_indices} ({len(self.active_camera_indices)}/{len(self.camera_names)})")

        self.intrinsics = {}
        self.extrinsics = {}
        self.cam_name_to_idx: Dict[str, int] = {}
        if not self._is_radar:
            self._load_dataset_calibrations()
        else:
            for cam_id, cam_name in enumerate(self.camera_names):
                self.cam_name_to_idx[cam_name] = cam_id

        if self._is_radar:
            print(f"[ModTrackEvaluator] RadarScenes mode: skipping YOLO/Lift/semantic encoder (radar has no images)")
            self.yolo = None
            self.lift = None
            self.semantic_encoder = None
            self.depth_bins = None
        else:
            print(f"[ModTrackEvaluator] Loading YOLO: {yolo_checkpoint} (imgsz={self.yolo_imgsz})")
            self.yolo = YOLO(yolo_checkpoint)

        if not self._is_radar:
            # Load Lift depth estimator
            max_depth = get_depth_max(self.dataset)
            self.depth_bins = torch.linspace(1.0, max_depth, 41, device=self.device)
            depth_var_cutoff = get_depth_var_cutoff(self.dataset)
            pedestrian_height_m = get_pedestrian_height_m(self.dataset)
            depth_prior_min_range_m = get_depth_prior_min_range_m(self.dataset)
            print(f"[ModTrackEvaluator] Loading Lift depth estimator")
            self.lift = CamEncodeDepthEstimator(
                lift_checkpoint,
                depth_bins=self.depth_bins,
                depth_sample_count=256,
                depth_sample_sigma_scale=0.2,
                depth_prob_power=2.0,
                pedestrian_height_m=pedestrian_height_m,
                depth_prior_min_range_m=depth_prior_min_range_m,
                depth_prior_sigma_scale=0.05,
                depth_var_cutoff=depth_var_cutoff,
                device=self.device,
            )

            # Load semantic encoder (OSNet)
            print("[ModTrackEvaluator] Loading semantic encoder (OSNet)")
            self.semantic_encoder = None
            if matching_mode in ["semantic", "joint"]:
                self.semantic_encoder = SemanticEncoder(
                    checkpoint_path=semantic_checkpoint, device=self.device,
                )
                if self.semantic_encoder.model is None:
                    raise RuntimeError(
                        "Semantic mode requires OSNet semantic encoder, but it failed to initialize. "
                        "Install torchreid (or run `uv pip install -r requirements-train.txt`) "
                        "and ensure semantic checkpoint is valid."
                    )

        # Initialize matching module
        self.matcher = DualModalMatcher(
            matching_mode=self.pre_fusion_mode,
            sigma_spatial=1.0,
            tau_geo=self.config.tau_geo,
            tau_sem=self.config.tau_sem,
            lambda_kl=self.config.lambda_kl,
            mahal_thresh=self.config.mahal_thresh,
        )

        # Initialize GM-PHD-HMM filter
        # Determine sampling period from dataset frame rate.
        DATASET_FPS = {"wildtrack": 2.0, "multiviewx": 2.0}
        dt_base = 1.0 / DATASET_FPS.get(self.dataset, 2.0)
        dt = dt_base
        self.phd_process_noise_scale = get_phd_process_noise_scale(self.dataset)
        birth_scale = 1.0
        # Dataset-specific lifecycle thresholds
        if self._is_radar:
            confirmed_miss_tol = 3   # Highway objects: allow 3 misses before CONFIRMED → LOST
            max_lost = 5             # Keep LOST tracks for 5 frames
            tent_miss_tol = 2        # Allow 2 misses before deleting TENTATIVE
        elif self.dataset == "wildtrack":
            confirmed_miss_tol = 0
            max_lost = 2
            tent_miss_tol = 1
        else:
            confirmed_miss_tol = 1
            max_lost = 2
            tent_miss_tol = 0
        # GM-PHD covariance clamp
        if self._is_radar:
            cov_clamp_mode = "diag"
            cov_clamp_max_vel_var = 100.0
        else:
            cov_clamp_mode = "eig"
            cov_clamp_max_vel_var = 4.0
        mahal_thresh = 13.82 if self.dataset == "radarscenes" else 9.21
        self._radar_dt_frame = 0.2  
        if self._is_radar:
            dt = self._radar_dt_frame  # Will be re-created per scene
        self.phd_filter = IdentityInformedGMPHDFilter(
            dt=dt,
            process_noise_scale=self.phd_process_noise_scale,
            birth_threshold_scale=birth_scale,
            confirmed_miss_tolerance=confirmed_miss_tol,
            max_lost_frames=max_lost,
            tentative_miss_tolerance=tent_miss_tol,
            cov_clamp_mode=cov_clamp_mode,
            cov_clamp_max_vel_var=cov_clamp_max_vel_var,
            mahal_thresh=mahal_thresh,
            dataset=self.dataset,
            use_class_birth_mode_priors=self._is_radar,
        )

        # MOT evaluator (not needed for RadarScenes — uses LSTQ instead)
        self.mot_evaluator = None
        if not self._is_radar and _HAS_MOT_EVAL:
            self.mot_evaluator = TrackEvalMOTEvaluator(max_distance=self.max_distance_mot)

        # Timing statistics
        self.timing_stats = defaultdict(list)
        self.per_frame_timing: List[Dict[str, float]] = []
        # Exclude initial warm-up frames from reported timing aggregates.
        self.timing_warmup_skip_frames = 5
        self._last_detection_depth_detail: Dict[str, float] = {}
        self.total_frames_processed = 0
        self.fusion_assoc_threshold_m = 0.5
        self.fusion_f1_frame_metrics: List[Dict[str, float]] = []

    def get_runtime_hyperparameter_snapshot(self) -> Dict[str, float]:
        """Return the currently active post-projection hyperparameters."""
        ped_profile = get_profile_by_name_for_dataset("pedestrian", self.dataset)
        ped_birth_threshold = float(getattr(ped_profile, "birth_weight_threshold", 0.65))
        ped_survival = float(getattr(ped_profile, "survival_probability", self.phd_filter.survival_prob))
        ped_detection = getattr(ped_profile, "detection_probability", None)
        if ped_detection is None:
            ped_detection = self.phd_filter.detection_prob
        return {
            "r_pose_variance": float(self.r_pose_variance),
            "bev_min_variance": float(self.bev_min_variance),
            "mahal_thresh": float(self.config.mahal_thresh),
            "max_euclidean_dist": float(self.clustering_max_euclidean_dist),
            "default_det_min_conf": float(self.det_min_conf),
            "sem_thresh": float(self.config.sem_thresh),
            "tau_geo": float(self.config.tau_geo),
            "tau_sem": float(self.config.tau_sem),
            "lambda_kl": float(self.config.lambda_kl),
            "new_id_cost": float(self.new_id_cost),
            "WEIGHT_PRUNE_THRESHOLD": float(gm_phd_hmm_module.WEIGHT_PRUNE_THRESHOLD),
            "phd_process_noise_scale": float(self.phd_filter.process_noise_scale),
            "high_conf_thresh": float(self.high_conf_thresh),
            "low_conf_min": float(self.low_conf_min),
            "birth_weight_threshold": ped_birth_threshold,
            "survival_probability": ped_survival,
            "detection_probability": float(ped_detection),
            "IDENTITY_ASSOC_BOOST": float(self.phd_filter.identity_assoc_boost),
            "w_boost": float(gm_phd_hmm_module.MATCHED_CONFIRMED_WEIGHT_BOOST),
            "SEMANTIC_COST_BOOST": float(gm_phd_hmm_module.SEMANTIC_COST_BOOST),
            "LOST_TRACK_REID_SEMANTIC_BOOST": float(gm_phd_hmm_module.LOST_TRACK_REID_SEMANTIC_BOOST),
            "TURN_PENALTY_BASE": float(gm_phd_hmm_module.TURN_PENALTY_BASE),
            "N_INIT": float(gm_phd_hmm_module.N_INIT),
            "tentative_miss_tolerance": float(self.phd_filter.tentative_miss_tolerance),
            "confirmed_miss_tolerance": float(self.phd_filter.confirmed_miss_tolerance),
            "max_lost_frames": float(self.phd_filter.max_lost_frames),
        }

    def apply_runtime_hyperparameter_overrides(self, overrides: Optional[Dict[str, float]] = None) -> None:
        """Apply runtime hyperparameter overrides across clustering/fusion/tracking modules."""
        if not overrides:
            return

        updates = {str(k): float(v) for k, v in overrides.items() if v is not None}
        for key, value in updates.items():
            if key == "id_assignment_gate":
                # Backward compatibility: id_assignment_gate is now unified with mahal_thresh.
                key = "mahal_thresh"
            if key == "r_pose_variance":
                if value <= 0.0:
                    raise ValueError("r_pose_variance must be > 0")
                self.r_pose_variance = value
                self.r_pose_bev = np.eye(2, dtype=np.float32) * value
            elif key == "bev_min_variance":
                if value <= 0.0:
                    raise ValueError("bev_min_variance must be > 0")
                self.bev_min_variance = value
            elif key == "mahal_thresh":
                if value <= 0.0:
                    raise ValueError("mahal_thresh must be > 0")
                self.config.mahal_thresh = value
            elif key == "max_euclidean_dist":
                if value <= 0.0:
                    raise ValueError("max_euclidean_dist must be > 0")
                self.clustering_max_euclidean_dist = value
            elif key == "default_det_min_conf":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("default_det_min_conf must be in [0, 1]")
                self.det_min_conf = value
            elif key == "sem_thresh":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("sem_thresh must be in [0, 1]")
                self.config.sem_thresh = value
            elif key == "tau_geo":
                if value <= 0.0:
                    raise ValueError("tau_geo must be > 0")
                self.config.tau_geo = value
            elif key == "tau_sem":
                if value <= 0.0:
                    raise ValueError("tau_sem must be > 0")
                self.config.tau_sem = value
            elif key == "lambda_kl":
                if value < 0.0:
                    raise ValueError("lambda_kl must be >= 0")
                self.config.lambda_kl = value
            elif key == "new_id_cost":
                if value <= 0.0:
                    raise ValueError("new_id_cost must be > 0")
                self.new_id_cost = value
            elif key == "WEIGHT_PRUNE_THRESHOLD":
                if value < 0.0:
                    raise ValueError("WEIGHT_PRUNE_THRESHOLD must be >= 0")
                gm_phd_hmm_module.WEIGHT_PRUNE_THRESHOLD = float(value)
            elif key == "phd_process_noise_scale":
                if value <= 0.0:
                    raise ValueError("phd_process_noise_scale must be > 0")
                self.phd_process_noise_scale = value
                self.phd_filter.process_noise_scale = value
            elif key == "high_conf_thresh":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("high_conf_thresh must be in [0, 1]")
                self.high_conf_thresh = value
            elif key == "low_conf_min":
                if not (0.0 <= value <= 1.0):
                    raise ValueError("low_conf_min must be in [0, 1]")
                self.low_conf_min = value
            elif key == "birth_weight_threshold":
                if not (0.0 < value <= 1.0):
                    raise ValueError("birth_weight_threshold must be in (0, 1]")
                ped_profile = get_profile_by_name_for_dataset("pedestrian", self.dataset)
                ped_profile.birth_weight_threshold = value
            elif key == "survival_probability":
                if not (0.0 < value < 1.0):
                    raise ValueError("survival_probability must be in (0, 1)")
                ped_profile = get_profile_by_name_for_dataset("pedestrian", self.dataset)
                ped_profile.survival_probability = value
                self.phd_filter.survival_prob = value
            elif key == "detection_probability":
                if not (0.0 < value < 1.0):
                    raise ValueError("detection_probability must be in (0, 1)")
                ped_profile = get_profile_by_name_for_dataset("pedestrian", self.dataset)
                ped_profile.detection_probability = value
                self.phd_filter.detection_prob = value
            elif key == "IDENTITY_ASSOC_BOOST":
                if value < 0.0:
                    raise ValueError("IDENTITY_ASSOC_BOOST must be >= 0")
                self.phd_filter.identity_assoc_boost = value
            elif key == "w_boost":
                if value < 0.0:
                    raise ValueError("w_boost must be >= 0")
                gm_phd_hmm_module.MATCHED_CONFIRMED_WEIGHT_BOOST = float(value)
            elif key == "SEMANTIC_COST_BOOST":
                if value < 0.0:
                    raise ValueError("SEMANTIC_COST_BOOST must be >= 0")
                gm_phd_hmm_module.SEMANTIC_COST_BOOST = float(value)
            elif key == "LOST_TRACK_REID_SEMANTIC_BOOST":
                if value < 0.0:
                    raise ValueError("LOST_TRACK_REID_SEMANTIC_BOOST must be >= 0")
                gm_phd_hmm_module.LOST_TRACK_REID_SEMANTIC_BOOST = float(value)
            elif key == "TURN_PENALTY_BASE":
                if value < 0.0:
                    raise ValueError("TURN_PENALTY_BASE must be >= 0")
                gm_phd_hmm_module.TURN_PENALTY_BASE = float(value)
            elif key == "N_INIT":
                n_init_value = int(round(value))
                if n_init_value < 1:
                    raise ValueError("N_INIT must be >= 1")
                gm_phd_hmm_module.N_INIT = n_init_value
            elif key == "tentative_miss_tolerance":
                tent_tol_value = int(round(value))
                if tent_tol_value < 0:
                    raise ValueError("tentative_miss_tolerance must be >= 0")
                self.phd_filter.tentative_miss_tolerance = tent_tol_value
            elif key == "confirmed_miss_tolerance":
                conf_tol_value = int(round(value))
                if conf_tol_value < 0:
                    raise ValueError("confirmed_miss_tolerance must be >= 0")
                self.phd_filter.confirmed_miss_tolerance = conf_tol_value
            elif key == "max_lost_frames":
                max_lost_value = int(round(value))
                if max_lost_value < 0:
                    raise ValueError("max_lost_frames must be >= 0")
                self.phd_filter.max_lost_frames = max_lost_value
            else:
                raise ValueError(f"Unknown runtime hyperparameter override: {key}")

        # Keep matcher and config views synchronized.
        self.matcher.mahal_thresh = float(self.config.mahal_thresh)
        self.matcher.tau_geo = float(self.config.tau_geo)
        self.matcher.tau_sem = float(self.config.tau_sem)
        self.matcher.lambda_kl = float(self.config.lambda_kl)

        if self.low_conf_min >= self.high_conf_thresh:
            raise ValueError(
                f"low_conf_min ({self.low_conf_min}) must be < high_conf_thresh ({self.high_conf_thresh})"
            )

    def _raw_proxy_depth_from_box(
        self,
        *,
        box_height_px: float,
        K_cam: Optional[np.ndarray],
        yolo_class: Optional[int] = None,
    ) -> Optional[float]:
        """
        Depth proxy from bbox height + intrinsics only (no Lift):
            Z = fy * H / h_px

        Returns:
            Positive depth magnitude (meters), or None if unavailable.
        """
        if K_cam is None:
            return None
        ped_h_m = getattr(self.lift, "pedestrian_height_m", None)
        if ped_h_m is None:
            return None
        # Proxy is only meaningful for pedestrians; restrict to YOLO "person" (0) when provided.
        if yolo_class is not None and int(yolo_class) != 0:
            return None
        try:
            fy = float(np.asarray(K_cam, dtype=np.float32)[1, 1])
        except Exception:
            return None
        if not np.isfinite(fy) or fy <= 0.0:
            return None
        h_px = float(box_height_px)
        if not np.isfinite(h_px):
            return None
        h_px = max(h_px, 1.0)
        z = float(fy) * float(ped_h_m) / h_px
        if not np.isfinite(z) or z <= 0.0:
            return None
        return z

    def _footpoint_fused_depth_from_bbox(
        self,
        *,
        det: Dict,
        bbox,
        K_cam: np.ndarray,
        R_cam: np.ndarray,
        t_cam: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Dict[str, float]]:
        """
        Compute a fused depth estimate along the footpoint ray.

        Returns:
            u_foot, v_foot: footpoint pixel coordinates
            depth_mean_signed: fused mean depth along the ray (signed for MultiviewX)
            sigma2_depth: fused depth variance (after disagreement inflation)
            metadata: extra fields (includes depth_fp, depth_mean, sigma2_depth, etc.)
        """
        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
        elif isinstance(bbox, dict):
            x1 = bbox.get("x1", 0.0)
            y1 = bbox.get("y1", 0.0)
            x2 = bbox.get("x2", 0.0)
            y2 = bbox.get("y2", 0.0)
        else:
            x1 = y1 = x2 = y2 = 0.0

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        allow_negative = self.dataset == "multiviewx"
        u_foot, v_foot, depth_fp, X_w = footpoint_ray_plane(
            (x1, y1, x2, y2),
            K_cam,
            R_cam,
            t_cam,
            allow_negative_depth=allow_negative,
        )
        if X_w is None:
            return None, None, None, None, {}

        depth_fp_abs = abs(float(depth_fp))
        base_sigma_scale = get_footpoint_base_sigma_scale(self.dataset)
        min_depth_var = get_footpoint_min_depth_var()
        var_fp = max((base_sigma_scale * depth_fp_abs) ** 2, min_depth_var)

        # BBox proxy cue (for Bayesian mean update only).
        bbox_proxy_depth_m = det.get("bbox_proxy_depth_m")
        bbox_proxy_var = det.get("bbox_proxy_depth_var")
        if bbox_proxy_depth_m is None:
            bbox_proxy_depth_m = self._raw_proxy_depth_from_box(
                box_height_px=float(det.get("box_height", y2 - y1)),
                K_cam=K_cam,
                yolo_class=det.get("yolo_class", None),
            )
            if bbox_proxy_depth_m is not None:
                sigma_scale = float(getattr(self.lift, "depth_prior_sigma_scale", 0.1) or 0.1)
                bbox_proxy_var = max((sigma_scale * float(bbox_proxy_depth_m)) ** 2, min_depth_var)

        depth_mean_abs = depth_fp_abs
        if bbox_proxy_depth_m is not None and bbox_proxy_var is not None:
            prec_fp = 3.0 / max(var_fp, 1e-9)
            prec_bbox = 1.0 / max(float(bbox_proxy_var), 1e-9)
            depth_mean_abs = (prec_fp * depth_fp_abs + prec_bbox * float(bbox_proxy_depth_m)) / (prec_fp + prec_bbox)

        depth_mean_signed = float(np.sign(depth_fp) if depth_fp != 0.0 else 1.0) * depth_mean_abs
        sigma2_depth = max((base_sigma_scale * depth_mean_abs) ** 2, min_depth_var)
        bbox_agree_shrink_applied = False

        if bbox_proxy_depth_m is not None:
            bbox_diff = abs(float(bbox_proxy_depth_m) - depth_mean_abs)
            if bbox_diff <= get_footpoint_bbox_disagree_m(self.dataset):
                bbox_shrink = get_footpoint_bbox_agree_shrink(self.dataset)
                if bbox_shrink < 1.0:
                    sigma2_depth *= bbox_shrink
                    bbox_agree_shrink_applied = True

        lift_depth = det.get("depth_lift", det.get("depth"))
        lift_var = det.get("depth_var_lift", det.get("depth_var"))
        lift_disagree = False
        if lift_depth is not None and lift_var is not None:
            lift_diff = abs(abs(float(lift_depth)) - depth_mean_abs)
            if (
                lift_diff > get_footpoint_lift_disagree_m()
                and float(lift_var) <= get_footpoint_lift_confident_var_max()
            ):
                sigma2_depth *= get_footpoint_lift_inflate()
                lift_disagree = True

        metadata = {
            "u_foot": float(u_foot),
            "v_foot": float(v_foot),
            "depth_fp": float(depth_fp),
            "depth_mean": float(depth_mean_signed),
            "depth_mean_abs": float(depth_mean_abs),
            "sigma2_depth": float(sigma2_depth),
            "bbox_agree_shrink_applied": bool(bbox_agree_shrink_applied),
            "lift_disagree": bool(lift_disagree),
        }
        return float(u_foot), float(v_foot), float(depth_mean_signed), float(sigma2_depth), metadata

    def _load_dataset_calibrations(self):
        """Load camera calibrations for the selected dataset."""
        calibs = self.dataset_spec.load_calibration(str(self.dataset_root))
        if not calibs:
            raise RuntimeError(f"Failed to load calibrations for dataset '{self.dataset}'")

        native_h, native_w = self.dataset_spec.native_hw
        target_h, target_w = self.target_hw
        sx, sy = target_w / float(native_w), target_h / float(native_h)

        self.cam_name_to_idx: Dict[str, int] = {}
        for cam_id, cam_name in enumerate(self.camera_names):
            self.cam_name_to_idx[cam_name] = cam_id
            if cam_name not in calibs:
                print(f"[Warning] Calibration missing for {cam_name}")
                continue

            K, T = calibs[cam_name]
            K_rescaled = K.copy()
            K_rescaled[0, 0] *= sx
            K_rescaled[1, 1] *= sy
            K_rescaled[0, 2] *= sx
            K_rescaled[1, 2] *= sy

            self.intrinsics[cam_id] = K_rescaled
            self.extrinsics[cam_id] = T

        missing = [cam for cam in self.camera_names if self.cam_name_to_idx[cam] not in self.intrinsics]
        if missing:
            raise RuntimeError(f"Missing calibrations for cameras: {', '.join(missing)}")

    def _load_wildtrack_all_gt_points(
        self,
        frame_idx: int,
        frame_token: Optional[str] = None,
    ) -> Dict[int, np.ndarray]:
        """Load ALL annotated WildTrack identities for a frame (no camera visibility filter)."""
        if self.dataset != "wildtrack":
            return {}

        from modtrack.data.adapters.wildtrack import _load_frame_annotations, _position_id_to_world

        ann_dir = self.dataset_root / "annotations_positions"
        records: List[Dict] = []

        candidate_tokens: List[str] = []
        if frame_token is not None:
            token = str(frame_token).strip()
            if token:
                candidate_tokens.extend([token, f"frame_{token}"])

        idx = int(frame_idx)
        candidate_tokens.extend(
            [
                f"{idx}",
                f"{idx:05d}",
                f"{idx:06d}",
                f"frame_{idx:05d}",
                f"frame_{idx:06d}",
            ]
        )

        seen: Set[str] = set()
        for token in candidate_tokens:
            if not token or token in seen:
                continue
            seen.add(token)
            records = _load_frame_annotations(ann_dir, token)
            if records:
                break

        if not records:
            return {}

        gt_points_by_id_all: Dict[int, np.ndarray] = {}
        for record in records:
            pos_id = record.get("positionID")
            if pos_id is None:
                continue
            person_id = int(record.get("personID", pos_id))
            gt_points_by_id_all[person_id] = _position_id_to_world(int(pos_id))
        return gt_points_by_id_all

    def run_detection_stage(
        self,
        images: Dict[int, np.ndarray],
        frame_calibs: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]],
        frame_idx: int,
    ) -> List[Dict]:
        """
        Stage 1: Multi-camera detection with depth estimation.

        Input: Camera images
        Output: List of detections with:
            - cam_id: int
            - bbox: (x1, y1, x2, y2)
            - conf: float
            - z_bev: np.ndarray [2,] (BEV position)
            - R_bev: np.ndarray [2,2] (BEV covariance)
            - feature: np.ndarray [D,] (semantic embedding, optional)
            - depth/depth_var
        """
        stage_perf_start = time.perf_counter()
        det_detail = {
            "detector_forward_ms": 0.0,
            "depth_forward_ms": 0.0,
            "depth_estimation_ms": 0.0,
            "semantic_feature_ms": 0.0,
            "bev_projection_ms": 0.0,
            "detection_other_ms": 0.0,
            "total_ms": 0.0,
            "cameras_processed": 0.0,
            "detections_out": 0.0,
        }

        def _finalize_detection_detail(dets: List[Dict], elapsed_s: float) -> None:
            total_ms = float(elapsed_s * 1000.0)
            tracked_ms = (
                float(det_detail["detector_forward_ms"])
                + float(det_detail["depth_forward_ms"])
                + float(det_detail["depth_estimation_ms"])
                + float(det_detail["semantic_feature_ms"])
                + float(det_detail["bev_projection_ms"])
            )
            det_detail["total_ms"] = total_ms
            det_detail["detection_other_ms"] = max(0.0, total_ms - tracked_ms)
            det_detail["detections_out"] = float(len(dets))
            self._last_detection_depth_detail = dict(det_detail)

        all_detections = []
        global_semantic_batch_targets: List[Tuple[Dict, np.ndarray, Tuple[int, int, int, int]]] = []

        image_rgb_by_cam: Dict[int, np.ndarray] = {}
        image_bgr_by_cam: Dict[int, np.ndarray] = {}
        for cam_id, image in images.items():
            rgb = np.asarray(image)
            image_rgb_by_cam[cam_id] = rgb
            image_bgr_by_cam[cam_id] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Run YOLO once for all cameras that require detector inference.
        yolo_results_by_cam: Dict[int, object] = {}
        yolo_cam_ids: List[int] = list(image_rgb_by_cam.keys())

        if yolo_cam_ids:
            yolo_inputs = [image_bgr_by_cam[cid] for cid in yolo_cam_ids]
            t_detector = time.perf_counter()
            yolo_batch_results = self.yolo(yolo_inputs, imgsz=self.yolo_imgsz, verbose=False)
            det_detail["detector_forward_ms"] += (time.perf_counter() - t_detector) * 1000.0
            if yolo_batch_results is None:
                yolo_batch_results = []
            if not isinstance(yolo_batch_results, (list, tuple)):
                yolo_batch_results = [yolo_batch_results]
            for idx, cam_id in enumerate(yolo_cam_ids):
                if idx < len(yolo_batch_results):
                    yolo_results_by_cam[cam_id] = yolo_batch_results[idx]

        # Run Lift depth forward once for all cameras, then reuse per-camera slices.
        depth_probs_by_cam: Dict[int, torch.Tensor] = {}
        if self.lift is not None and image_rgb_by_cam:
            lift_cam_ids = list(image_rgb_by_cam.keys())
            t_depth_forward = time.perf_counter()
            try:
                depth_probs_batch, _ = self.lift.run_inference_batch(
                    [image_rgb_by_cam[cid] for cid in lift_cam_ids]
                )
                for idx, cam_id in enumerate(lift_cam_ids):
                    depth_probs_by_cam[cam_id] = depth_probs_batch[idx:idx + 1]
            except Exception as exc:
                print(f"[Warning] Batched Lift inference failed, falling back per camera: {exc}")
            det_detail["depth_forward_ms"] += (time.perf_counter() - t_depth_forward) * 1000.0

        for cam_id in image_rgb_by_cam.keys():
            det_detail["cameras_processed"] += 1.0
            image_rgb = image_rgb_by_cam[cam_id]
            image_bgr = image_bgr_by_cam[cam_id]
            if frame_calibs is not None and cam_id in frame_calibs:
                K_cam, T_cam_world = frame_calibs[cam_id]
            else:
                K_cam = self.intrinsics[cam_id]
                T_cam_world = self.extrinsics[cam_id]

            # Get bounding boxes from YOLO
            candidate_dets = []
            yolo_result = yolo_results_by_cam.get(cam_id)
            boxes = getattr(yolo_result, "boxes", None) if yolo_result is not None else None
            if boxes is None or boxes.xyxy is None or int(boxes.xyxy.shape[0]) == 0:
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            if hasattr(boxes, "cls") and boxes.cls is not None:
                cls_arr = boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
            else:
                cls_arr = np.zeros((xyxy.shape[0],), dtype=np.int32)

            for i, box_coords in enumerate(xyxy):
                x1, y1, x2, y2 = box_coords
                conf = float(confs[i])

                if conf < self.det_min_conf:
                    continue

                bbox = (int(x1), int(y1), int(x2), int(y2))
                # Extract YOLO class ID for class-based motion model selection
                yolo_class_id = int(cls_arr[i])
                # Track class filtering
                if self.track_class_ids is not None and yolo_class_id not in self.track_class_ids:
                    continue
                candidate_dets.append({
                    "pixel_u": (x1 + x2) / 2.0,
                    "pixel_v": (y1 + y2) / 2.0,
                    "box_width": x2 - x1,
                    "box_height": y2 - y1,
                    "bbox": bbox,
                    "conf": conf,
                    "cam_id": cam_id,
                    "frame_idx": frame_idx,
                    "det_idx": int(i),
                    "det_label": f"{cam_id}:{i}",
                    "yolo_class": yolo_class_id,
                })

            # Get depth estimates
            depth_probs = None
            if self.lift is not None:
                depth_probs = depth_probs_by_cam.get(cam_id)
                if depth_probs is None:
                    t_depth_forward = time.perf_counter()
                    with torch.no_grad():
                        depth_probs, _ = self.lift.run_inference(image_rgb)
                    det_detail["depth_forward_ms"] += (time.perf_counter() - t_depth_forward) * 1000.0

            if not candidate_dets:
                continue

            # Estimate depth at detections
            H, W = image_rgb.shape[:2]
            cutoff_dets: List[Dict] = []
            if self.lift is not None:
                t_depth_est = time.perf_counter()
                candidate_dets = self.lift.estimate_depth_at_detections(
                    depth_probs,
                    candidate_dets,
                    image_hw=(H, W),
                    K=K_cam,
                    cutoff_dets=cutoff_dets,
                )
                det_detail["depth_estimation_ms"] += (time.perf_counter() - t_depth_est) * 1000.0
            else:
                continue

            R_cam = T_cam_world[:3, :3]
            t_cam = T_cam_world[:3, 3]
            t_bev_work = time.perf_counter()

            proj_dets: List[Dict] = []
            proj_u: List[float] = []
            proj_v: List[float] = []
            proj_depth: List[float] = []
            proj_sigma2_depth: List[float] = []
            fp_metas: List[Dict[str, float]] = []
            stored_depths: List[float] = []

            for det in candidate_dets:
                depth = det.get("depth")
                depth_var = det.get("depth_var")
                if depth is None or depth_var is None:
                    continue

                u_center = float(det["pixel_u"])
                v_center = float(det["pixel_v"])
                bbox = det.get("bbox", {})

                u_fp, v_fp, depth_mean, sigma2_depth, fp_meta = self._footpoint_fused_depth_from_bbox(
                    det=det,
                    bbox=bbox,
                    K_cam=K_cam,
                    R_cam=R_cam,
                    t_cam=t_cam,
                )

                if u_fp is None or v_fp is None or depth_mean is None or sigma2_depth is None:
                    if isinstance(bbox, dict):
                        bx1 = bbox.get("x1", -1)
                        by1 = bbox.get("y1", -1)
                        bx2 = bbox.get("x2", -1)
                        by2 = bbox.get("y2", -1)
                    else:
                        bx1, by1, bx2, by2 = (-1, -1, -1, -1)
                        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                            bx1, by1, bx2, by2 = bbox[:4]
                    print(
                        "[Footpoint Fail] "
                        f"frame={frame_idx} cam={cam_id} "
                        f"bbox=({float(bx1):.1f},{float(by1):.1f},"
                        f"{float(bx2):.1f},{float(by2):.1f}) "
                        "falling back to depth projection"
                    )

                    depth_signed = -float(depth) if self.dataset == "multiviewx" else float(depth)
                    proj_u.append(u_center)
                    proj_v.append(v_center)
                    proj_depth.append(float(depth_signed))
                    proj_sigma2_depth.append(float(depth_var))
                    fp_metas.append({})

                    stored_depths.append(float(depth_signed))
                else:
                    proj_u.append(float(u_fp))
                    proj_v.append(float(v_fp))
                    proj_depth.append(float(depth_mean))
                    proj_sigma2_depth.append(float(sigma2_depth))
                    fp_metas.append(fp_meta if fp_meta is not None else {})
                    stored_depths.append(float(depth))

                proj_dets.append(det)

            if not proj_dets:
                continue

            # project depth estimate to bev space
            z_bev_arr, R_bev_arr, R_pose_cam = bev_project(
                np.asarray(proj_u, dtype=float),
                np.asarray(proj_v, dtype=float),
                np.asarray(proj_depth, dtype=float),
                np.asarray(proj_sigma2_depth, dtype=float),
                K_cam,
                R_cam,
                t_cam,
                R_pose=self.r_pose_bev,
                min_variance=self.bev_min_variance,
            )
            if z_bev_arr is None or R_bev_arr is None:
                continue

            raw_proxy_proj_targets: List[Tuple[Dict, float, float, float, float, float]] = []

            for i, det in enumerate(proj_dets):
                fp_meta = fp_metas[i] or {}
                bbox = det.get("bbox", {})
                z_bev = z_bev_arr[i]
                R_bev = R_bev_arr[i]
                R_common_bev = self.r_pose_bev.copy()

                if fp_meta.get("lift_disagree"):
                    if isinstance(bbox, dict):
                        bx1 = bbox.get("x1", -1)
                        by1 = bbox.get("y1", -1)
                        bx2 = bbox.get("x2", -1)
                        by2 = bbox.get("y2", -1)
                    elif isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                        bx1, by1, bx2, by2 = bbox[:4]
                    else:
                        bx1 = by1 = bx2 = by2 = -1
                    print(
                        "[Footpoint/Lift Disagree] "
                        f"frame={frame_idx} cam={cam_id} "
                        f"uv=({fp_meta.get('u_foot', -1):.1f},{fp_meta.get('v_foot', -1):.1f}) "
                        f"z_bev=({float(z_bev[0]):.2f},{float(z_bev[1]):.2f}) "
                        f"bbox=({float(bx1):.1f},{float(by1):.1f},{float(bx2):.1f},{float(by2):.1f})"
                    )

                feature_dim = getattr(self.semantic_encoder, "feature_dim", 128) if self.semantic_encoder else 128
                empty_desc = np.empty((0, 32), dtype=np.uint8)

                detection = {
                    "cam_id": cam_id,
                    "bbox": det["bbox"],
                    "conf": det["conf"],
                    "depth": float(stored_depths[i]),
                    "depth_var": float(det.get("depth_var")),
                    "depth_cutoff": bool(det.get("depth_cutoff", False)),
                    "det_label": det.get("det_label"),
                    "z_bev": np.asarray(z_bev, dtype=np.float32),
                    "R_bev": np.asarray(R_bev, dtype=np.float32),
                    "R_common_bev": np.asarray(R_common_bev, dtype=np.float32),
                    "footpoint_depth": float(fp_meta.get("depth_fp", np.nan)) if fp_meta else None,
                    "footpoint_depth_var": float(fp_meta.get("sigma2_depth", np.nan)) if fp_meta else None,
                    "vec": np.zeros(feature_dim, dtype=np.float32),
                    "kp": [],
                    "desc": empty_desc,
                    "yolo_class": det.get("yolo_class", 0),
                }

                # Raw proxy depth baseline (bbox height + intrinsics), no Lift.
                raw_proxy_depth_m = self._raw_proxy_depth_from_box(
                    box_height_px=float(det.get("box_height", 0.0)),
                    K_cam=K_cam,
                    yolo_class=det.get("yolo_class", None),
                )
                if raw_proxy_depth_m is not None:
                    raw_proxy_depth_signed = (
                        -float(raw_proxy_depth_m) if self.dataset == "multiviewx" else float(raw_proxy_depth_m)
                    )
                    sigma_scale = float(getattr(self.lift, "depth_prior_sigma_scale", 0.1) or 0.1)
                    raw_proxy_var = float(
                        max((sigma_scale * float(raw_proxy_depth_m)) ** 2, get_footpoint_min_depth_var())
                    )
                    raw_proxy_proj_targets.append(
                        (
                            detection,
                            float(det["pixel_u"]),
                            float(det["pixel_v"]),
                            float(raw_proxy_depth_signed),
                            float(raw_proxy_var),
                            float(raw_proxy_depth_m),
                        )
                    )

                if self.semantic_encoder is not None and isinstance(det.get("bbox"), (list, tuple, np.ndarray)):
                    bbox_arr = det.get("bbox")
                    if bbox_arr is not None and len(bbox_arr) >= 4:
                        global_semantic_batch_targets.append(
                            (
                                detection,
                                image_bgr,
                                (
                                    int(bbox_arr[0]),
                                    int(bbox_arr[1]),
                                    int(bbox_arr[2]),
                                    int(bbox_arr[3]),
                                ),
                            )
                        )

                all_detections.append(detection)

            bev_ms_cam = (time.perf_counter() - t_bev_work) * 1000.0
            det_detail["bev_projection_ms"] += max(0.0, bev_ms_cam)

        # Extract semantic features in one batched model call across all cameras.
        if self.semantic_encoder is not None and global_semantic_batch_targets:
            valid_crops: List[np.ndarray] = []
            valid_targets: List[Dict] = []
            for detection, image_bgr, bbox in global_semantic_batch_targets:
                img_h, img_w = image_bgr.shape[:2]
                x_min, y_min, x_max, y_max = bbox
                x_min = max(0, min(int(x_min), img_w - 1))
                x_max = max(0, min(int(x_max), img_w))
                y_min = max(0, min(int(y_min), img_h - 1))
                y_max = max(0, min(int(y_max), img_h))
                if x_max <= x_min or y_max <= y_min:
                    continue
                crop = image_bgr[y_min:y_max, x_min:x_max]
                if crop.size == 0:
                    continue
                valid_crops.append(crop)
                valid_targets.append(detection)

            if valid_crops:
                t_sem = time.perf_counter()
                try:
                    features = self.semantic_encoder.extract_features_batch(valid_crops)
                except Exception as e:
                    features = None
                    print(f"[Warning] Batched semantic feature extraction failed: {e}")
                sem_dt_ms = (time.perf_counter() - t_sem) * 1000.0
                det_detail["semantic_feature_ms"] += sem_dt_ms

                if isinstance(features, np.ndarray) and features.ndim == 2 and features.shape[0] > 0:
                    n_assign = min(len(valid_targets), int(features.shape[0]))
                    for idx in range(n_assign):
                        feat = np.asarray(features[idx], dtype=np.float32)
                        valid_targets[idx]["feature"] = feat
                        valid_targets[idx]["vec"] = feat

        elapsed_total_s = time.perf_counter() - stage_perf_start
        _finalize_detection_detail(all_detections, elapsed_total_s)
        self.timing_stats["detection_depth"].append(elapsed_total_s)
        return all_detections

    def run_clustering_stage(
        self,
        detections: List[Dict],
        frame_idx: Optional[int] = None,
        gt_data: Optional[Dict] = None,
    ) -> List[List[int]]:
        """
        Stage 2: Graph-based chi-squared probability clustering.

        Input: Detections from Stage 1
        Output: List of clusters, where each cluster is a list of detection indices
        """
        if not detections:
            return []

        t0 = time.time()

        from scipy.stats import chi2
        prob_thresh = float(chi2.sf(self.config.mahal_thresh, df=2))

        clustering_mode = self.pre_fusion_mode

        clusters, _ = graph_clustering(
            detections,
            prob_thresh=prob_thresh,
            sem_thresh=self.config.sem_thresh,
            allow_single_cam=self.config.allow_single_cam,
            mode=clustering_mode,
            max_euclidean_dist=self.clustering_max_euclidean_dist,
        )

        n_pre_filter = len(clusters)
        n_singletons = sum(1 for c in clusters if len(c) == 1)
        if n_singletons > 0 and not self._is_radar:
            clusters = [c for c in clusters if len(c) > 1]

        self.timing_stats["clustering"] = self.timing_stats.get("clustering", [])
        self.timing_stats["clustering"].append(time.time() - t0)
        return clusters

    def run_fusion_stage(
        self,
        clusters: List[List[int]],
        detections: List[Dict],
        existing_tracks: Dict[int, Dict] = None,
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[float],
        List[Optional[int]],
        List[Dict],
        List[str],
        List[np.ndarray],
        List[List[int]],
    ]:
        """
        Stage 3: Kalman fusion with identity-aware soft updates.

        Input:
            - clusters: List of detection index groups
            - detections: List of detection dicts with z_bev, R_bev
            - existing_tracks: Dict[track_id -> {pos, cov, weight}] from PHD filter

        Output: Tuple of (positions, covariances, confidences, identities, identity_probs_list, object_classes, pooled_features)
            - identity_probs_list: Full probability distribution for each cluster
            - object_classes: Majority YOLO class name per cluster (for class-based motion models)
            - pooled_features: Multi-view pooled appearance features per cluster
        """
        if not clusters:
            return [], [], [], [], [], [], [], []

        t0 = time.time()
        fusion_mode = self.pre_fusion_mode

        positions = []
        covariances = []
        confidences = []
        identities = []
        object_classes = []  # Class-based motion model support
        identity_probs_list = []  # NEW: Full identity distributions for PHD update

        # Track global identity counter (increments for new IDs)
        if not hasattr(self, '_global_identity_counter'):
            self._global_identity_counter = 0

        if existing_tracks is None:
            existing_tracks = {}


        # ============================================================
        # PHASE 1: Fuse all clusters (Kalman fusion, no identity assignment yet)
        # ============================================================
        det_confidences_per_cluster = []
        pooled_features = []  # Multi-view feature pooling

        for cluster_indices in clusters:
            pos, cov = kalman_fusion(
                cluster_indices,
                detections,
                mode=fusion_mode,
                tau_geo=self.config.tau_geo,
                tau_sem=self.config.tau_sem,
                lambda_kl=self.config.lambda_kl,
                alpha=self.config.alpha,
            )

            if self._is_radar:
                cluster_class_names = [detections[i].get("class_name", "radar_vehicle") for i in cluster_indices]
                obj_class = max(set(cluster_class_names), key=cluster_class_names.count)
            else:
                cluster_classes = [detections[i].get("yolo_class", 0) for i in cluster_indices]
                majority_class_id = max(set(cluster_classes), key=cluster_classes.count) if cluster_classes else 0
                obj_class = get_profile_for_yolo_class(majority_class_id).class_name

            det_confs = [detections[i].get("conf", 0.0) for i in cluster_indices]
            max_det_conf = max(det_confs) if det_confs else 0.5

            cluster_feats = []
            cluster_feat_weights = []
            for idx in cluster_indices:
                det = detections[idx]
                feat = det.get("vec")
                if feat is not None and np.linalg.norm(feat) > 1e-8:
                    cluster_feats.append(feat / (np.linalg.norm(feat) + 1e-8))
                    cluster_feat_weights.append(det.get("conf", 0.5))
            if cluster_feats:
                fw = np.array(cluster_feat_weights, dtype=np.float32)
                fw /= fw.sum() + 1e-8
                pooled = sum(wi * fi for wi, fi in zip(fw, cluster_feats))
                pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
            else:
                feature_dim = getattr(self.semantic_encoder, "feature_dim", 128) if self.semantic_encoder else 128
                pooled = np.zeros(feature_dim, dtype=np.float32)
            pooled_features.append(pooled)

            positions.append(pos)
            covariances.append(cov)
            object_classes.append(obj_class)
            det_confidences_per_cluster.append(max_det_conf)

        N = len(positions)

        # ============================================================
        # PHASE 2: Covariance-aware Hungarian 1:1 identity assignment
        # ============================================================
        track_ids = list(existing_tracks.keys())
        M = len(track_ids)
        NEW_ID_COST = float(self.new_id_cost)
        GATE = float(self.config.mahal_thresh)

        mahal_sq_matrix = np.full((N, M), np.inf, dtype=np.float64)
        for i in range(N):
            for j, tid in enumerate(track_ids):
                diff = positions[i] - existing_tracks[tid]["pos"]
                cov_sum = covariances[i] + existing_tracks[tid]["cov"]
                try:
                    mahal_sq_matrix[i, j] = float(diff @ np.linalg.solve(cov_sum, diff))
                except Exception:
                    mahal_sq_matrix[i, j] = np.inf

        if M > 0:
            BIG = 1e6 

            cost_matrix = np.full((N, M + N), BIG, dtype=np.float64)

            for i in range(N):
                for j in range(M):
                    if mahal_sq_matrix[i, j] < GATE:
                        cost_matrix[i, j] = mahal_sq_matrix[i, j]

            FEATURE_COST_WEIGHT = 2.0  
            if fusion_mode in ("joint", "semantic") and pooled_features:
                for i in range(N):
                    cluster_feat = pooled_features[i]
                    if np.linalg.norm(cluster_feat) < 1e-8:
                        continue
                    for j, tid in enumerate(track_ids):
                        track_feat = existing_tracks[tid].get("feature")
                        if track_feat is None or np.linalg.norm(track_feat) < 1e-8:
                            continue
                        cos_sim = float(np.dot(cluster_feat, track_feat) /
                                       (np.linalg.norm(cluster_feat) * np.linalg.norm(track_feat) + 1e-8))
                        if cost_matrix[i, j] < BIG:
                            cost_matrix[i, j] -= FEATURE_COST_WEIGHT * cos_sim

            for i in range(N):
                cost_matrix[i, M + i] = NEW_ID_COST

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            assigned_ids = [None] * N
            for row, col in zip(row_indices, col_indices):
                if col < M:
                    assigned_ids[row] = track_ids[col]
                else:
                    assigned_ids[row] = self._global_identity_counter
                    self._global_identity_counter += 1

            for i in range(N):
                if assigned_ids[i] is None:
                    assigned_ids[i] = self._global_identity_counter
                    self._global_identity_counter += 1

        else:
            assigned_ids = []
            for i in range(N):
                assigned_ids.append(self._global_identity_counter)
                self._global_identity_counter += 1

        # ============================================================
        # PHASE 3: Build identity probability distributions for PHD soft association
        # ============================================================

        existing_identities = {}
        for tid in track_ids:
            track_data = existing_tracks[tid]
            # Create detection-like dict for each track
            existing_identities[tid] = [{
                "z_bev": track_data["pos"],
                "R_bev": track_data["cov"],
                "feature": track_data.get("feature"),
            }]

        kl_confidences = []  

        for i in range(N):
            identity_probs = {}
            kl_conf = 1.0  

            if M > 0 and len(existing_identities) > 0:
                cluster_detection = {
                    "z_bev": positions[i],
                    "R_bev": covariances[i],
                    "feature": pooled_features[i] if pooled_features else None,
                }

                probs_array, kl_conf_result = self.matcher.match_detection_to_identities(
                    cluster_detection, existing_identities
                )

                if kl_conf_result is not None:
                    kl_conf = kl_conf_result

                if len(probs_array) > 0:
                    for j, tid in enumerate(track_ids):
                        if mahal_sq_matrix[i, j] < GATE:
                            identity_probs[tid] = float(probs_array[j])

                    total = sum(identity_probs.values()) + 1e-8
                    identity_probs = {k: v / total for k, v in identity_probs.items()}

            assigned_id = assigned_ids[i]
            max_det_conf = det_confidences_per_cluster[i]
            if assigned_id not in identity_probs:
                identity_probs[assigned_id] = max_det_conf

            total = sum(identity_probs.values())
            if total > 1.0:
                identity_probs = {k: v / total for k, v in identity_probs.items()}

            identities.append(assigned_id)
            confidences.append(max_det_conf)
            identity_probs_list.append(identity_probs)
            kl_confidences.append(kl_conf)

        if fusion_mode == "joint":
            KL_INFLATE_TRIGGER = 0.55
            KL_MIN_CONF = 0.30
            KL_MAX_INFLATION = 1.18
            for i in range(N):
                kl_conf = kl_confidences[i]
                if kl_conf < KL_INFLATE_TRIGGER:
                    kl_conf_safe = max(kl_conf, KL_MIN_CONF)
                    inflation = min(1.0 / kl_conf_safe, KL_MAX_INFLATION)
                    covariances[i] = covariances[i] * inflation

        self.timing_stats["fusion"].append(time.time() - t0)
        return positions, covariances, confidences, identities, identity_probs_list, object_classes, pooled_features, clusters

    @staticmethod
    def _extract_cluster_velocities(
        clusters: List[List[int]],
        detections: List[Dict],
    ) -> List[Optional[np.ndarray]]:
        """Extract per-cluster velocity from the highest-confidence detection."""
        velocities = []
        for cluster_indices in clusters:
            best_vel = None
            best_conf = -1.0
            for idx in cluster_indices:
                det = detections[idx]
                vel = det.get("velocity_bev")
                if vel is None:
                    vel = det.get("velocity")
                if vel is None:
                    continue
                conf = det.get("conf", 0.0)
                if conf > best_conf:
                    best_conf = conf
                    best_vel = np.array(vel, dtype=np.float32)[:2]
            velocities.append(best_vel)
        return velocities

    def run_tracking_stage(
        self,
        fused_positions: List[np.ndarray],
        fused_covariances: List[np.ndarray],
        confidences: List[float],
        identities: List[Optional[int]],
        identity_probs_list: List[Dict] = None,
        object_classes: List[str] = None,
        pooled_features: List[np.ndarray] = None,
        camera_positions: Optional[Dict[int, np.ndarray]] = None,
        velocities: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[Dict]:
        """
        Stage 4: GM-PHD-HMM multi-target tracking.

        Input:
            - fused_positions: BEV positions from fusion
            - fused_covariances: Position covariances
            - confidences: Birth weights
            - identities: Assigned identity IDs
            - identity_probs_list: FULL probability distributions over ALL existing track IDs
            - object_classes: Class names for class-based motion models (from fusion stage)
            - velocities: Optional per-target velocity [vx, vy]

        Output: List of tracks with persistent IDs
        """
        if not fused_positions:
            return []

        t0 = time.time()

        # Convert to PHD birth format with assigned identities, object classes, and pooled features
        if object_classes is None:
            object_classes = ["pedestrian"] * len(fused_positions)
        if pooled_features is None:
            pooled_features = [None] * len(fused_positions)
        if velocities is None:
            velocities = [None] * len(fused_positions)
        new_targets = [
            {
                "z_bev": pos,
                "R_bev": cov,
                "identity": identity,
                "confidence": conf,
                "object_class": obj_cls,
                "feature": feat,  # Multi-view pooled feature for re-ID
                "velocity": vel,  # [vx, vy] for birth init
            }
            for pos, cov, conf, identity, obj_cls, feat, vel in zip(
                fused_positions, fused_covariances, confidences, identities, object_classes, pooled_features, velocities
            )
        ]

        # Build identity_probs_list fallback.
        if identity_probs_list is None:
            identity_probs_list = [
                {identity: conf} if identity is not None else {}
                for identity, conf in zip(identities, confidences)
            ]

        # Build measurements list
        measurements = []
        for i, (pos, cov) in enumerate(zip(fused_positions, fused_covariances)):
            measurements.append({
                "z": pos,
                "R": cov,
                "identity_probs": identity_probs_list[i],
                "feature": pooled_features[i] if i < len(pooled_features) else None,
            })

        self.phd_filter.predict(new_targets=new_targets)

        # Pass camera positions for BEV line-of-sight occlusion model
        self.phd_filter.update(measurements, camera_positions=camera_positions)
        self.phd_filter.manage_components()

        tracks = self.phd_filter.extract_tracks()

        # Log tracking statistics
        n_components = len(self.phd_filter.components)
        n_tracks = len(tracks)
        expected_targets = self.phd_filter.expected_number_of_targets()

        self.timing_stats["tracking"].append(time.time() - t0)
        return tracks

    def evaluate_on_sequence(
        self,
        sequence_name: str = "sequence_001",
    ) -> Dict:
        """
        Evaluate full pipeline on the selected dataset.

        Returns:
            Results dict with MOT metrics and timing
        """
        print(f"\n[Evaluator] Running evaluation on {sequence_name}")

        frame_count = 0
        t_total_start = time.time()
        self.fusion_f1_frame_metrics.clear()
        self.per_frame_timing = []

        frame_start = 360 if self.dataset in ("wildtrack", "multiviewx") else 0
        frame_end = 400 if self.dataset in ("wildtrack", "multiviewx") else 0
        if frame_end > 0:
            print(f"  Evaluation frame window: [{frame_start}, {frame_end})")

        frame_idx = frame_start

        # Create iterators for each camera.
        camera_iters = {}
        for cam in self.camera_names:
            camera_iters[cam] = self.dataset_spec.iter_frames(
                str(self.dataset_root),
                camera=cam,
                target_hw=self.target_hw,
                every_n=1,
                frame_start=frame_start,
                frame_end=frame_end,
            )

        all_done = False
        while not all_done:
            images = {}
            calibs = {}
            gt_points_by_id: Dict[int, np.ndarray] = {}
            gt_points_by_cam: Dict[int, np.ndarray] = {}
            gt_data = None
            frame_token: Optional[str] = None

            # Collect frames from all cameras.
            for cam in self.camera_names:
                try:
                    frame_data = next(camera_iters[cam])
                    if len(frame_data) >= 11:
                        (
                            img_path,
                            rgb,
                            target_hw,
                            K_img,
                            T_cam_world,
                            T_world_cam,
                            world_pts,
                            boxes_scaled,
                            person_ids,
                        ) = frame_data[:9]
                    elif len(frame_data) == 9:
                        (
                            img_path,
                            rgb,
                            target_hw,
                            K_img,
                            T_cam_world,
                            T_world_cam,
                            world_pts,
                            boxes_scaled,
                            person_ids,
                        ) = frame_data
                    elif len(frame_data) >= 7:
                        (
                            rgb,
                            target_hw,
                            K_img,
                            T_cam_world,
                            T_world_cam,
                        ) = frame_data[:5]
                        img_path = None
                        world_pts = None
                        boxes_scaled = None
                        person_ids = None
                    else:
                        raise ValueError(f"Unexpected frame payload length: {len(frame_data)}")

                    cam_id = self.cam_name_to_idx[cam]
                    images[cam_id] = rgb
                    calibs[cam_id] = (K_img, T_cam_world)

                    if frame_token is None and img_path is not None:
                        frame_token = Path(img_path).stem
                    if world_pts is not None and person_ids is not None:
                        gt_points_by_cam[cam_id] = world_pts
                        for pid, pt in zip(person_ids, world_pts):
                            gt_points_by_id[int(pid)] = pt
                except StopIteration:
                    all_done = True
                    break

            if all_done or not images:
                break

            # Camera selection: use only specified camera indices.
            if self.active_camera_indices is not None:
                all_cam_ids = sorted(images.keys())
                active_cams = [c for c in all_cam_ids if c in self.active_camera_indices]
                if not active_cams:
                    active_cams = all_cam_ids[:1]
                if set(active_cams) != set(all_cam_ids):
                    dropped = [c for c in all_cam_ids if c not in active_cams]
                    images = {c: images[c] for c in active_cams}
                    calibs = {c: calibs[c] for c in active_cams}
                    gt_points_by_cam = {c: v for c, v in gt_points_by_cam.items() if c in active_cams}
                    if frame_count == 0:
                        print(f"  [Camera Select] Using cameras {active_cams}, dropped {dropped}")

            if self.dataset == "wildtrack":
                gt_points_by_id_all = self._load_wildtrack_all_gt_points(frame_idx, frame_token=frame_token)
                if gt_points_by_id_all:
                    gt_points_by_id = gt_points_by_id_all

            if gt_points_by_id:
                gt_ids = np.array(sorted(gt_points_by_id.keys()), dtype=np.int32)
                gt_world_pts = np.stack([gt_points_by_id[int(pid)] for pid in gt_ids], axis=0).astype(np.float32)
                gt_points_by_cam = {int(cam_id): gt_world_pts.copy() for cam_id in images.keys()}
                gt_data = {"world_pts": gt_world_pts, "person_ids": gt_ids}

            frame_count += 1
            frame_inference_t0 = time.perf_counter()
            frame_detection_ms = 0.0
            frame_clustering_ms = 0.0
            frame_fusion_ms = 0.0
            frame_tracking_ms = 0.0

            t_stage = time.perf_counter()
            detections = self.run_detection_stage(
                images,
                calibs,
                frame_idx,
            )
            frame_detection_ms = (time.perf_counter() - t_stage) * 1000.0
            det_detail_frame = dict(self._last_detection_depth_detail) if self._last_detection_depth_detail else {}
            if det_detail_frame:
                frame_detection_ms = float(det_detail_frame.get("total_ms", frame_detection_ms))

            HIGH_CONF_THRESHOLD = self.high_conf_thresh
            LOW_CONF_MIN = self.low_conf_min

            high_conf_dets = [d for d in detections if d.get("conf", 0) >= HIGH_CONF_THRESHOLD]
            low_conf_dets = [d for d in detections if LOW_CONF_MIN <= d.get("conf", 0) < HIGH_CONF_THRESHOLD]

            existing_tracks = self.phd_filter.get_existing_track_info(weight_threshold=0.06)

            camera_positions_bev: Dict[int, np.ndarray] = {}
            for cam_id, (K_cam, T_cam_world) in calibs.items():
                R_cam = T_cam_world[:3, :3]
                t_cam = T_cam_world[:3, 3]
                cam_world = -R_cam.T @ t_cam
                camera_positions_bev[cam_id] = cam_world[:2].astype(np.float32)

            # Stage 1: high-confidence.
            t_stage = time.perf_counter()
            clusters_high = self.run_clustering_stage(high_conf_dets, frame_idx, gt_data=gt_data)
            frame_clustering_ms += (time.perf_counter() - t_stage) * 1000.0
            t_stage = time.perf_counter()
            pos_h, cov_h, conf_h, ids_h, probs_h, cls_h, feat_h, clusters_high_used = self.run_fusion_stage(
                clusters_high, high_conf_dets, existing_tracks
            )
            frame_fusion_ms += (time.perf_counter() - t_stage) * 1000.0

            matched_track_ids = set(ids_h) & set(existing_tracks.keys())
            unmatched_tracks = {k: v for k, v in existing_tracks.items() if k not in matched_track_ids}

            # Stage 2: low-confidence against unmatched tracks.
            pos_l, cov_l, conf_l, ids_l, probs_l, cls_l, feat_l = [], [], [], [], [], [], []
            if low_conf_dets and unmatched_tracks:
                t_stage = time.perf_counter()
                clusters_low = self.run_clustering_stage(low_conf_dets, frame_idx, gt_data=gt_data)
                frame_clustering_ms += (time.perf_counter() - t_stage) * 1000.0
                if clusters_low:
                    t_stage = time.perf_counter()
                    pos_l, cov_l, conf_l, ids_l, probs_l, cls_l, feat_l, clusters_low_used = self.run_fusion_stage(
                        clusters_low, low_conf_dets, unmatched_tracks
                    )
                    frame_fusion_ms += (time.perf_counter() - t_stage) * 1000.0
                else:
                    clusters_low_used = []
            else:
                clusters_low_used = []

            fused_positions = pos_h + pos_l
            fused_covariances = cov_h + cov_l
            confidences = conf_h + conf_l
            identities = ids_h + ids_l
            identity_probs_list = probs_h + probs_l
            object_classes = cls_h + cls_l
            all_pooled_features = feat_h + feat_l

            vel_h = self._extract_cluster_velocities(clusters_high_used, high_conf_dets)
            vel_l = self._extract_cluster_velocities(clusters_low_used, low_conf_dets) if clusters_low_used else []
            all_velocities = vel_h + vel_l

            t_stage = time.perf_counter()
            tracks = self.run_tracking_stage(
                fused_positions,
                fused_covariances,
                confidences,
                identities,
                identity_probs_list,
                object_classes,
                pooled_features=all_pooled_features,
                camera_positions=camera_positions_bev,
                velocities=all_velocities,
            )
            frame_tracking_ms = (time.perf_counter() - t_stage) * 1000.0
            frame_total_ms = (time.perf_counter() - frame_inference_t0) * 1000.0

            self.per_frame_timing.append(
                {
                    "frame_idx": int(frame_idx),
                    "detection_depth_ms": float(frame_detection_ms),
                    "detection_detector_forward_ms": float(det_detail_frame.get("detector_forward_ms", 0.0)),
                    "detection_depth_forward_ms": float(det_detail_frame.get("depth_forward_ms", 0.0)),
                    "detection_depth_estimation_ms": float(det_detail_frame.get("depth_estimation_ms", 0.0)),
                    "detection_semantic_feature_ms": float(det_detail_frame.get("semantic_feature_ms", 0.0)),
                    "detection_bev_projection_ms": float(det_detail_frame.get("bev_projection_ms", 0.0)),
                    "detection_other_ms": float(det_detail_frame.get("detection_other_ms", 0.0)),
                    "detection_cameras_processed": float(det_detail_frame.get("cameras_processed", 0.0)),
                    "detection_detections_out": float(det_detail_frame.get("detections_out", 0.0)),
                    "clustering_ms": float(frame_clustering_ms),
                    "fusion_ms": float(frame_fusion_ms),
                    "tracking_ms": float(frame_tracking_ms),
                    "inference_total_ms": float(frame_total_ms),
                    "n_detections": int(len(detections)),
                    "n_tracks": int(len(tracks)),
                }
            )

            if gt_data:
                predicted_tracks = {track["identity"]: track["position"] for track in tracks}
                self.mot_evaluator.update(
                    predicted_tracks=predicted_tracks,
                    ground_truth=gt_data,
                    frame_idx=frame_idx,
                )

            if frame_count % 10 == 0:
                print(f"  [Progress] Processed {frame_count} frames")

            frame_idx += 1

        t_total = time.time() - t_total_start

        mot_metrics = self.mot_evaluator.compute_metrics()

        mot_pymot_evaluator = PyMotMetricsEvaluator(
            max_distance=self.max_distance_mot,
        )
        mot_pymot_evaluator.predictions = defaultdict(
            list, {f: list(rows) for f, rows in self.mot_evaluator.predictions.items()}
        )
        mot_pymot_evaluator.ground_truth = defaultdict(
            list, {f: list(rows) for f, rows in self.mot_evaluator.ground_truth.items()}
        )
        mot_metrics_pymotmetrics = mot_pymot_evaluator.compute_metrics()

        gospa_evaluator = GOSPAEvaluator(
            max_distance=self.max_distance_mot,
        )
        gospa_evaluator.predictions = defaultdict(
            list, {f: list(rows) for f, rows in self.mot_evaluator.predictions.items()}
        )
        gospa_evaluator.ground_truth = defaultdict(
            list, {f: list(rows) for f, rows in self.mot_evaluator.ground_truth.items()}
        )
        gospa_metrics = gospa_evaluator.compute_metrics()

        calibration_stats = {}

        timing_summary = self._build_timing_summary(t_total, frame_count)

        if torch.cuda.is_available():
            timing_summary["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            timing_summary["gpu_memory_current_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)

        results = {
            "sequence": sequence_name,
            "mode": self.matching_mode,
            "mot_metrics": mot_metrics,
            "mot_metrics_pymotmetrics": mot_metrics_pymotmetrics,
            "gospa_metrics": gospa_metrics,
            "timing": timing_summary,
            "calibration": calibration_stats,
        }
        if self.fusion_f1_frame_metrics:
            precisions = [m["precision"] for m in self.fusion_f1_frame_metrics]
            recalls = [m["recall"] for m in self.fusion_f1_frame_metrics]
            f1s = [m["f1"] for m in self.fusion_f1_frame_metrics]
            results["fusion_f1"] = {
                "association_threshold_m": float(self.fusion_assoc_threshold_m),
                "num_frames": int(len(self.fusion_f1_frame_metrics)),
                "avg_precision": float(np.mean(precisions)),
                "avg_recall": float(np.mean(recalls)),
                "avg_f1": float(np.mean(f1s)),
            }

        return results
    def evaluate_radarscenes_benchmark(
        self,
        max_scenes: int = 0,
        dt_frame: float = 0.2,
        dbscan_eps: float = 2.5,
        min_rcs: float = -30.0,
        min_doppler: float = 0.5,
        oracle: bool = False,
        mos_model: Optional[str] = None,
        mos_threshold: float = 0.5,
        merge_dist: float = 6.0,
        merge_gap: int = 5,
        prob_thresh: float = 0.01,
        assign_tentative: bool = True,
        detection_prob: float = 0.80,
        birth_conf: float = 0.65,
    ) -> Dict:
        """Run multi-scene RadarScenes tracking benchmark with LSTQ evaluation.

        Returns:
            Results dict with per-scene LSTQ and aggregate metrics.
        """
        from modtrack.data.adapters.radarscenes import (
            iter_radarscenes_scenes_lstq,
            load_mos_classifier,
            set_mos_threshold,
            MOVING_CLASSES,
        )
        from modtrack.core.tracker.motion_models import get_profile_by_name
        from modtrack.core.tracker.gm_phd_hmm import GMPHDComponent, TrackState

        # Load MOS classifier if provided
        if mos_model is not None:
            load_mos_classifier(mos_model)
            set_mos_threshold(mos_threshold)

        print(f"\n{'='*60}")
        print(f"RadarScenes Benchmark (evaluate_modtrack.py, spatial-only, LSTQ)")
        print(f"  dt_frame={dt_frame}s, dbscan_eps={dbscan_eps}, "
              f"min_rcs={min_rcs}, min_doppler={min_doppler}")
        print(f"  merge_dist={merge_dist}, merge_gap={merge_gap}")
        print(f"  oracle={oracle}, mos_model={mos_model}")
        print(f"{'='*60}\n")

        # Class-aware birth thresholds (same as runner)
        CLASS_BIRTH_CONF = {
            "radar_vehicle": 0.50, "truck": 0.50,
            "motorcycle": 0.60, "pedestrian": 0.80,
        }

        all_results = []
        total_frames = 0
        t_benchmark_start = time.time()

        for scene_idx, (scene_id, scene_name, frame_iter, raw_data) in enumerate(
            iter_radarscenes_scenes_lstq(
                str(self.dataset_root),
                dt_frame=dt_frame,
                max_scenes=max_scenes,
                min_doppler=min_doppler,
                dbscan_eps=dbscan_eps,
                min_rcs=min_rcs,
                use_classifier=(mos_model is not None),
                oracle=oracle,
            )
        ):
            # ---- Per-scene setup ----
            # Re-create PHD filter for each scene (fresh state)
            self.phd_filter = IdentityInformedGMPHDFilter(
                dt=dt_frame,
                process_noise_scale=self.phd_process_noise_scale,
                birth_threshold_scale=1.0,
                clutter_intensity=1.0,
                detection_prob=detection_prob,
                mahal_thresh=self.config.mahal_thresh,
                dataset="radarscenes",
                confirmed_miss_tolerance=self.phd_filter.confirmed_miss_tolerance,
                max_lost_frames=self.phd_filter.max_lost_frames,
                tentative_miss_tolerance=self.phd_filter.tentative_miss_tolerance,
                use_class_birth_mode_priors=True,
            )
            if not hasattr(self, '_global_identity_counter'):
                self._global_identity_counter = 0
            self._global_identity_counter = 0

            gt_label_id = raw_data["label_id"]
            gt_track_id = raw_data["track_id"]
            frame_to_all_points = raw_data["frame_to_all_points"]

            n_frames = 0
            next_synth_id = 1
            t_scene_start = time.time()

            # Accumulate per-frame data for tube merging + LSTQ
            frame_data_list = []
            frame_positions: Dict[int, Dict[int, np.ndarray]] = {}

            for frame in frame_iter:
                frame_idx = frame["frame_idx"]
                detections = frame["detections"]

                all_pt_indices = frame_to_all_points.get(frame_idx, [])
                if not all_pt_indices:
                    continue

                n_frames += 1

                # Build GT arrays for this frame
                n_pts = len(all_pt_indices)
                gt_sem = np.zeros(n_pts, dtype=np.int32)
                gt_inst = np.zeros(n_pts, dtype=np.int64)
                pred_sem = np.zeros(n_pts, dtype=np.int32)
                pred_inst = np.zeros(n_pts, dtype=np.int64)

                pt_to_local = {}
                for local_idx, raw_idx in enumerate(all_pt_indices):
                    pt_to_local[raw_idx] = local_idx
                    lid = int(gt_label_id[raw_idx])
                    gt_sem[local_idx] = LSTQ_CLASS_MOVING if lid in MOVING_CLASSES else LSTQ_CLASS_STATIC
                    gt_inst[local_idx] = int(gt_track_id[raw_idx])

                if not detections:
                    self.phd_filter.predict(new_targets=[])
                    self.phd_filter.update([])
                    self.phd_filter.manage_components()
                    frame_data_list.append((frame_idx, all_pt_indices, pred_sem, pred_inst, gt_sem, gt_inst))
                    continue

                # ==== Stage 1: Clustering (SAME as WildTrack/MultiviewX) ====
                clusters = self.run_clustering_stage(detections, frame_idx)

                if not clusters:
                    self.phd_filter.predict(new_targets=[])
                    self.phd_filter.update([])
                    self.phd_filter.manage_components()
                    frame_data_list.append((frame_idx, all_pt_indices, pred_sem, pred_inst, gt_sem, gt_inst))
                    continue

                # ==== Stage 2: Fusion (SAME as WildTrack/MultiviewX) ====
                existing_tracks = self.phd_filter.get_existing_track_info(weight_threshold=0.06)
                (fused_positions, fused_covariances, confidences,
                 identities, identity_probs_list, object_classes,
                 pooled_features, clusters_used) = self.run_fusion_stage(
                    clusters, detections, existing_tracks
                )

                # Extract velocities from radar detections
                fused_velocities = []
                for cl_indices in clusters_used:
                    vels = []
                    weights = []
                    for i in cl_indices:
                        v = detections[i].get("velocity_bev")
                        if v is not None:
                            vels.append(v)
                            weights.append(detections[i]["conf"])
                    if vels:
                        w_arr = np.array(weights)
                        w_arr /= w_arr.sum()
                        fused_vel = np.sum([w * v for w, v in zip(w_arr, vels)], axis=0)
                        fused_velocities.append(fused_vel)
                    else:
                        fused_velocities.append(None)

                # ==== Stage 3: Predict (deferred birth — predict with NO births) ====
                self.phd_filter.predict(new_targets=[])

                # ==== Stage 4: PHD Update ====
                measurements = [
                    {"z": pos, "R": cov, "identity_probs": id_probs, "feature": None}
                    for pos, cov, id_probs in zip(
                        fused_positions, fused_covariances, identity_probs_list
                    )
                ]
                self.phd_filter.update(measurements)
                self.phd_filter.manage_components()

                # ==== Stage 5: Deferred birth ====
                n_births = 0
                for pos, cov, conf, cls, vel in zip(
                    fused_positions, fused_covariances, confidences,
                    object_classes, fused_velocities,
                ):
                    cls_birth = CLASS_BIRTH_CONF.get(cls, birth_conf)
                    if conf < cls_birth:
                        continue

                    suppress = False
                    for comp in self.phd_filter.components:
                        if comp.weight < 0.05:
                            continue
                        delta = pos - comp.mean[:2]
                        P_pos = comp.covariance[:2, :2] + cov
                        try:
                            d2 = float(delta @ np.linalg.solve(P_pos, delta))
                        except np.linalg.LinAlgError:
                            d2 = float(np.sum(delta ** 2))
                        if d2 < self.config.mahal_thresh:
                            suppress = True
                            break
                    if suppress:
                        continue

                    synth_id = next_synth_id
                    next_synth_id += 1
                    profile = get_profile_by_name(cls)

                    speed = float(np.linalg.norm(vel)) if vel is not None else 0.0
                    moving = speed > 1.0

                    if moving and vel is not None:
                        vx, vy = float(vel[0]), float(vel[1])
                        birth_state = np.array([pos[0], pos[1], vx, vy], dtype=np.float32)
                        vel_std = max(2.0, speed * 0.3)
                    else:
                        birth_state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float32)
                        vel_std = profile.initial_velocity_std

                    birth_cov = np.eye(4, dtype=np.float32)
                    birth_cov[:2, :2] = cov
                    birth_cov[2:, 2:] = vel_std ** 2 * np.eye(2)

                    tm = profile.transition_matrix
                    n_modes = len(profile.modes)
                    mode_weights = np.array([tm[0, m] for m in range(n_modes)], dtype=np.float32)
                    mode_weights /= mode_weights.sum()
                    for mode in range(n_modes):
                        self.phd_filter.components.append(GMPHDComponent(
                            weight=conf * float(mode_weights[mode]),
                            mean=birth_state.copy(),
                            covariance=birth_cov.copy(),
                            identity=synth_id,
                            motion_mode=mode,
                            object_class=profile.class_name,
                            feature=None,
                            track_state=TrackState.TENTATIVE,
                            hit_count=0,
                            miss_count=0,
                        ))
                    n_births += 1

                # ==== Stage 6: Propagate track IDs to points ====
                comp_list = [(comp.mean[:2].copy(), comp.identity)
                             for comp in self.phd_filter.components
                             if comp.identity is not None and comp.weight > 0.01
                             and (assign_tentative or comp.track_state == TrackState.CONFIRMED)]

                comp_positions = {}
                for comp in self.phd_filter.components:
                    if (comp.identity is not None and comp.weight > 0.01
                            and (assign_tentative or comp.track_state == TrackState.CONFIRMED)):
                        comp_positions[comp.identity] = comp.mean[:2].copy()
                frame_positions[n_frames] = comp_positions

                if comp_list:
                    comp_pos_arr = np.array([c[0] for c in comp_list])
                    comp_id_arr = [c[1] for c in comp_list]

                    for det in detections:
                        raw_indices = det.get("raw_point_indices", [])
                        det_pos = det["z_bev"]
                        dists = np.linalg.norm(comp_pos_arr - det_pos, axis=1)
                        best_idx = np.argmin(dists)
                        track_id = comp_id_arr[best_idx] if dists[best_idx] <= 5.0 else 0

                        for raw_idx in raw_indices:
                            local_idx = pt_to_local.get(raw_idx)
                            if local_idx is None:
                                continue
                            pred_sem[local_idx] = LSTQ_CLASS_MOVING
                            if track_id > 0:
                                pred_inst[local_idx] = track_id

                frame_data_list.append((frame_idx, all_pt_indices, pred_sem, pred_inst, gt_sem, gt_inst))

            evaluated_frames = {fd[0] for fd in frame_data_list}
            for fi, pt_indices in sorted(frame_to_all_points.items()):
                if fi in evaluated_frames:
                    continue
                n_pts_extra = len(pt_indices)
                if n_pts_extra == 0:
                    continue
                gt_sem_extra = np.zeros(n_pts_extra, dtype=np.int32)
                gt_inst_extra = np.zeros(n_pts_extra, dtype=np.int64)
                for local_idx, raw_idx in enumerate(pt_indices):
                    lid = int(gt_label_id[raw_idx])
                    gt_sem_extra[local_idx] = LSTQ_CLASS_MOVING if lid in MOVING_CLASSES else LSTQ_CLASS_STATIC
                    gt_inst_extra[local_idx] = int(gt_track_id[raw_idx])
                pred_sem_extra = np.zeros(n_pts_extra, dtype=np.int32)
                pred_inst_extra = np.zeros(n_pts_extra, dtype=np.int64)
                frame_data_list.append((fi, pt_indices, pred_sem_extra, pred_inst_extra,
                                       gt_sem_extra, gt_inst_extra))

            id_map = self._merge_tubes_radar([], frame_positions, merge_dist=merge_dist, merge_gap=merge_gap)

            evaluator = LSTQEvaluator(min_tube_points=1)
            for (fi, all_pt_indices, pred_sem, pred_inst, gt_sem, gt_inst) in frame_data_list:
                if id_map:
                    merged_inst = pred_inst.copy()
                    for i in range(len(merged_inst)):
                        old_id = int(merged_inst[i])
                        if old_id > 0 and old_id in id_map:
                            new_id = old_id
                            while new_id in id_map:
                                new_id = id_map[new_id]
                            merged_inst[i] = new_id
                    evaluator.add_frame(pred_sem, merged_inst, gt_sem, gt_inst)
                else:
                    evaluator.add_frame(pred_sem, pred_inst, gt_sem, gt_inst)

            elapsed = time.time() - t_scene_start
            lstq_metrics = evaluator.compute()

            result = {
                "scene": scene_name,
                "n_frames": n_frames,
                "elapsed_s": round(elapsed, 2),
                **lstq_metrics,
            }
            all_results.append(result)
            total_frames += n_frames

            n_merged = len(id_map)
            print(f"  [{scene_name}] {n_frames}f {elapsed:.1f}s | "
                  f"LSTQ={lstq_metrics['LSTQ']:.1f} "
                  f"S_cls={lstq_metrics['S_cls']:.1f} "
                  f"S_assoc={lstq_metrics['S_assoc']:.1f} "
                  f"tubes={lstq_metrics['n_pred_tubes']}/{lstq_metrics['n_gt_tubes']} "
                  f"merge={n_merged}")

        t_benchmark = time.time() - t_benchmark_start
        if all_results:
            avg_lstq = np.mean([r["LSTQ"] for r in all_results])
            avg_s_cls = np.mean([r["S_cls"] for r in all_results])
            avg_s_assoc = np.mean([r["S_assoc"] for r in all_results])
        else:
            avg_lstq = avg_s_cls = avg_s_assoc = 0.0

        print(f"\n{'='*60}")
        print(f"AGGREGATE ({len(all_results)} scenes, {total_frames} frames)")
        print(f"  LSTQ={avg_lstq:.2f}  S_cls={avg_s_cls:.2f}  S_assoc={avg_s_assoc:.2f}")
        print(f"  Total time: {t_benchmark:.1f}s ({total_frames / t_benchmark:.1f} fps)")
        print(f"{'='*60}\n")

        summary = {
            "n_scenes": len(all_results),
            "total_frames": total_frames,
            "dt_frame": dt_frame,
            "dbscan_eps": dbscan_eps,
            "LSTQ": round(avg_lstq, 2),
            "S_cls": round(avg_s_cls, 2),
            "S_assoc": round(avg_s_assoc, 2),
            "total_time_s": round(t_benchmark, 2),
            "fps": round(total_frames / t_benchmark, 2) if t_benchmark > 0 else 0,
            "per_scene": all_results,
            "timing": {
                "total_time_s": round(t_benchmark, 2),
                "fps": round(total_frames / t_benchmark, 2) if t_benchmark > 0 else 0,
                "frames_processed": total_frames,
            },
        }
        return summary

    def _merge_tubes_radar(
        self,
        frame_assignments,
        frame_positions: Dict[int, Dict[int, np.ndarray]],
        merge_dist: float = 6.0,
        merge_gap: int = 5,
    ) -> Dict[int, int]:
        """Post-hoc velocity-aware tube merging for RadarScenes.

        Velocity-aware merge for temporally adjacent tube fragments.
        """
        tube_first_frame: Dict[int, int] = {}
        tube_last_frame: Dict[int, int] = {}
        tube_first_pos: Dict[int, np.ndarray] = {}
        tube_last_pos: Dict[int, np.ndarray] = {}
        tube_velocity: Dict[int, np.ndarray] = {}

        for fi, positions in sorted(frame_positions.items()):
            for tid, pos in positions.items():
                if tid not in tube_first_frame:
                    tube_first_frame[tid] = fi
                    tube_first_pos[tid] = pos
                else:
                    prev_pos = tube_last_pos[tid]
                    prev_frame = tube_last_frame[tid]
                    df = fi - prev_frame
                    if df > 0:
                        tube_velocity[tid] = (pos - prev_pos) / df
                tube_last_frame[tid] = fi
                tube_last_pos[tid] = pos

        tube_ids = sorted(tube_first_frame.keys(), key=lambda t: tube_first_frame[t])
        id_map: Dict[int, int] = {}

        def canonical(tid: int) -> int:
            seen = set()
            while tid in id_map and tid not in seen:
                seen.add(tid)
                tid = id_map[tid]
            return tid

        for i, tid in enumerate(tube_ids):
            best_merge = None
            best_score = merge_dist
            start_frame = tube_first_frame[tid]
            start_pos = tube_first_pos[tid]

            for j in range(i - 1, -1, -1):
                other = tube_ids[j]
                canon = canonical(other)
                other_end = tube_last_frame[canon]
                gap = start_frame - other_end
                if gap < 0:
                    continue
                if gap > merge_gap:
                    if tube_first_frame[other] < start_frame - merge_gap * 2:
                        break
                    continue

                other_pos = tube_last_pos[canon]
                vel = tube_velocity.get(canon, np.zeros(2))
                predicted_pos = other_pos + vel * gap
                d = float(np.linalg.norm(start_pos - predicted_pos))

                if d < best_score:
                    best_score = d
                    best_merge = canon

            if best_merge is not None:
                id_map[tid] = best_merge
                tube_last_frame[best_merge] = max(
                    tube_last_frame.get(best_merge, 0), tube_last_frame[tid])
                tube_last_pos[best_merge] = tube_last_pos[tid]
                if tid in tube_velocity:
                    tube_velocity[best_merge] = tube_velocity[tid]

        return id_map

    def _build_timing_summary(self, total_time_s: float, frame_count: int) -> Dict:
        """Build aggregate + per-frame timing summary."""
        if self.per_frame_timing:
            skip_n = max(0, int(getattr(self, "timing_warmup_skip_frames", 0)))
            timing_rows = self.per_frame_timing[skip_n:] if len(self.per_frame_timing) > skip_n else self.per_frame_timing

            det_vals = np.asarray(
                [row.get("detection_depth_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            clus_vals = np.asarray(
                [row.get("clustering_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            fus_vals = np.asarray(
                [row.get("fusion_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            track_vals = np.asarray(
                [row.get("tracking_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            infer_vals = np.asarray(
                [row.get("inference_total_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            det_forward_vals = np.asarray(
                [row.get("detection_detector_forward_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            depth_forward_vals = np.asarray(
                [row.get("detection_depth_forward_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            depth_est_vals = np.asarray(
                [row.get("detection_depth_estimation_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            sem_vals = np.asarray(
                [row.get("detection_semantic_feature_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            bev_vals = np.asarray(
                [row.get("detection_bev_projection_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            det_other_vals = np.asarray(
                [row.get("detection_other_ms", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            det_cam_vals = np.asarray(
                [row.get("detection_cameras_processed", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            det_out_vals = np.asarray(
                [row.get("detection_detections_out", 0.0) for row in timing_rows],
                dtype=np.float64,
            )
            timing_total_s = float(np.sum(infer_vals) / 1000.0) if infer_vals.size else 0.0
            timing_frame_count = int(len(timing_rows))
            return {
                "detection_depth_ms": float(np.mean(det_vals)) if det_vals.size else 0.0,
                "detection_detector_forward_ms": float(np.mean(det_forward_vals)) if det_forward_vals.size else 0.0,
                "detection_depth_forward_ms": float(np.mean(depth_forward_vals)) if depth_forward_vals.size else 0.0,
                "detection_depth_estimation_ms": float(np.mean(depth_est_vals)) if depth_est_vals.size else 0.0,
                "detection_semantic_feature_ms": float(np.mean(sem_vals)) if sem_vals.size else 0.0,
                "detection_bev_projection_ms": float(np.mean(bev_vals)) if bev_vals.size else 0.0,
                "detection_other_ms": float(np.mean(det_other_vals)) if det_other_vals.size else 0.0,
                "detection_cameras_processed": float(np.mean(det_cam_vals)) if det_cam_vals.size else 0.0,
                "detection_detections_out": float(np.mean(det_out_vals)) if det_out_vals.size else 0.0,
                "clustering_ms": float(np.mean(clus_vals)) if clus_vals.size else 0.0,
                "fusion_ms": float(np.mean(fus_vals)) if fus_vals.size else 0.0,
                "tracking_ms": float(np.mean(track_vals)) if track_vals.size else 0.0,
                "inference_total_ms": float(np.mean(infer_vals)) if infer_vals.size else 0.0,
                "inference_total_std_ms": float(np.std(infer_vals)) if infer_vals.size else 0.0,
                "total_time_s": float(total_time_s),
                "fps": timing_frame_count / timing_total_s if timing_total_s > 0 else 0.0,
                "frames_processed": int(frame_count),
                "timing_frames_used": timing_frame_count,
                "timing_warmup_skip_frames": int(skip_n),
                "per_frame": list(self.per_frame_timing),
            }

        return {
            "detection_depth_ms": np.mean(self.timing_stats["detection_depth"]) * 1000
            if self.timing_stats["detection_depth"]
            else 0,
            "detection_detector_forward_ms": 0.0,
            "detection_depth_forward_ms": 0.0,
            "detection_depth_estimation_ms": 0.0,
            "detection_semantic_feature_ms": 0.0,
            "detection_bev_projection_ms": 0.0,
            "detection_other_ms": 0.0,
            "detection_cameras_processed": 0.0,
            "detection_detections_out": 0.0,
            "clustering_ms": np.mean(self.timing_stats["clustering"]) * 1000
            if self.timing_stats.get("clustering")
            else 0,
            "fusion_ms": np.mean(self.timing_stats["fusion"]) * 1000
            if self.timing_stats["fusion"]
            else 0,
            "tracking_ms": np.mean(self.timing_stats["tracking"]) * 1000
            if self.timing_stats["tracking"]
            else 0,
            "inference_total_ms": 0.0,
            "inference_total_std_ms": 0.0,
            "total_time_s": float(total_time_s),
            "fps": frame_count / total_time_s if total_time_s > 0 else 0.0,
            "frames_processed": int(frame_count),
            "per_frame": [],
        }

    def save_results(self, results: Dict, suffix: str = ""):
        """Save evaluation results to JSON."""
        output_file = (
            self.output_dir / f"results_{self.matching_mode}{suffix}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Evaluator] Saved results to {output_file}")
        return output_file

    def print_results(self, results: Dict):
        """Pretty-print evaluation results."""
        mode = results["mode"]
        mot = results["mot_metrics"]
        mot_pymot = results.get("mot_metrics_pymotmetrics", {})
        gospa = results.get("gospa_metrics", {})
        timing = results["timing"]

        def _fmt_or_na(value, fmt: str) -> str:
            if value is None:
                return "N/A"
            try:
                v = float(value)
            except Exception:
                return "N/A"
            if np.isnan(v) or np.isinf(v):
                return "N/A"
            return format(v, fmt)

        print(f"\n{'='*70}")
        print(f"ModTrack Evaluation Results - Mode: {mode.upper()}")
        print(f"{'='*70}")

        fusion_f1 = results.get("fusion_f1")
        if fusion_f1 is not None:
            print(f"\nFusion F1 (per-frame average):")
            print(f"  Assoc threshold:                          {fusion_f1.get('association_threshold_m', 0):.2f} m")
            print(f"  Frames evaluated:                         {fusion_f1.get('num_frames', 0)}")
            print(f"  Avg precision:                            {fusion_f1.get('avg_precision', 0):.3f}")
            print(f"  Avg recall:                               {fusion_f1.get('avg_recall', 0):.3f}")
            print(f"  Avg F1:                                   {fusion_f1.get('avg_f1', 0):.3f}")

        print(f"\nMOT Metrics (TrackEval):")
        print(f"  MOTA (Multi-Object Tracking Accuracy):    {mot.get('MOTA', 0):.1f}%")
        print(f"  MOTP (Multi-Object Tracking Precision):   {mot.get('MOTP', 0):.3f} m ({mot.get('MOTP_pct', 0):.1f}%)")
        print(f"  IDF1 (ID F1 Score - Identity Preserve):   {mot.get('IDF1', 0):.1f}%")
        print(f"  HOTA (Higher Order Tracking Accuracy):    {mot.get('HOTA', 0):.1f}%")
        # Some evaluators use 'IDS' while others use 'IDSW'; prefer IDS when available.
        id_switches = mot.get("IDS", mot.get("IDSW", 0))
        print(f"  ID Switches:                              {id_switches}")
        print(f"  Fragmentations:                           {mot.get('Frag', 0)}")
        print(f"  Mostly Tracked (MT):                      {mot.get('MT', 0):.1f}%")
        print(f"  Mostly Lost (ML):                         {mot.get('ML', 0):.1f}%")

        if mot_pymot:
            print(f"\nMOT Metrics (py-motmetrics):")
            print(f"  MOTA (Multi-Object Tracking Accuracy):    {_fmt_or_na(mot_pymot.get('MOTA'), '.1f')}%")
            print(
                f"  MOTP (Multi-Object Tracking Precision):   "
                f"{_fmt_or_na(mot_pymot.get('MOTP'), '.3f')} m ({_fmt_or_na(mot_pymot.get('MOTP_pct'), '.1f')}%)"
            )
            print(f"  IDF1 (ID F1 Score - Identity Preserve):   {_fmt_or_na(mot_pymot.get('IDF1'), '.1f')}%")
            print(f"  HOTA (Higher Order Tracking Accuracy):    {_fmt_or_na(mot_pymot.get('HOTA'), '.1f')}%")
            id_switches_pymot = mot_pymot.get("IDS", mot_pymot.get("IDSW", 0))
            print(f"  ID Switches:                              {id_switches_pymot}")
            print(f"  Fragmentations:                           {mot_pymot.get('Frag', 0)}")
            print(f"  Mostly Tracked (MT):                      {_fmt_or_na(mot_pymot.get('MT'), '.1f')}%")
            print(f"  Mostly Lost (ML):                         {_fmt_or_na(mot_pymot.get('ML'), '.1f')}%")

        if gospa:
            print(f"\nGOSPA Metrics (gospapy):")
            print(f"  GOSPA:                                    {_fmt_or_na(gospa.get('GOSPA'), '.4f')}")
            print(f"  Localization component:                   {_fmt_or_na(gospa.get('GOSPA_loc'), '.4f')}")
            print(f"  Missed-target component:                  {_fmt_or_na(gospa.get('GOSPA_missed'), '.4f')}")
            print(f"  False-track component:                    {_fmt_or_na(gospa.get('GOSPA_false'), '.4f')}")
            print(f"  Frames evaluated:                         {int(gospa.get('num_frames', 0))}")
            print(
                f"  Params (c / p / alpha):                   "
                f"{_fmt_or_na(gospa.get('max_distance'), '.3f')} / "
                f"{_fmt_or_na(gospa.get('p'), '.2f')} / "
                f"{_fmt_or_na(gospa.get('alpha'), '.2f')}"
            )

        neural_ms = timing['detection_depth_ms']
        det_forward_ms = float(timing.get("detection_detector_forward_ms", 0.0))
        depth_forward_ms = float(timing.get("detection_depth_forward_ms", 0.0))
        depth_est_ms = float(timing.get("detection_depth_estimation_ms", 0.0))
        sem_feat_ms = float(timing.get("detection_semantic_feature_ms", 0.0))
        bev_proj_ms = float(timing.get("detection_bev_projection_ms", 0.0))
        det_other_ms = float(timing.get("detection_other_ms", 0.0))
        det_cams = float(timing.get("detection_cameras_processed", 0.0))
        det_out = float(timing.get("detection_detections_out", 0.0))
        clustering_ms = timing.get('clustering_ms', 0)
        fusion_ms = timing['fusion_ms']
        tracking_ms = timing['tracking_ms']
        classical_ms = clustering_ms + fusion_ms + tracking_ms
        total_ms = neural_ms + classical_ms
        inference_total_ms = timing.get("inference_total_ms", total_ms)
        inference_std_ms = timing.get("inference_total_std_ms", 0.0)

        print(f"\nTiming (per-frame average):")
        timing_skip = int(timing.get("timing_warmup_skip_frames", 0) or 0)
        timing_used = int(timing.get("timing_frames_used", 0) or 0)
        if timing_skip > 0 and timing_used > 0:
            print(f"  Timing warmup skip:                        first {timing_skip} frame(s) excluded ({timing_used} used)")
        print(f"  Detection + Depth (neural):               {neural_ms:.1f} ms")
        if neural_ms > 0.0:
            print(f"    Detector forward:                       {det_forward_ms:.1f} ms ({(det_forward_ms / neural_ms) * 100:.0f}%)")
            print(f"    Depth forward:                          {depth_forward_ms:.1f} ms ({(depth_forward_ms / neural_ms) * 100:.0f}%)")
            print(f"    Depth at detections:                    {depth_est_ms:.1f} ms ({(depth_est_ms / neural_ms) * 100:.0f}%)")
            print(f"    BEV projection + geometry:              {bev_proj_ms:.1f} ms ({(bev_proj_ms / neural_ms) * 100:.0f}%)")
            print(f"    Semantic feature extraction:            {sem_feat_ms:.1f} ms ({(sem_feat_ms / neural_ms) * 100:.0f}%)")
            print(f"    Other detection/depth overhead:         {det_other_ms:.1f} ms ({(det_other_ms / neural_ms) * 100:.0f}%)")
            if det_cams > 0.0 or det_out > 0.0:
                print(f"    Avg cameras processed / detections out: {det_cams:.1f} / {det_out:.1f}")
        print(f"  Graph Clustering (classical):              {clustering_ms:.1f} ms")
        print(f"  Identity Fusion + Hungarian (classical):   {fusion_ms:.1f} ms")
        print(f"  GM-PHD-HMM Tracking (classical):           {tracking_ms:.1f} ms")
        print(f"  -----------------------------------------")
        print(f"  Stage sum total:                           {total_ms:.1f} ms")
        print(f"  End-to-end inference total:                {inference_total_ms:.1f} ms")
        if inference_std_ms > 0:
            print(f"  Inference total std:                       {inference_std_ms:.1f} ms")
        if total_ms > 0:
            print(f"  Neural / Classical split:                  {neural_ms:.1f} / {classical_ms:.1f} ms ({neural_ms/total_ms*100:.0f}% / {classical_ms/total_ms*100:.0f}%)")
        print(f"  FPS:                                       {timing['fps']:.1f}")
        if timing.get('gpu_memory_peak_mb'):
            print(f"  GPU Memory (peak):                         {timing['gpu_memory_peak_mb']:.0f} MB")

        # Calibration summary
        cal = results.get("calibration", {})
        if cal and not cal.get("error"):
            ci95 = cal.get("mean_nees_ci_95", [0.0, 0.0])
            if isinstance(ci95, (tuple, list)) and len(ci95) == 2:
                ci95_lower = float(ci95[0])
                ci95_upper = float(ci95[1])
            else:
                ci95_lower = 0.0
                ci95_upper = 0.0
            supporting_pct = float(
                cal.get("supporting_pct_in_chi2_95", cal.get("pct_in_chi2_95", 0.0))
            )
            supporting_ks_p = float(
                cal.get("supporting_ks_pvalue", cal.get("ks_pvalue", 0.0))
            )
            print(f"\nCalibration Analysis:")
            print(f"  Samples:                                   {cal.get('n_samples', 0)}")
            print(f"  Primary verdict (CI-based mean NEES):")
            print(f"    Mean NEES (expected=2.0):                 {cal.get('mean_nees', 0):.2f}")
            print(f"    Mean NEES 95% CI under H0:                [{ci95_lower:.2f}, {ci95_upper:.2f}]")
            print(f"    Verdict:                                  {cal.get('interpretation', 'N/A')}")
            print(f"  Supporting stats:")
            print(f"    % in chi2(2) 95% interval:                {supporting_pct:.1f}%")
            print(f"    KS test p-value:                          {supporting_ks_p:.4f}")
            if cal.get("coverage_test_enabled", False):
                cov1_emp = float(cal.get("coverage_1sigma_empirical", 0.0)) * 100.0
                cov1_exp = float(cal.get("coverage_1sigma_expected", 0.0)) * 100.0
                cov2_emp = float(cal.get("coverage_2sigma_empirical", 0.0)) * 100.0
                cov2_exp = float(cal.get("coverage_2sigma_expected", 0.0)) * 100.0
                print(f"  Coverage stats:")
                print(f"    1σ coverage (empirical/expected):         {cov1_emp:.1f}% / {cov1_exp:.1f}%")
                print(f"    2σ coverage (empirical/expected):         {cov2_emp:.1f}% / {cov2_exp:.1f}%")

        print(f"\n{'='*70}\n")
