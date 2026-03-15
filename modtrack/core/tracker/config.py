from typing import Dict, Any, Literal, Optional
from dataclasses import dataclass, asdict


# ======================== Mode Definitions ========================

MODE_SPATIAL_ONLY = "spatial"
MODE_SEMANTIC_ONLY = "semantic"
MODE_JOINT = "joint"

VALID_MODES = {MODE_SPATIAL_ONLY, MODE_SEMANTIC_ONLY, MODE_JOINT}

# ======================== Dataset Depth Cutoffs ========================

MULTIVIEWX_DEPTH_VAR_CUTOFF = 20.0
WILDTRACK_DEPTH_VAR_CUTOFF = 20.0  
RADARSCENES_DEPTH_VAR_CUTOFF = None  # Radar gives direct positions — no depth estimation

# ======================== Dataset BEV Pose Uncertainty ========================
WILDTRACK_R_POSE = 0.03
MULTIVIEWX_R_POSE = 0.05
RADARSCENES_R_POSE = 0.10 
DEFAULT_R_POSE = MULTIVIEWX_R_POSE

R_POSE_VARIANCES = {
    "multiviewx": MULTIVIEWX_R_POSE,
    "wildtrack": WILDTRACK_R_POSE,
    "radarscenes": RADARSCENES_R_POSE,
}


def get_r_pose_variance(dataset: Optional[str]) -> float:
    """Return the shared BEV pose/calibration variance (m^2) for a dataset."""
    if not dataset:
        return float(DEFAULT_R_POSE)
    return float(R_POSE_VARIANCES.get(dataset.lower(), DEFAULT_R_POSE))

# ======================== BEV Covariance Eigenvalue Floor ========================
DEFAULT_MIN_BEV_EIGENVALUE = 0.80  

# ======================== Dataset Frame Step ========================

# ======================== Dataset NMS Distances ========================
# Class-dependent NMS to handle different object sizes
# Pedestrians: 0.3m (shoulder-width separation)
# Vehicles: 1.5m (car/truck width allows tighter suppression of camera-overlap duplicates)
# Two-wheelers: 0.5m (motorcycle/bicycle)
NMS_DIST_BY_CLASS = {
    "pedestrian": 0.3,
    "car": 1.5,
    "truck": 1.5,
    "bus": 2.0,
    "trailer": 2.0,
    "motorcycle": 0.5,
    "bicycle": 0.5,
}
NMS_DIST_DEFAULT = 0.3   # fallback for unknown classes

# ======================== Dataset Fusion F1 Threshold ========================
DEFAULT_FUSION_F1_THRESHOLD = 0.5   

DEPTH_VAR_CUTOFFS = {
    "multiviewx": MULTIVIEWX_DEPTH_VAR_CUTOFF,
    "wildtrack": WILDTRACK_DEPTH_VAR_CUTOFF,
    "radarscenes": RADARSCENES_DEPTH_VAR_CUTOFF,
}

DEFAULT_DEPTH_VAR_CUTOFF = None

def get_depth_var_cutoff(dataset: Optional[str]) -> Optional[float]:
    """Return dataset-specific depth variance cutoff (m^2), or None to disable."""
    if not dataset:
        return DEFAULT_DEPTH_VAR_CUTOFF
    return DEPTH_VAR_CUTOFFS.get(dataset.lower(), DEFAULT_DEPTH_VAR_CUTOFF)

# ======================== Dataset Depth Bins ========================
MULTIVIEWX_DEPTH_MAX = 36.0
WILDTRACK_DEPTH_MAX = 36.0
RADARSCENES_DEPTH_MAX = 100.0  # 77 GHz radar max range

DEPTH_MAXES = {
    "multiviewx": MULTIVIEWX_DEPTH_MAX,
    "wildtrack": WILDTRACK_DEPTH_MAX,
    "radarscenes": RADARSCENES_DEPTH_MAX,
}

DEFAULT_DEPTH_MAX = WILDTRACK_DEPTH_MAX

def get_depth_max(dataset: Optional[str]) -> float:
    """Return dataset-specific max depth (m) for depth binning."""
    if not dataset:
        return DEFAULT_DEPTH_MAX
    return float(DEPTH_MAXES.get(dataset.lower(), DEFAULT_DEPTH_MAX))

# ======================== Pedestrian Height Prior ========================

MULTIVIEWX_PEDESTRIAN_HEIGHT_M = 1.75
WILDTRACK_PEDESTRIAN_HEIGHT_M = 1.73

RADARSCENES_PEDESTRIAN_HEIGHT_M = 1.75  
PEDESTRIAN_HEIGHTS = {
    "multiviewx": MULTIVIEWX_PEDESTRIAN_HEIGHT_M,
    "wildtrack": WILDTRACK_PEDESTRIAN_HEIGHT_M,
    "radarscenes": RADARSCENES_PEDESTRIAN_HEIGHT_M,
}

DEFAULT_PEDESTRIAN_HEIGHT_M = WILDTRACK_PEDESTRIAN_HEIGHT_M

def get_pedestrian_height_m(dataset: Optional[str]) -> float:
    """Return dataset-specific pedestrian height prior (m)."""
    if not dataset:
        return float(DEFAULT_PEDESTRIAN_HEIGHT_M)
    return float(PEDESTRIAN_HEIGHTS.get(dataset.lower(), DEFAULT_PEDESTRIAN_HEIGHT_M))

# ======================== Depth Prior Range Gate ========================
MULTIVIEWX_DEPTH_PRIOR_MIN_RANGE_M = 0.0
WILDTRACK_DEPTH_PRIOR_MIN_RANGE_M = 0.0

DEPTH_PRIOR_MIN_RANGES = {
    "multiviewx": MULTIVIEWX_DEPTH_PRIOR_MIN_RANGE_M,
    "wildtrack": WILDTRACK_DEPTH_PRIOR_MIN_RANGE_M,
}

DEFAULT_DEPTH_PRIOR_MIN_RANGE_M = WILDTRACK_DEPTH_PRIOR_MIN_RANGE_M

def get_depth_prior_min_range_m(dataset: Optional[str]) -> float:
    """Return dataset-specific min range (m) to activate the depth prior."""
    if not dataset:
        return float(DEFAULT_DEPTH_PRIOR_MIN_RANGE_M)
    return float(DEPTH_PRIOR_MIN_RANGES.get(dataset.lower(), DEFAULT_DEPTH_PRIOR_MIN_RANGE_M))

# ======================== BEV Covariance Floor ========================
MULTIVIEWX_BEV_MIN_VARIANCE = 0.21
WILDTRACK_BEV_MIN_VARIANCE = 0.16
RADARSCENES_BEV_MIN_VARIANCE = 0.21

BEV_MIN_VARIANCES = {
    "multiviewx": MULTIVIEWX_BEV_MIN_VARIANCE,
    "wildtrack": WILDTRACK_BEV_MIN_VARIANCE,
    "radarscenes": RADARSCENES_BEV_MIN_VARIANCE,
}

DEFAULT_BEV_MIN_VARIANCE = MULTIVIEWX_BEV_MIN_VARIANCE

def get_bev_min_variance(dataset: Optional[str]) -> float:
    """Return dataset-specific minimum BEV covariance eigenvalue floor (m^2)."""
    if not dataset:
        return float(DEFAULT_BEV_MIN_VARIANCE)
    return float(BEV_MIN_VARIANCES.get(dataset.lower(), DEFAULT_BEV_MIN_VARIANCE))

# ======================== PHD Process Noise Scaling ========================
MULTIVIEWX_PHD_PROCESS_NOISE_SCALE = 0.9
WILDTRACK_PHD_PROCESS_NOISE_SCALE = 0.9
RADARSCENES_PHD_PROCESS_NOISE_SCALE = 15.0 

PHD_PROCESS_NOISE_SCALES = {
    "multiviewx": MULTIVIEWX_PHD_PROCESS_NOISE_SCALE,
    "wildtrack": WILDTRACK_PHD_PROCESS_NOISE_SCALE,
    "radarscenes": RADARSCENES_PHD_PROCESS_NOISE_SCALE,
}

DEFAULT_PHD_PROCESS_NOISE_SCALE = MULTIVIEWX_PHD_PROCESS_NOISE_SCALE

def get_phd_process_noise_scale(dataset: Optional[str]) -> float:
    """Return dataset-specific multiplicative scale for GM-PHD process noise Q."""
    if not dataset:
        return float(DEFAULT_PHD_PROCESS_NOISE_SCALE)
    return float(PHD_PROCESS_NOISE_SCALES.get(dataset.lower(), DEFAULT_PHD_PROCESS_NOISE_SCALE))

# ======================== Footpoint Projection Tuning ========================
MULTIVIEWX_FOOTPOINT_BASE_SIGMA_SCALE = 0.035
WILDTRACK_FOOTPOINT_BASE_SIGMA_SCALE = 0.035

FOOTPOINT_BASE_SIGMA_SCALES = {
    "multiviewx": MULTIVIEWX_FOOTPOINT_BASE_SIGMA_SCALE,
    "wildtrack": WILDTRACK_FOOTPOINT_BASE_SIGMA_SCALE,
}

DEFAULT_FOOTPOINT_BASE_SIGMA_SCALE = MULTIVIEWX_FOOTPOINT_BASE_SIGMA_SCALE

FOOTPOINT_MIN_DEPTH_VAR = 1e-4

FOOTPOINT_LIFT_DISAGREE_M = 3.0
MULTIVIEWX_FOOTPOINT_BBOX_DISAGREE_M = 3.0
WILDTRACK_FOOTPOINT_BBOX_DISAGREE_M = 3.0

FOOTPOINT_BBOX_DISAGREE_MS = {
    "multiviewx": MULTIVIEWX_FOOTPOINT_BBOX_DISAGREE_M,
    "wildtrack": WILDTRACK_FOOTPOINT_BBOX_DISAGREE_M,
}

DEFAULT_FOOTPOINT_BBOX_DISAGREE_M = MULTIVIEWX_FOOTPOINT_BBOX_DISAGREE_M

FOOTPOINT_LIFT_CONFIDENT_VAR_MAX = 1.2

FOOTPOINT_LIFT_INFLATE = 1.75

MULTIVIEWX_FOOTPOINT_BBOX_AGREE_SHRINK = 1.0
WILDTRACK_FOOTPOINT_BBOX_AGREE_SHRINK = 1.0

FOOTPOINT_BBOX_AGREE_SHRINKS = {
    "multiviewx": MULTIVIEWX_FOOTPOINT_BBOX_AGREE_SHRINK,
    "wildtrack": WILDTRACK_FOOTPOINT_BBOX_AGREE_SHRINK,
}

DEFAULT_FOOTPOINT_BBOX_AGREE_SHRINK = MULTIVIEWX_FOOTPOINT_BBOX_AGREE_SHRINK


def get_footpoint_base_sigma_scale(dataset: Optional[str] = None) -> float:
    """Return dataset-specific base scale for footpoint depth uncertainty."""
    if not dataset:
        return float(DEFAULT_FOOTPOINT_BASE_SIGMA_SCALE)
    return float(FOOTPOINT_BASE_SIGMA_SCALES.get(dataset.lower(), DEFAULT_FOOTPOINT_BASE_SIGMA_SCALE))


def get_footpoint_min_depth_var() -> float:
    """Return the minimum variance floor for depth cues."""
    return float(FOOTPOINT_MIN_DEPTH_VAR)


def get_footpoint_lift_disagree_m() -> float:
    """Return disagreement threshold (meters) for Lift-vs-footpoint checks."""
    return float(FOOTPOINT_LIFT_DISAGREE_M)


def get_footpoint_bbox_disagree_m(dataset: Optional[str] = None) -> float:
    """Return dataset-specific disagreement threshold for bbox proxy depth."""
    if not dataset:
        return float(DEFAULT_FOOTPOINT_BBOX_DISAGREE_M)
    return float(FOOTPOINT_BBOX_DISAGREE_MS.get(dataset.lower(), DEFAULT_FOOTPOINT_BBOX_DISAGREE_M))


def get_footpoint_lift_confident_var_max() -> float:
    """Return max Lift variance still considered 'confident'."""
    return float(FOOTPOINT_LIFT_CONFIDENT_VAR_MAX)


def get_footpoint_lift_inflate() -> float:
    """Return covariance inflation factor used when Lift strongly disagrees."""
    return float(FOOTPOINT_LIFT_INFLATE)


def get_footpoint_bbox_agree_shrink(dataset: Optional[str] = None) -> float:
    """Return dataset-specific covariance shrink factor for agreeing bbox proxy depth."""
    if not dataset:
        return float(DEFAULT_FOOTPOINT_BBOX_AGREE_SHRINK)
    return float(FOOTPOINT_BBOX_AGREE_SHRINKS.get(dataset.lower(), DEFAULT_FOOTPOINT_BBOX_AGREE_SHRINK))


# ======================== Base Configuration ========================

@dataclass
class ModTrackConfig:

    # Mode & Thresholds
    mode: str = MODE_JOINT
    mahal_thresh: float = 9.21 
    sem_thresh: float = 0.78    

    # Temperature scaling 
    tau_geo: float = 5.0  
    tau_sem: float = 2.0   
    lambda_kl: float = 0.5  

    # Kalman Fusion 
    alpha: float = 1.0    
    tau_new: float = 0.3  

    # Graph Clustering
    min_support: int = 2
    allow_single_cam: bool = True
    reproj_thresh: float = 5.0  
    max_iter: int = 100  

    # Depth Processing
    depth_sample_count: int = 10

    # Kalman/IMM (Section 4.5)
    kalman_process_noise_cv: float = 0.5
    kalman_process_noise_ca: float = 1.0
    kalman_process_noise_stat: float = 0.1

    def __post_init__(self):
        """Validate configuration on initialization."""
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of {VALID_MODES}"
        )

        # Validate mode-specific parameters
        if self.mode == MODE_SPATIAL_ONLY:
            # Spatial mode ignores semantic parameters
            pass

        elif self.mode == MODE_SEMANTIC_ONLY:
            # Semantic mode requires appearance features
            if self.sem_thresh < 0 or self.sem_thresh > 1:
                raise ValueError(f"sem_thresh must be in [0, 1], got {self.sem_thresh}")

        elif self.mode == MODE_JOINT:
            # Joint mode uses both
            if self.sem_thresh < 0 or self.sem_thresh > 1:
                raise ValueError(f"sem_thresh must be in [0, 1], got {self.sem_thresh}")
            if self.lambda_kl < 0:
                raise ValueError(f"lambda_kl must be >= 0, got {self.lambda_kl}")

        # Common validations
        if self.mahal_thresh <= 0:
            raise ValueError(f"mahal_thresh must be > 0, got {self.mahal_thresh}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.tau_new < 0 or self.tau_new > 1:
            raise ValueError(f"tau_new must be in [0, 1], got {self.tau_new}")
        if self.min_support < 1:
            raise ValueError(f"min_support must be >= 1, got {self.min_support}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for legacy code compatibility)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModTrackConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_tracker_dict(self) -> Dict[str, Any]:
        """Get dictionary format for backward compatibility with existing code."""
        d = self.to_dict()
        # Add legacy 'spatial_only' flag based on mode
        d["spatial_only"] = (self.mode == MODE_SPATIAL_ONLY)
        return d


# ======================== Preset Configurations ========================

SPATIAL_CONFIG = ModTrackConfig(
    mode=MODE_SPATIAL_ONLY,
    mahal_thresh=9.21,
    sem_thresh=0.0,  # Not used in spatial mode
    max_iter=100,
    min_support=2,
    allow_single_cam=True,
)

SEMANTIC_CONFIG = ModTrackConfig(
    mode=MODE_SEMANTIC_ONLY,
    mahal_thresh=9.21,
    sem_thresh=0.6,  # Sweet spot: 0.70 too low (over-clusters), 0.85 too high (fragments)
    tau_sem=2.0,
    max_iter=100,
    min_support=2,
    allow_single_cam=True,
)

JOINT_CONFIG = ModTrackConfig(
    mode=MODE_JOINT,
    mahal_thresh=9.21,
    sem_thresh=0.15,
    tau_geo=5.0,
    tau_sem=2.0,
    lambda_kl=0.5,  # KL consistency check (reduced from 1.0 to prevent aggressive inflation)
    alpha=1.0,  # Entropy scaling (reduced from 2.0 to prevent excessive measurement noise)
    tau_new=0.3,
    max_iter=100,
    min_support=2,
    allow_single_cam=True,
)

CONFIG_PRESETS = {
    "spatial": SPATIAL_CONFIG,
    "semantic": SEMANTIC_CONFIG,
    "joint": JOINT_CONFIG,
}

# ======================== RadarScenes-Specific Presets ========================
# RadarScenes: 4 automotive radars with overlapping FoVs on a moving ego-vehicle.
# Radar gives direct BEV positions (no depth estimation) with precise range/azimuth.
# min_support=1 because single-radar observations are common at FoV edges.

RADARSCENES_SPATIAL_CONFIG = ModTrackConfig(
    mode=MODE_SPATIAL_ONLY,
    mahal_thresh=13.82,            # χ²(2, 0.999) — looser gating for highway-speed objects
    sem_thresh=0.0,
    min_support=1,
    allow_single_cam=True,
    kalman_process_noise_cv=3.0,   # Highway vehicles: ~25 m/s, need large Q for gating
    kalman_process_noise_ca=5.0,   # Maneuvering: lane changes, acceleration at highway speed
    kalman_process_noise_stat=0.5, # "Stationary" includes parked cars with ego-motion residuals
)

RADARSCENES_SEMANTIC_CONFIG = ModTrackConfig(
    mode=MODE_SEMANTIC_ONLY,
    mahal_thresh=13.82,
    sem_thresh=0.85,              # Radar pseudo-features are compact; high threshold needed
    tau_sem=2.0,
    min_support=1,
    allow_single_cam=True,
    kalman_process_noise_cv=3.0,
    kalman_process_noise_ca=5.0,
    kalman_process_noise_stat=0.5,
)

RADARSCENES_JOINT_CONFIG = ModTrackConfig(
    mode=MODE_JOINT,
    mahal_thresh=13.82,
    sem_thresh=0.50,              # Spatial is strong for radar; semantic is soft tiebreaker
    tau_geo=5.0,
    tau_sem=2.0,
    lambda_kl=0.3,
    alpha=1.0,
    tau_new=0.3,
    min_support=1,
    allow_single_cam=True,
    kalman_process_noise_cv=3.0,
    kalman_process_noise_ca=5.0,
    kalman_process_noise_stat=0.5,
)

RADARSCENES_CONFIG_PRESETS = {
    "spatial": RADARSCENES_SPATIAL_CONFIG,
    "semantic": RADARSCENES_SEMANTIC_CONFIG,
    "joint": RADARSCENES_JOINT_CONFIG,
}

# Per-dataset config lookup
DATASET_CONFIG_PRESETS = {
    "wildtrack": CONFIG_PRESETS,
    "multiviewx": CONFIG_PRESETS,
    "radarscenes": RADARSCENES_CONFIG_PRESETS,
}


SEM_THRESH_BY_MODE_DATASET: Dict[str, Dict[str, float]] = {
    MODE_SEMANTIC_ONLY: {
        "wildtrack": 0.70,
        "multiviewx": 0.60,
        "radarscenes": 0.85,  # Compact 8-dim radar features → high threshold
    },
    MODE_JOINT: {
        "wildtrack": 0.15,
        "multiviewx": 0.15,
        "radarscenes": 0.50,  # Spatial is primary; semantic is soft tiebreaker
    },
}

DEFAULT_SEM_THRESH_BY_MODE: Dict[str, float] = {
    MODE_SPATIAL_ONLY: SPATIAL_CONFIG.sem_thresh,
    MODE_SEMANTIC_ONLY: SEMANTIC_CONFIG.sem_thresh,
    MODE_JOINT: JOINT_CONFIG.sem_thresh,
}


def get_sem_thresh_for_dataset(mode: str, dataset: Optional[str] = None) -> float:
    """Return semantic threshold for (mode, dataset) with sensible fallback."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {sorted(VALID_MODES)}")

    mode_thresholds = SEM_THRESH_BY_MODE_DATASET.get(mode, {})
    default = float(DEFAULT_SEM_THRESH_BY_MODE.get(mode, 0.0))
    if not dataset:
        return default
    return float(mode_thresholds.get(dataset.lower().strip(), default))



# ======================== Configuration Builder ========================

class ConfigBuilder:

    def __init__(self, base: ModTrackConfig = None):
        """Initialize builder with optional base configuration."""
        self._config = base or JOINT_CONFIG.to_dict()

    def set_mode(self, mode: str) -> "ConfigBuilder":
        """Set matching mode."""
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode. Must be one of {VALID_MODES}")
        self._config["mode"] = mode
        return self

    def set_spatial_mode(self) -> "ConfigBuilder":
        """Set to spatial-only mode."""
        return self.set_mode(MODE_SPATIAL_ONLY)

    def set_semantic_mode(self) -> "ConfigBuilder":
        """Set to semantic-only mode."""
        return self.set_mode(MODE_SEMANTIC_ONLY)

    def set_joint_mode(self) -> "ConfigBuilder":
        """Set to joint mode (recommended)."""
        return self.set_mode(MODE_JOINT)

    def set_mahal_thresh(self, thresh: float) -> "ConfigBuilder":
        """Set Mahalanobis threshold (χ²₂,₀.₉₉ = 9.21 for 99% confidence)."""
        self._config["mahal_thresh"] = thresh
        return self

    def set_semantic_thresh(self, thresh: float) -> "ConfigBuilder":
        """Set semantic similarity threshold."""
        self._config["sem_thresh"] = thresh
        return self

    def set_kl_penalty(self, weight: float) -> "ConfigBuilder":
        """Set KL divergence penalty (joint mode only)."""
        self._config["lambda_kl"] = weight
        return self

    def set_entropy_scaling(self, alpha: float) -> "ConfigBuilder":
        """Set entropy scaling for measurement noise inflation."""
        self._config["alpha"] = alpha
        return self

    def set_new_identity_threshold(self, tau: float) -> "ConfigBuilder":
        """Set threshold for new identity initialization."""
        self._config["tau_new"] = tau
        return self

    def build(self) -> ModTrackConfig:
        """Build and validate configuration."""
        return ModTrackConfig.from_dict(self._config)


# ======================== Helper Functions ========================

def get_config_for_mode(mode: str, dataset: Optional[str] = None) -> ModTrackConfig:
    """Get preset configuration for a mode, optionally dataset-specific.

    Args:
        mode: One of 'spatial', 'semantic', or 'joint'
                 If provided and a dataset-specific preset exists, it is used.
                 Semantic threshold is further overridden per dataset.

    Returns:
        ModTrackConfig: Optimized configuration for the mode and dataset
    """
    presets = CONFIG_PRESETS
    if dataset:
        presets = DATASET_CONFIG_PRESETS.get(dataset.lower(), CONFIG_PRESETS)
    if mode not in presets:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {list(presets.keys())}")
    cfg = ModTrackConfig.from_dict(presets[mode].to_dict())
    # Apply per-dataset semantic threshold override
    cfg.sem_thresh = get_sem_thresh_for_dataset(mode, dataset)
    return cfg


def describe_mode(mode: str) -> str:
    """Get human-readable description of a mode.

    Returns:
        str: Description of the mode and when to use it
    """
    descriptions = {
        MODE_SPATIAL_ONLY: (
            "SPATIAL-ONLY: Uses only BEV position consistency (Mahalanobis distance). "
            "Fastest method, assumes reliable camera calibration. "
            "Best for: well-calibrated multi-view systems."
        ),
        MODE_SEMANTIC_ONLY: (
            "SEMANTIC-ONLY: Uses only learned feature similarity (OSNet embeddings). "
            "Robust to depth errors and calibration issues. "
            "Best for: uncalibrated cameras or when appearance is reliable."
        ),
        MODE_JOINT: (
            "JOINT (RECOMMENDED): Combines spatial position + semantic features with KL consistency. "
            "Most robust approach from the paper. "
            "Best for: challenging real-world scenarios with varying lighting/occlusion."
        ),
    }
    return descriptions.get(mode, f"Unknown mode: {mode}")


# ======================== Legacy Compatibility ========================

# For backward compatibility with existing code using dict config
DEFAULT_TRACKER_CONFIG = JOINT_CONFIG.to_dict()
