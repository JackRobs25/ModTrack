"""Class-Based Motion Model Profiles for ModTrack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ======================== Motion Mode Definition ========================

@dataclass
class MotionMode:
    """A single motion mode within the HMM.

    Attributes:
        name: Human-readable mode name
        F: State transition matrix [4x4] for state [x, y, vx, vy]
        Q: Process noise covariance [4x4]
    """
    name: str
    F: np.ndarray  # [4, 4]
    Q: np.ndarray  # [4, 4]

    def __post_init__(self):
        self.F = np.asarray(self.F, dtype=np.float32)
        self.Q = np.asarray(self.Q, dtype=np.float32)
        assert self.F.shape == (4, 4), f"F shape mismatch: {self.F.shape}"
        assert self.Q.shape == (4, 4), f"Q shape mismatch: {self.Q.shape}"


# ======================== Class Motion Profile ========================

@dataclass
class ClassMotionProfile:
    """Complete motion model profile for an object class.
    """
    class_name: str
    yolo_class_ids: List[int]
    transition_matrix: np.ndarray
    modes: List[MotionMode]
    initial_velocity_std: float
    birth_weight_threshold: float
    survival_probability: float
    detection_probability: Optional[float] = None
    clutter_intensity: Optional[float] = None

    def __post_init__(self):
        self.transition_matrix = np.asarray(self.transition_matrix, dtype=np.float32)
        n_modes = len(self.modes)
        assert self.transition_matrix.shape == (n_modes, n_modes), (
            f"Transition matrix shape {self.transition_matrix.shape} doesn't match "
            f"{n_modes} modes"
        )
        # Rows must sum to 1
        row_sums = self.transition_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"Transition matrix rows must sum to 1, got {row_sums}"
        )

    @property
    def num_modes(self) -> int:
        return len(self.modes)

    def get_mode(self, mode_idx: int) -> MotionMode:
        return self.modes[mode_idx]

    def get_F(self, mode_idx: int) -> np.ndarray:
        return self.modes[mode_idx].F

    def get_Q(self, mode_idx: int) -> np.ndarray:
        return self.modes[mode_idx].Q

    def get_mode_name(self, mode_idx: int) -> str:
        return self.modes[mode_idx].name


# ======================== Common State Transition Matrices ========================
F_STATIONARY = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
], dtype=np.float32)

F_CONSTANT_VELOCITY = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)

F_MANEUVERING = F_CONSTANT_VELOCITY.copy()


# ======================== Class Profiles ========================

PEDESTRIAN_PROFILE = ClassMotionProfile(
    class_name="pedestrian",
    yolo_class_ids=[0],  # YOLO 'person'
    transition_matrix=np.array([
        [0.75, 0.20, 0.05],  # stationary → starts walking (less aggressive)
        [0.03, 0.94, 0.03],  # CV → VERY STICKY (routes = long straight segments, 94% = ~16 frames before turn)
        [0.05, 0.85, 0.10],  # maneuvering → FAST return to CV (brief turns only, 85% = ~2 frames)
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.20, 0.20, 0.02, 0.02]).astype(np.float32)),  # 2x original - tighter for NEES ~2.0
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.20, 0.20, 1.0, 1.0]).astype(np.float32)),    # 2x original - trust straight-line motion
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.25, 0.25, 2.0, 2.0]).astype(np.float32)),    # 2x original - tighter during brief turns
    ],
    initial_velocity_std=1.0,   # m/s — typical pedestrian speed uncertainty
    birth_weight_threshold=0.65,
    survival_probability=0.99,
    detection_probability=0.75,  # Pedestrians: CenterPoint pedestrian recall ~70%
    clutter_intensity=0.40,      # Higher false positive rate
)

PEDESTRIAN_PROFILE_WILDTRACK = ClassMotionProfile(
    class_name="pedestrian",
    yolo_class_ids=[0], 
    transition_matrix=np.array([
        [0.75, 0.20, 0.05],  
        [0.03, 0.94, 0.03], 
        [0.05, 0.85, 0.10],
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.20, 0.20, 0.02, 0.02]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.20, 0.20, 1.0, 1.0]).astype(np.float32)),
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.25, 0.25, 2.0, 2.0]).astype(np.float32)),
    ],
    initial_velocity_std=1.0,
    birth_weight_threshold=0.65,
    survival_probability=0.99,
    detection_probability=0.9,
    clutter_intensity=0.30,
)


PEDESTRIAN_PROFILE_MULTIVIEWX = ClassMotionProfile(
    class_name="pedestrian",
    yolo_class_ids=[0],  
    transition_matrix=np.array([
        [0.75, 0.20, 0.05], 
        [0.03, 0.94, 0.03],  
        [0.05, 0.85, 0.10],  
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.20, 0.20, 0.02, 0.02]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.20, 0.20, 1.0, 1.0]).astype(np.float32)),
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.25, 0.25, 2.0, 2.0]).astype(np.float32)),
    ],
    initial_velocity_std=1.0,
    birth_weight_threshold=0.65,
    survival_probability=0.99,
    detection_probability=0.95,
    clutter_intensity=0.30,
)

CAR_PROFILE = ClassMotionProfile(
    class_name="car",
    yolo_class_ids=[2],  
    transition_matrix=np.array([
        [0.85, 0.10, 0.05], 
        [0.05, 0.90, 0.05], 
        [0.10, 0.10, 0.80], 
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.05, 0.05, 0.01, 0.01]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.05, 0.05, 0.10, 0.10]).astype(np.float32)),  
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.10, 0.10, 0.50, 0.50]).astype(np.float32)),
    ],
    initial_velocity_std=5.0,   
    birth_weight_threshold=0.70,
    survival_probability=0.97,  
    detection_probability=0.75,  
    clutter_intensity=0.10,      
)

TRUCK_PROFILE = ClassMotionProfile(
    class_name="truck",
    yolo_class_ids=[7],  
    transition_matrix=np.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.90, 0.05],
        [0.10, 0.10, 0.80],
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.05, 0.05, 0.005, 0.005]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.05, 0.05, 0.08, 0.08]).astype(np.float32)),  
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.10, 0.10, 0.30, 0.30]).astype(np.float32)),
    ],
    initial_velocity_std=4.0,
    birth_weight_threshold=0.70,
    survival_probability=0.97,
    detection_probability=0.70,  
    clutter_intensity=0.15,
)

BUS_PROFILE = ClassMotionProfile(
    class_name="bus",
    yolo_class_ids=[5],  
    transition_matrix=np.array([
        [0.80, 0.15, 0.05],  
        [0.10, 0.85, 0.05],
        [0.10, 0.10, 0.80],
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.05, 0.05, 0.005, 0.005]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.05, 0.05, 0.10, 0.10]).astype(np.float32)),
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.10, 0.10, 0.40, 0.40]).astype(np.float32)),
    ],
    initial_velocity_std=4.0,
    birth_weight_threshold=0.70,
    survival_probability=0.97,
    detection_probability=0.70,  
    clutter_intensity=0.10,
)


MOTORCYCLE_PROFILE = ClassMotionProfile(
    class_name="motorcycle",
    yolo_class_ids=[3],  
    transition_matrix=np.array([
        [0.80, 0.10, 0.10],  
        [0.05, 0.80, 0.15],
        [0.10, 0.15, 0.75],  
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.08, 0.08, 0.01, 0.01]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.08, 0.08, 0.30, 0.30]).astype(np.float32)),
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.15, 0.15, 0.80, 0.80]).astype(np.float32)),
    ],
    initial_velocity_std=5.0,
    birth_weight_threshold=0.65,
    survival_probability=0.95,
    detection_probability=0.90,  
    clutter_intensity=0.30,
)

BICYCLE_PROFILE = ClassMotionProfile(
    class_name="bicycle",
    yolo_class_ids=[1], 
    transition_matrix=np.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.85, 0.10],
        [0.10, 0.10, 0.80],
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.08, 0.08, 0.01, 0.01]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.08, 0.08, 0.30, 0.30]).astype(np.float32)),
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.12, 0.12, 0.70, 0.70]).astype(np.float32)),
    ],
    initial_velocity_std=3.0,   
    birth_weight_threshold=0.65,
    survival_probability=0.95,
    detection_probability=0.90,  
    clutter_intensity=0.30,
)


TRAILER_PROFILE = ClassMotionProfile(
    class_name="trailer",
    yolo_class_ids=[8],  
    transition_matrix=np.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.92, 0.03], 
        [0.10, 0.10, 0.80],
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.03, 0.03, 0.005, 0.005]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.03, 0.03, 0.05, 0.05]).astype(np.float32)),  # Extremely smooth
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.08, 0.08, 0.25, 0.25]).astype(np.float32)),
    ],
    initial_velocity_std=3.0,
    birth_weight_threshold=0.70,
    survival_probability=0.98,  
    detection_probability=0.85,  
    clutter_intensity=0.15,
)


CONSTRUCTION_VEHICLE_PROFILE = ClassMotionProfile(
    class_name="construction_vehicle",
    yolo_class_ids=[9],  
    transition_matrix=np.array([
        [0.70, 0.15, 0.15],
        [0.10, 0.80, 0.10],
        [0.15, 0.10, 0.75],
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.05, 0.05, 0.01, 0.01]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.05, 0.05, 0.15, 0.15]).astype(np.float32)),
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.15, 0.15, 0.40, 0.40]).astype(np.float32)),
    ],
    initial_velocity_std=2.0,
    birth_weight_threshold=0.65,
    survival_probability=0.95,
    detection_probability=0.80, 
    clutter_intensity=0.20,
)


# ======================== Profile Registry ========================

CLASS_PROFILES: Dict[str, ClassMotionProfile] = {
    "pedestrian": PEDESTRIAN_PROFILE,
    "car": CAR_PROFILE,
    "truck": TRUCK_PROFILE,
    "bus": BUS_PROFILE,
    "motorcycle": MOTORCYCLE_PROFILE,
    "bicycle": BICYCLE_PROFILE,
    "trailer": TRAILER_PROFILE,
    "construction_vehicle": CONSTRUCTION_VEHICLE_PROFILE,
}

_YOLO_CLASS_TO_PROFILE: Dict[int, ClassMotionProfile] = {}
for _profile in CLASS_PROFILES.values():
    for _cls_id in _profile.yolo_class_ids:
        _YOLO_CLASS_TO_PROFILE[_cls_id] = _profile

DEFAULT_PROFILE = PEDESTRIAN_PROFILE

RADAR_VEHICLE_PROFILE = ClassMotionProfile(
    class_name="radar_vehicle",
    yolo_class_ids=[2],  
    transition_matrix=np.array([
        [0.70, 0.25, 0.05],   
        [0.02, 0.95, 0.03],   
        [0.05, 0.80, 0.15],   
    ], dtype=np.float32),
    modes=[
        MotionMode("stationary", F_STATIONARY,
                   np.diag([0.05, 0.05, 0.01, 0.01]).astype(np.float32)),
        MotionMode("constant_velocity", F_CONSTANT_VELOCITY,
                   np.diag([0.08, 0.08, 0.20, 0.20]).astype(np.float32)),
        MotionMode("maneuvering", F_MANEUVERING,
                   np.diag([0.15, 0.15, 1.0, 1.0]).astype(np.float32)),
    ],
    initial_velocity_std=8.0,     
    birth_weight_threshold=0.40,
    survival_probability=0.99,    
    detection_probability=0.90,   
    clutter_intensity=0.15,
)

CLASS_PROFILES["radar_vehicle"] = RADAR_VEHICLE_PROFILE


PEDESTRIAN_PROFILE_BY_DATASET: Dict[str, ClassMotionProfile] = {
    "wildtrack": PEDESTRIAN_PROFILE_WILDTRACK,
    "multiviewx": PEDESTRIAN_PROFILE_MULTIVIEWX,
    "radarscenes": RADAR_VEHICLE_PROFILE,  
}


def get_profile_for_yolo_class(class_id: int) -> ClassMotionProfile:
    """Map a YOLO class ID to the appropriate motion profile.

    Args:
        class_id: YOLO detection class ID (e.g., 0=person, 2=car)

    Returns:
        ClassMotionProfile for the class, or PEDESTRIAN_PROFILE as fallback.
    """
    return _YOLO_CLASS_TO_PROFILE.get(class_id, DEFAULT_PROFILE)

_RADAR_YOLO_CLASS_TO_PROFILE: Dict[int, ClassMotionProfile] = {
    0: PEDESTRIAN_PROFILE,      
    2: RADAR_VEHICLE_PROFILE,   
}


def get_profile_for_radar_yolo_class(class_id: int) -> ClassMotionProfile:
    """Map a radar-inferred YOLO class ID to a radar-tuned motion profile.

    Args:
        class_id: Heuristic class ID (0=pedestrian, 2=vehicle)

    Returns:
        ClassMotionProfile for radar tracking.
    """
    return _RADAR_YOLO_CLASS_TO_PROFILE.get(class_id, RADAR_VEHICLE_PROFILE)


def get_profile_by_name(class_name: str) -> ClassMotionProfile:
    """Get a motion profile by class name.

    Args:
        class_name: Class name (e.g., "pedestrian", "car")

    Returns:
        ClassMotionProfile for the class, or PEDESTRIAN_PROFILE as fallback.
    """
    return CLASS_PROFILES.get(class_name.lower(), DEFAULT_PROFILE)


def get_profile_by_name_for_dataset(
    class_name: str, dataset: Optional[str]
) -> ClassMotionProfile:
    """Get a motion profile by class name with dataset-specific overrides.
    """
    class_key = (class_name or "").lower().strip()
    dataset_key = (dataset or "").lower().strip()
    if class_key == "pedestrian":
        ped_profile = PEDESTRIAN_PROFILE_BY_DATASET.get(dataset_key)
        if ped_profile is not None:
            return ped_profile
    return CLASS_PROFILES.get(class_key, DEFAULT_PROFILE)


def list_profiles() -> List[str]:
    """List all available motion profile names."""
    return list(CLASS_PROFILES.keys())


# ======================== RadarScenes Label Mapping ========================

# RadarScenes label_id → motion profile mapping
RADARSCENES_LABEL_TO_PROFILE: Dict[int, ClassMotionProfile] = {
    0: CAR_PROFILE,           # Car
    1: TRUCK_PROFILE,         # Large Vehicle
    2: TRUCK_PROFILE,         # Truck
    3: BUS_PROFILE,           # Bus
    4: BUS_PROFILE,           # Train (closest dynamics)
    5: BICYCLE_PROFILE,       # Bicycle
    6: MOTORCYCLE_PROFILE,    # Motorized Two-Wheeler
    7: PEDESTRIAN_PROFILE,    # Pedestrian
    8: PEDESTRIAN_PROFILE,    # Pedestrian Group
    9: PEDESTRIAN_PROFILE,    # Animal (similar dynamics)
    10: CAR_PROFILE,          # Other Dynamic (default)
    # 11: Static — not tracked
}


def get_profile_for_radarscenes_label(label_id: int) -> ClassMotionProfile:
    """Map a RadarScenes label_id to the appropriate motion profile.

    Args:
        label_id: RadarScenes class label (0-11)

    Returns:
        ClassMotionProfile for the class, or PEDESTRIAN_PROFILE as fallback.
    """
    return RADARSCENES_LABEL_TO_PROFILE.get(label_id, DEFAULT_PROFILE)
