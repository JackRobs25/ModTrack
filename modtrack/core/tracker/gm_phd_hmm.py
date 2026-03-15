"""
Identity-Informed Gaussian Mixture Probability Hypothesis Density Filter with HMM
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from collections import defaultdict as _defaultdict

logger = logging.getLogger(__name__)
logger.disabled = True

# Class-based motion model profiles
try:
    from .motion_models import (
        ClassMotionProfile,
        get_profile_by_name,
        get_profile_by_name_for_dataset,
        get_profile_for_yolo_class,
        DEFAULT_PROFILE,
        PEDESTRIAN_PROFILE,
    )
except ImportError:
    # Fallback: define inline if motion_models not available
    get_profile_by_name = None
    get_profile_by_name_for_dataset = None
    get_profile_for_yolo_class = None
    DEFAULT_PROFILE = None
    PEDESTRIAN_PROFILE = None

# ======================== Configuration Constants ========================

DEFAULT_DT = 0.5 

MOTION_MODES = {
    "stationary": {
        "name": "Stationary",
        "F": np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "Q": np.diag([0.20, 0.20, 0.04, 0.04]).astype(np.float32),  # further reduced process noise for less conservative covariance growth
    },
    "constant_velocity": {
        "name": "Constant Velocity",
        "F": np.array([[1.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "Q": np.diag([0.20, 0.20, 0.75, 0.75]).astype(np.float32),  # further reduced process noise to limit covariance over-growth
    },
    "maneuvering": {
        "name": "Maneuvering",
        "F": np.array([[1.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "Q": np.diag([0.35, 0.35, 1.5, 1.5]).astype(np.float32),  # further reduced process noise while preserving maneuver tolerance
    },
}

TRANSITION_MATRIX = np.array(
    [[0.90, 0.05, 0.05],   # From stationary
     [0.05, 0.90, 0.05],   # From constant velocity
     [0.05, 0.05, 0.90]],  # From maneuvering
    dtype=np.float32,
)

# PHD Filter Parameters
SURVIVAL_PROBABILITY = 0.995  
DETECTION_PROBABILITY = 0.90  
CLUTTER_INTENSITY = 0.3 

CONFIRMED_MISS_TOLERANCE = 0
OCCLUSION_PROXIMITY = 1.0  
OCCLUSION_P_D_SCALE = 0.5  
SHADOW_RADIUS = 0.5        
MIN_VISIBILITY_RATIO = 0.3 

# Component Management Thresholds
WEIGHT_PRUNE_THRESHOLD = 0.05 
WEIGHT_MERGE_THRESHOLD = 2.5  
WEIGHT_EXTRACT_THRESHOLD = 0.20  
LOST_POOL_WEIGHT_THRESHOLD = 0.005
MAX_COMPONENTS = 100  # J_max

# Identity Association Boosting
IDENTITY_ASSOC_BOOST = 2.5  
SEMANTIC_COST_BOOST = 2.0  
LOST_TRACK_REID_SEMANTIC_BOOST = 3.0  
MATCHED_CONFIRMED_WEIGHT_BOOST = 0.15  
ASSOC_DIAG_MAX_DETAILS = 8  
TURN_PENALTY_BASE = 1.5            
TURN_PENALTY_V_LOW = 0.20          
TURN_PENALTY_V_HIGH = 1.20         
TURN_PENALTY_MIN_DISP = 0.20       
TURN_PENALTY_COS_SOFT = 0.00       
TURN_PENALTY_MAX = 2.5             
# ======================== Track Lifecycle States ========================

class TrackState:
    """
    Discrete track lifecycle states for N-hit confirmation and lost recovery.
    """
    TENTATIVE = 0   # Born but not yet confirmed
    CONFIRMED = 1   # Actively tracked, output to evaluator
    LOST = 2        # Not detected, still predicted for re-identification

# Lifecycle parameters
N_INIT = 2              # Consecutive hits to confirm a tentative track
MAX_LOST_FRAMES = 2    # Frames to keep a lost track before deletion (original)
HIT_ASSOCIATION_THRESHOLD = 0.05  # Min sum(q_tilde) to count as a "hit" (original)

# ======================== Data Structures ========================

@dataclass
class GMPHDComponent:
    """
    Gaussian Mixture PHD Component with Identity Label and Lifecycle State.
    """

    weight: float
    mean: np.ndarray  # [4,] state vector [x, y, vx, vy]
    covariance: np.ndarray  # [4,4] state covariance
    identity: Optional[int] = None  # Persistent identity label
    motion_mode: int = 1  # Default: constant velocity
    object_class: str = "pedestrian"  # Object class for motion model selection
    feature: Optional[np.ndarray] = None  # Pooled multi-view appearance feature
    track_state: int = TrackState.TENTATIVE
    hit_count: int = 0
    miss_count: int = 0
    det_confidence: float = 0.5  # Best matched detection confidence (for AMOTA scoring)

    def __post_init__(self):
        """Ensure proper data types and shapes."""
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.covariance = np.asarray(self.covariance, dtype=np.float32)
        assert self.mean.shape == (4,), f"Mean shape mismatch: {self.mean.shape}"
        assert self.covariance.shape == (4, 4), f"Cov shape mismatch: {self.covariance.shape}"
        if self.feature is not None:
            self.feature = np.asarray(self.feature, dtype=np.float32)

    def copy(self) -> GMPHDComponent:
        """Create deep copy of component."""
        return GMPHDComponent(
            weight=self.weight,
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            identity=self.identity,
            motion_mode=self.motion_mode,
            object_class=self.object_class,
            feature=self.feature.copy() if self.feature is not None else None,
            track_state=self.track_state,
            hit_count=self.hit_count,
            miss_count=self.miss_count,
        )

# ======================== Covariance Safeguards ========================

def _clamp_covariance_eig(
    cov: np.ndarray,
    *,
    max_pos_var: float,
    max_vel_var: float,
    min_var: float,
) -> np.ndarray:
    """Clamp covariance eigenvalues to prevent explosion (original behavior)."""
    if not np.isfinite(cov).all():
        return np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Clamp position (indices 0,1) and velocity (indices 2,3) separately
        eigvals[:2] = np.clip(eigvals[:2], min_var, max_pos_var)
        eigvals[2:] = np.clip(eigvals[2:], min_var, max_vel_var)
        cov_clamped = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return ((cov_clamped + cov_clamped.T) / 2).astype(np.float32)
    except np.linalg.LinAlgError:
        return np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)

def _clamp_covariance_diag(
    cov: np.ndarray,
    *,
    max_pos_var: float,
    max_vel_var: float,
    min_var: float,
) -> np.ndarray:
    """Clamp diagonal variances and bound off-diagonal correlations."""
    if not np.isfinite(cov).all():
        return np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)

    cov = cov.copy().astype(np.float64)
    max_vars = np.array([max_pos_var, max_pos_var, max_vel_var, max_vel_var])
    for i in range(4):
        cov[i, i] = np.clip(cov[i, i], min_var, max_vars[i])

    for i in range(4):
        for j in range(i + 1, 4):
            max_offdiag = np.sqrt(cov[i, i] * cov[j, j])
            cov[i, j] = np.clip(cov[i, j], -max_offdiag, max_offdiag)
            cov[j, i] = cov[i, j]

    return cov.astype(np.float32)

def _clamp_covariance(
    cov: np.ndarray,
    *,
    mode: str = "eig",
    max_pos_var: float = 25.0,
    max_vel_var: float = 4.0,
    min_var: float = 1e-4,
) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode == "eig":
        return _clamp_covariance_eig(
            cov, max_pos_var=max_pos_var, max_vel_var=max_vel_var, min_var=min_var
        )
    if mode == "diag":
        return _clamp_covariance_diag(
            cov, max_pos_var=max_pos_var, max_vel_var=max_vel_var, min_var=min_var
        )
    raise ValueError(f"Unknown covariance clamp mode '{mode}'")

# ======================== Class-Based Motion Model Helper ========================

def _get_motion_profile(object_class: str, dataset: Optional[str] = None):
    if get_profile_by_name_for_dataset is not None:
        return get_profile_by_name_for_dataset(object_class, dataset)
    if get_profile_by_name is not None:
        return get_profile_by_name(object_class)
    return None

# ======================== GM-PHD-HMM Filter ========================

class IdentityInformedGMPHDFilter:

    def __init__(
        self,
        clutter_intensity: float = CLUTTER_INTENSITY,
        detection_prob: float = DETECTION_PROBABILITY,
        survival_prob: float = SURVIVAL_PROBABILITY,
        identity_assoc_boost: float = IDENTITY_ASSOC_BOOST,
        dt: float = DEFAULT_DT,
        process_noise_scale: float = 1.0,
        birth_threshold_scale: float = 1.0,
        confirmed_miss_tolerance: int = CONFIRMED_MISS_TOLERANCE,
        max_lost_frames: int = MAX_LOST_FRAMES,
        tentative_miss_tolerance: int = 1,
        cov_clamp_mode: str = "eig",
        cov_clamp_max_vel_var: float = 4.0,
        mahal_thresh: float = 9.21,
        dataset: Optional[str] = None,
        use_class_birth_mode_priors: bool = False,
    ):
        self.dt = dt
        self.dataset = str(dataset).lower().strip() if dataset is not None else None
        self.process_noise_scale = float(max(process_noise_scale, 1e-6))
        self.birth_threshold_scale = birth_threshold_scale
        self.clutter_intensity = clutter_intensity
        self._base_clutter_intensity = clutter_intensity  # Adaptive clutter baseline
        self._clutter_ema_alpha = 0.3  # EMA smoothing factor for clutter adaptation
        self.detection_prob = detection_prob
        self.survival_prob = survival_prob
        self.identity_assoc_boost = identity_assoc_boost
        self.confirmed_miss_tolerance = confirmed_miss_tolerance
        self.max_lost_frames = max_lost_frames
        self.tentative_miss_tolerance = tentative_miss_tolerance
        self.cov_clamp_mode = str(cov_clamp_mode).lower().strip()
        self.cov_clamp_max_vel_var = float(cov_clamp_max_vel_var)
        self.mahal_thresh = float(mahal_thresh)
        self.use_class_birth_mode_priors = bool(use_class_birth_mode_priors)

        # Components (GM representation of intensity function)
        self.components: List[GMPHDComponent] = []

        # Track extraction history
        self.extracted_tracks: List[Dict] = []
    def reset(self):
        self.components.clear()
        self.extracted_tracks.clear()
        self.clutter_intensity = self._base_clutter_intensity

    def _clamp_state_covariance(self, cov: np.ndarray) -> np.ndarray:
        return _clamp_covariance(
            cov,
            mode=self.cov_clamp_mode,
            max_vel_var=self.cov_clamp_max_vel_var,
        )

    @staticmethod
    def _state_name(state: int) -> str:
        if state == TrackState.TENTATIVE:
            return "TENT"
        if state == TrackState.CONFIRMED:
            return "CONF"
        if state == TrackState.LOST:
            return "LOST"
        return f"UNK({state})"

    def _get_class_detection_prob(self, component: GMPHDComponent) -> float:
        """Get per-class detection probability, falling back to global default."""
        profile = _get_motion_profile(component.object_class, self.dataset)
        if profile is not None and profile.detection_probability is not None:
            return profile.detection_probability
        return self.detection_prob

    def _get_class_clutter_intensity(self, component: GMPHDComponent) -> float:
        """Get per-class clutter intensity, falling back to global default."""
        profile = _get_motion_profile(component.object_class, self.dataset)
        if profile is not None and profile.clutter_intensity is not None:
            return profile.clutter_intensity
        return self.clutter_intensity

    def _speed_adaptive_turn_penalty(
        self,
        velocity_xy: np.ndarray,
        displacement_xy: np.ndarray,
        base_weight: float,
    ) -> float:
        if base_weight <= 0.0:
            return 0.0
        v = np.asarray(velocity_xy, dtype=np.float32).reshape(-1)
        d = np.asarray(displacement_xy, dtype=np.float32).reshape(-1)
        if v.size < 2 or d.size < 2:
            return 0.0

        speed = float(np.linalg.norm(v[:2]))
        disp = float(np.linalg.norm(d[:2]))
        if disp < TURN_PENALTY_MIN_DISP:
            return 0.0

        denom = max(TURN_PENALTY_V_HIGH - TURN_PENALTY_V_LOW, 1e-6)
        speed_factor = float(np.clip((speed - TURN_PENALTY_V_LOW) / denom, 0.0, 1.0))
        if speed_factor <= 0.0:
            return 0.0

        cos_theta = float(np.dot(v[:2], d[:2]) / (speed * disp + 1e-8))
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        turn_term = max(0.0, TURN_PENALTY_COS_SOFT - cos_theta)
        if turn_term <= 0.0:
            return 0.0

        penalty = base_weight * speed_factor * turn_term
        return float(min(penalty, TURN_PENALTY_MAX))

    def predict(self, new_targets: Optional[List[Dict]] = None):
        """
        Combined prediction + birth step (correct PHD semantics).

        Args:
            new_targets: Optional list of new targets from fusion stage (for birth)
        """
        new_components: List[GMPHDComponent] = []

        # PART 1: Survival/Motion Prediction
        for component in self.components:
            # Get class-specific motion profile (falls back to hardcoded if unavailable)
            profile = _get_motion_profile(component.object_class, self.dataset)
            n_modes = profile.num_modes if profile else 3
            kept_any_mode = False

            # Transition through all possible next modes
            for next_mode in range(n_modes):
                # Get motion model for this mode (class-based or legacy)
                if profile is not None:
                    F_base = profile.get_F(next_mode)
                    Q_base = profile.get_Q(next_mode)
                    mode_transition_prob = profile.transition_matrix[component.motion_mode, next_mode]
                    survival_prob = profile.survival_probability
                else:
                    # use hardcoded MOTION_MODES
                    mode_name = ["stationary", "constant_velocity", "maneuvering"][next_mode]
                    F_base = MOTION_MODELS[mode_name]["F"]
                    Q_base = MOTION_MODELS[mode_name]["Q"]
                    mode_transition_prob = TRANSITION_MATRIX[component.motion_mode, next_mode]
                    survival_prob = self.survival_prob

                F = F_base.copy()
                F[0, 2] = F_base[0, 2] * self.dt 
                F[1, 3] = F_base[1, 3] * self.dt 
                Q = Q_base.copy()
                Q[:2, :2] *= self.dt ** 2  
                Q[2:, 2:] *= self.dt       
                Q *= self.process_noise_scale  

                predicted_mean = F @ component.mean
                predicted_cov = F @ component.covariance @ F.T + Q
                predicted_cov = self._clamp_state_covariance(predicted_cov)

                predicted_weight = (
                    survival_prob
                    * mode_transition_prob
                    * component.weight
                )

                if mode_transition_prob < 0.05:
                    continue
                
                keep_component = (
                    component.track_state in (TrackState.CONFIRMED, TrackState.LOST)
                    or predicted_weight >= WEIGHT_PRUNE_THRESHOLD
                )
                if keep_component:
                    new_comp = GMPHDComponent(
                        weight=predicted_weight,
                        mean=predicted_mean,
                        covariance=predicted_cov,
                        identity=component.identity,  
                        motion_mode=next_mode,
                        object_class=component.object_class,  
                        feature=component.feature,  
                        track_state=component.track_state,  
                        hit_count=component.hit_count,
                        miss_count=component.miss_count,
                    )
                    new_components.append(new_comp)
                    kept_any_mode = True
                

        self.components = new_components

        # PART 2: Birth Intensity (if new targets provided)
        if new_targets is not None:
            self._add_birth_components(new_targets)

    def birth(self, new_targets: List[Dict]):
        self._add_birth_components(new_targets)

    def _add_birth_components(self, new_targets: List[Dict]):
        # Default birth parameters (used when no class-based profile available)
        DEFAULT_BIRTH_WEIGHT_THRESHOLD = 0.65
        DEFAULT_INITIAL_VELOCITY_STD = 1.0  # σ_v = 1.0 m/s

        # Get existing identities in PHD
        existing_ids = {comp.identity for comp in self.components if comp.identity is not None}

        # Birth-attempt bookkeeping

        birth_count = 0
        for target in new_targets:
            identity = target.get("identity")
            confidence = target.get("confidence", 0.5)
            obj_class = target.get("object_class", "pedestrian")

            # Get class-specific parameters
            profile = _get_motion_profile(obj_class, self.dataset)
            if profile is not None:
                birth_threshold = profile.birth_weight_threshold * self.birth_threshold_scale
                velocity_std = profile.initial_velocity_std
                n_modes = profile.num_modes
            else:
                birth_threshold = DEFAULT_BIRTH_WEIGHT_THRESHOLD * self.birth_threshold_scale
                velocity_std = DEFAULT_INITIAL_VELOCITY_STD
                n_modes = 3

            # Evaluate each target
            skip_reason = None
            if confidence < birth_threshold:
                skip_reason = f"low_conf({confidence:.3f}<{birth_threshold})"
            elif identity is not None and identity in existing_ids:
                skip_reason = f"existing_id({identity})"

            if skip_reason:
                continue
            else:
                birth_count += 1

            # Skip low-confidence or existing targets
            if confidence < birth_threshold:
                continue
            if identity is not None and identity in existing_ids:
                continue

            z = target["z_bev"]  # [2,]
            R = target["R_bev"]  # [2,2]

            det_vel = target.get("velocity")
            if det_vel is not None:
                vx, vy = float(det_vel[0]), float(det_vel[1])
                birth_state = np.array([z[0], z[1], vx, vy], dtype=np.float32)
                # Reduce velocity uncertainty when initialized from detector prediction
                vel_init_std = velocity_std * 0.5
            else:
                birth_state = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float32)
                vel_init_std = velocity_std

            birth_cov = np.eye(4, dtype=np.float32)
            birth_cov[:2, :2] = R
            birth_cov[2:, 2:] = vel_init_std ** 2 * np.eye(2)

            if self.use_class_birth_mode_priors:
                profile = _get_motion_profile(obj_class, self.dataset)
                if profile is not None and profile.transition_matrix is not None:
                    tm = profile.transition_matrix
                else:
                    tm = TRANSITION_MATRIX
            else:
                tm = TRANSITION_MATRIX
            mode_weights = np.array([tm[0, m] for m in range(n_modes)], dtype=np.float32)
            mode_weights /= mode_weights.sum()
            # Propagate pooled multi-view feature (if provided by fusion stage)
            birth_feature = target.get("feature", None)

            for mode in range(n_modes):
                component = GMPHDComponent(
                    weight=confidence * mode_weights[mode],  # Weighted by HMM prior
                    mean=birth_state,
                    covariance=birth_cov,
                    identity=identity,
                    motion_mode=mode,
                    object_class=obj_class,
                    feature=birth_feature.copy() if birth_feature is not None else None,
                    track_state=TrackState.TENTATIVE,  # Born as tentative
                    hit_count=0,  # Birth does NOT count as hit - must be confirmed by measurement
                    miss_count=0,
                )
                self.components.append(component)

        # Birth summary

    def update(self, measurements: List[Dict], camera_positions: Optional[Dict[int, np.ndarray]] = None):
        """
        Update step with HUNGARIAN-GATED association for confirmed tracks.

        Args:
            measurements: List of dicts with keys:
                - 'z': [2,] BEV position measurement
                - 'R': [2,2] measurement covariance
                - 'identity_probs': Dict[int, float] - P(ID_k | measurement) from front-end
            camera_positions: Optional dict mapping camera_id → BEV position [2,]
        """
        if not measurements or not self.components:
            return

        M = len(measurements)
        J = len(self.components)

        # Observation matrix: observe position only [x, y] from [x, y, vx, vy]
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        # Pre-extract all measurement data
        z_meas = np.array([m["z"] for m in measurements], dtype=np.float32)  # [M, 2]
        R_meas = np.array([m["R"] for m in measurements], dtype=np.float32)  # [M, 2, 2]

        # Pre-extract all component predicted measurements
        z_preds = np.array([H @ c.mean for c in self.components], dtype=np.float32)  # [J, 2]
        P_obs = np.array([H @ c.covariance @ H.T for c in self.components], dtype=np.float32)  # [J, 2, 2]

        # Compute Mahalanobis distances for cost matrix
        mahal_sq = np.full((M, J), np.inf, dtype=np.float32)
        log_det_S = np.full((M, J), np.inf, dtype=np.float32)
        for i in range(M):
            for j in range(J):
                S = P_obs[j] + R_meas[i]
                diff = z_meas[i] - z_preds[j]
                try:
                    sign, logdet = np.linalg.slogdet(S)
                    if sign > 0 and np.isfinite(logdet) and logdet > -50.0:
                        S_inv_diff = np.linalg.solve(S, diff)
                        mahal_sq[i, j] = float(diff @ S_inv_diff)
                        log_det_S[i, j] = float(logdet)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    # Leave default +inf costs for numerically unstable covariance sums.
                    continue

        # =========================================================================
        # HUNGARIAN ASSIGNMENT for CONFIRMED tracks (prevents weight dilution)
        # =========================================================================
        confirmed_indices = [j for j, c in enumerate(self.components) if c.track_state == TrackState.CONFIRMED]
        tentative_indices = [j for j, c in enumerate(self.components) if c.track_state != TrackState.CONFIRMED]

        # Build assignment for confirmed tracks only
        hungarian_matches = {}  # track_idx -> measurement_idx
        matched_measurements = set()
        chosen_cols_by_local: Dict[int, int] = {}

        cost_matrix = None
        miss_cost = None
        global_to_local_idx = {}
        turn_penalty_real = None
        nll_real = None
        identity_prob_real = None
        identity_bonus_real = None
        semantic_sim_real = None
        semantic_bonus_real = None
        if confirmed_indices:
            cost_base = 0.5 * (mahal_sq + log_det_S)  # [M, J]
            cost_real = cost_base[:, confirmed_indices].T.copy()  # [n_confirmed, M]
            for local_j, global_j in enumerate(confirmed_indices):
                p_d_j = float(np.clip(self._get_class_detection_prob(self.components[global_j]), 1e-6, 1.0 - 1e-6))
                cost_real[local_j, :] -= np.log(p_d_j)
            nll_real = cost_real.copy()

            # Gate by Mahalanobis distance
            gate_thresh = self.mahal_thresh  # chi2(2, 0.999) = 13.82
            gate_mask = mahal_sq[:, confirmed_indices].T > gate_thresh
            cost_real[gate_mask] = 1e6
            nll_real[gate_mask] = 1e6
            global_to_local_idx = {global_j: local_j for local_j, global_j in enumerate(confirmed_indices)}

            # Add identity boost + semantic feature similarity to cost (lower cost = better match)
            identity_prob_real = np.zeros_like(cost_real, dtype=np.float32)
            identity_bonus_real = np.zeros_like(cost_real, dtype=np.float32)
            semantic_sim_real = np.zeros_like(cost_real, dtype=np.float32)
            semantic_bonus_real = np.zeros_like(cost_real, dtype=np.float32)
            turn_penalty_real = np.zeros_like(cost_real, dtype=np.float32)
            for local_j, global_j in enumerate(confirmed_indices):
                comp = self.components[global_j]
                if comp.identity is not None:
                    for i, meas in enumerate(measurements):
                        identity_probs = meas.get("identity_probs", {})
                        if comp.identity in identity_probs:
                            id_prob = float(identity_probs[comp.identity])
                            id_bonus = float(self.identity_assoc_boost * id_prob)
                            # Identity match reduces cost
                            cost_real[local_j, i] -= id_bonus
                            identity_prob_real[local_j, i] = id_prob
                            identity_bonus_real[local_j, i] = id_bonus

                # Semantic feature similarity boost
                if comp.feature is not None:
                    comp_norm = comp.feature / (np.linalg.norm(comp.feature) + 1e-8)
                    for i, meas in enumerate(measurements):
                        meas_feat = meas.get("feature")
                        if meas_feat is not None and np.any(meas_feat != 0):
                            meas_norm = meas_feat / (np.linalg.norm(meas_feat) + 1e-8)
                            sem_sim = float(max(0.0, np.dot(comp_norm, meas_norm)))
                            sem_bonus = float(SEMANTIC_COST_BOOST * sem_sim)
                            # Semantic similarity reduces cost (high sim = likely same person)
                            cost_real[local_j, i] -= sem_bonus
                            semantic_sim_real[local_j, i] = sem_sim
                            semantic_bonus_real[local_j, i] = sem_bonus

                comp_vel = comp.mean[2:4]
                for i in range(M):
                    if cost_real[local_j, i] >= 1e5:
                        continue
                    prev_pos_est = z_preds[global_j] - comp_vel * self.dt
                    disp = z_meas[i] - prev_pos_est
                    turn_pen = self._speed_adaptive_turn_penalty(
                        comp_vel,
                        disp,
                        TURN_PENALTY_BASE,
                    )
                    if turn_pen > 0.0:
                        cost_real[local_j, i] += turn_pen
                        turn_penalty_real[local_j, i] = float(turn_pen)

            # ── Track-level Hungarian ──────────────────────────────────────
            _track_groups: Dict[int, list] = _defaultdict(list)  # identity → [(local_j, global_j)]
            for local_j, global_j in enumerate(confirmed_indices):
                tid = self.components[global_j].identity
                key = tid if tid is not None else -global_j  # unique key for id-less components
                _track_groups[key].append((local_j, global_j))

            _track_keys = list(_track_groups.keys())
            n_tracks = len(_track_keys)

            # Track-level cost: min across modes for each measurement
            _track_cost = np.full((n_tracks, M), 1e6, dtype=np.float32)
            for t_idx, key in enumerate(_track_keys):
                for local_j, _gj in _track_groups[key]:
                    for i in range(M):
                        if cost_real[local_j, i] < _track_cost[t_idx, i]:
                            _track_cost[t_idx, i] = cost_real[local_j, i]

            # Track-level miss cost: use highest-weight component's p_D
            _track_dummy = np.full((n_tracks, n_tracks), 1e6, dtype=np.float32)
            for t_idx, key in enumerate(_track_keys):
                modes = _track_groups[key]
                best_gj = max(modes, key=lambda m: self.components[m[1]].weight)[1]
                p_d = float(np.clip(self._get_class_detection_prob(self.components[best_gj]), 1e-6, 1.0 - 1e-6))
                _track_dummy[t_idx, t_idx] = float(-np.log(1.0 - p_d))

            miss_cost = float(np.mean(np.diag(_track_dummy)[:n_tracks])) if n_tracks > 0 else 2.303
            _track_full = np.concatenate([_track_cost, _track_dummy], axis=1)

            # Run Hungarian at track level (much smaller matrix than component level)
            row_ind, col_ind = linear_sum_assignment(_track_full)

            # Map back: all modes of a matched track get the same measurement
            for t_row, t_col in zip(row_ind, col_ind):
                key = _track_keys[t_row]
                modes = _track_groups[key]
                if t_col < M and _track_full[t_row, t_col] < 1e5:
                    for local_j, global_j in modes:
                        hungarian_matches[global_j] = int(t_col)
                        chosen_cols_by_local[local_j] = int(t_col)
                    matched_measurements.add(int(t_col))
                else:
                    for local_j, global_j in modes:
                        chosen_cols_by_local[local_j] = M + local_j  # dummy col

            # Component-level cost_matrix (kept at original shape)
            n_conf = len(confirmed_indices)
            dummy_block = np.full((n_conf, n_conf), 1e6, dtype=np.float32)
            for local_j, global_j in enumerate(confirmed_indices):
                p_d_j = float(np.clip(self._get_class_detection_prob(self.components[global_j]), 1e-6, 1.0 - 1e-6))
                dummy_block[local_j, local_j] = float(-np.log(1.0 - p_d_j))
            cost_matrix = np.concatenate([cost_real, dummy_block], axis=1)

        gated_pair_count = 0
        if confirmed_indices:
            gated_pair_count = int(np.sum(mahal_sq[:, confirmed_indices] < self.mahal_thresh))
        unmatched_confirmed = [j for j in confirmed_indices if j not in hungarian_matches]
        meas_to_track = {meas_idx: track_idx for track_idx, meas_idx in hungarian_matches.items()}
        risky_matches = [
            (j, i, float(mahal_sq[i, j]))
            for j, i in hungarian_matches.items()
            if float(mahal_sq[i, j]) > 6.0
        ]
        # Measurements not claimed by confirmed-track Hungarian pass.
        # Used by second-pass (tentative/lost) association.
        unmatched_meas_indices = [i for i in range(M) if i not in matched_measurements]

        # =========================================================================
        # UPDATE CONFIRMED TRACKS (Hungarian-gated)
        # =========================================================================
        updated_components = []

        for j in confirmed_indices:
            component = self.components[j]
            p_D_j = self._get_class_detection_prob(component)

            if j in hungarian_matches:
                # MATCHED: Full Kalman update with weight boost
                i = hungarian_matches[j]
                z = z_meas[i]
                R = R_meas[i]

                # Kalman update
                S = P_obs[j] + R
                try:
                    PHt = component.covariance @ H.T
                    K = np.linalg.solve(S.T, PHt.T).T
                except np.linalg.LinAlgError:
                    K = np.zeros((4, 2), dtype=np.float32)

                innovation = z - z_preds[j]
                new_mean = component.mean + K @ innovation

                # Joseph form covariance update
                I_KH = np.eye(4, dtype=np.float32) - K @ H
                new_cov = I_KH @ component.covariance @ I_KH.T + K @ R @ K.T
                new_cov = self._clamp_state_covariance(new_cov)

                # Weight boost for matched confirmed track (prevents weight collapse)
                new_weight = min(component.weight + MATCHED_CONFIRMED_WEIGHT_BOOST, 1.0)

                updated_components.append(GMPHDComponent(
                    weight=new_weight,
                    mean=new_mean,
                    covariance=new_cov,
                    identity=component.identity,
                    motion_mode=component.motion_mode,
                    object_class=component.object_class,
                    feature=component.feature,
                    track_state=TrackState.CONFIRMED,
                    hit_count=component.hit_count + 1,
                    miss_count=0,
                ))
            else:
                # UNMATCHED: Apply miss penalty but retain confirmed tracks.
                # Lifecycle transitions (CONFIRMED->LOST->delete) should control removal,
                # not weight pruning inside update().
                new_weight = (1.0 - p_D_j) * component.weight
                updated_components.append(GMPHDComponent(
                    weight=new_weight,
                    mean=component.mean.copy(),
                    covariance=component.covariance.copy(),
                    identity=component.identity,
                    motion_mode=component.motion_mode,
                    object_class=component.object_class,
                    feature=component.feature,
                    track_state=component.track_state,
                    hit_count=max(0, component.hit_count - 1),
                    miss_count=component.miss_count + 1,
                ))

        # =========================================================================
        # UPDATE TENTATIVE/LOST TRACKS (soft PHD association for fair competition)
        # =========================================================================
        tentative_lost_associated = 0
        lost_reassoc = 0

        for j in tentative_indices:
            component = self.components[j]
            p_D_j = self._get_class_detection_prob(component)
            q_sum = 0.0
            best_meas_idx = None
            best_score = -np.inf  # Combined spatial + semantic score
            best_turn_penalty = 0.0
            gated_unmatched_count = 0

            # Compute soft association with UNMATCHED measurements only
            if unmatched_meas_indices:
                for i in unmatched_meas_indices:
                    if mahal_sq[i, j] < self.mahal_thresh:  # Gated
                        gated_unmatched_count += 1
                        turn_penalty = 0.0

                        # Soft association evidence: downweight implausible turn proposals.
                        likelihood = np.exp(-0.5 * mahal_sq[i, j]) * np.exp(-turn_penalty)
                        q_sum += float(likelihood)

                        # Combined score: lower Mahalanobis + semantic similarity boost
                        score = -mahal_sq[i, j] - turn_penalty  # Spatial + heading consistency

                        # Semantic re-ID boost for LOST tracks
                        if (component.track_state == TrackState.LOST and
                                component.feature is not None):
                            meas_feat = measurements[i].get("feature")
                            if meas_feat is not None and np.any(meas_feat != 0):
                                comp_feat = component.feature
                                comp_norm = comp_feat / (np.linalg.norm(comp_feat) + 1e-8)
                                meas_norm = meas_feat / (np.linalg.norm(meas_feat) + 1e-8)
                                sem_sim = float(max(0.0, np.dot(comp_norm, meas_norm)))
                                # Semantic boost: high similarity nudges re-ID, but should not dominate geometry.
                                score += LOST_TRACK_REID_SEMANTIC_BOOST * sem_sim

                        if score > best_score:
                            best_score = score
                            best_meas_idx = i
                            best_turn_penalty = float(turn_penalty)

                # Weight update
                new_weight = (1.0 - p_D_j) * component.weight + 0.1 * q_sum
                new_weight = min(new_weight, 1.0)

                # State update using best matching measurement
                new_mean = component.mean.copy()
                new_cov = component.covariance.copy()

                if best_meas_idx is not None and mahal_sq[best_meas_idx, j] < 4.61:  # chi2(2, 0.90)
                    z = z_meas[best_meas_idx]
                    R = R_meas[best_meas_idx]
                    S = P_obs[j] + R
                    try:
                        PHt = component.covariance @ H.T
                        K = np.linalg.solve(S.T, PHt.T).T
                        innovation = z - z_preds[j]
                        new_mean = component.mean + K @ innovation
                        I_KH = np.eye(4, dtype=np.float32) - K @ H
                        new_cov = I_KH @ component.covariance @ I_KH.T + K @ R @ K.T
                        new_cov = self._clamp_state_covariance(new_cov)
                    except np.linalg.LinAlgError:
                        pass

                was_associated = q_sum > HIT_ASSOCIATION_THRESHOLD
            else:
                # No unmatched measurements available
                new_weight = (1.0 - p_D_j) * component.weight
                new_mean = component.mean.copy()
                new_cov = component.covariance.copy()
                was_associated = False

            if was_associated:
                tentative_lost_associated += 1
                if component.track_state == TrackState.LOST:
                    lost_reassoc += 1

            # Keep LOST tracks regardless of weight so they can survive up to
            # MAX_LOST_FRAMES and be re-identified.
            keep_component = (
                component.track_state == TrackState.LOST
                or new_weight >= WEIGHT_PRUNE_THRESHOLD
            )
            if keep_component:
                # Update feature if LOST track re-associated with a measurement
                updated_feature = component.feature
                if was_associated and best_meas_idx is not None and component.track_state == TrackState.LOST:
                    meas_feat = measurements[best_meas_idx].get("feature")
                    if meas_feat is not None and np.any(meas_feat != 0):
                        updated_feature = meas_feat.copy()

                updated_components.append(GMPHDComponent(
                    weight=new_weight,
                    mean=new_mean,
                    covariance=new_cov,
                    identity=component.identity,
                    motion_mode=component.motion_mode,
                    object_class=component.object_class,
                    feature=updated_feature,
                    track_state=component.track_state,
                    hit_count=component.hit_count + 1 if was_associated else max(0, component.hit_count - 1),
                    miss_count=0 if was_associated else component.miss_count + 1,
                ))

        self.components = updated_components

    def _compute_los_detection_probability(
        self, camera_positions: Dict[int, np.ndarray]
    ) -> np.ndarray:
        J = len(self.components)
        if J == 0 or not camera_positions:
            return np.array(
                [self._get_class_detection_prob(c) for c in self.components],
                dtype=np.float32,
            ) if J > 0 else np.empty(0, dtype=np.float32)

        positions = np.array([c.mean[:2] for c in self.components], dtype=np.float32)
        confirmed_mask = np.array([
            c.track_state == TrackState.CONFIRMED for c in self.components
        ])

        cam_ids = sorted(camera_positions.keys())
        n_cams = len(cam_ids)
        visible_count = np.zeros(J, dtype=np.float32)

        for cam_id in cam_ids:
            cam_pos = np.asarray(camera_positions[cam_id][:2], dtype=np.float32)

            for j in range(J):
                if self.components[j].track_state not in (
                    TrackState.CONFIRMED, TrackState.LOST
                ):
                    # Non-tracked components (tentative) get full visibility
                    visible_count[j] += 1
                    continue

                track_pos = positions[j]
                ray = track_pos - cam_pos
                dist_to_track = float(np.linalg.norm(ray))
                if dist_to_track < 1e-6:
                    visible_count[j] += 1
                    continue
                ray_unit = ray / dist_to_track

                occluded = False
                for k in range(J):
                    if k == j or not confirmed_mask[k]:
                        continue
                    occluder_ray = positions[k] - cam_pos
                    dist_to_occluder = float(np.linalg.norm(occluder_ray))

                    # Occluder must be closer to camera than target
                    if dist_to_occluder >= dist_to_track:
                        continue

                    # Project occluder onto LoS ray
                    proj = float(np.dot(occluder_ray, ray_unit))
                    if proj < 0:  # Behind camera
                        continue

                    # Perpendicular distance from occluder to LoS ray
                    perp_vec = occluder_ray - proj * ray_unit
                    perp_dist = float(np.linalg.norm(perp_vec))

                    if perp_dist < SHADOW_RADIUS:
                        occluded = True
                        break

                if not occluded:
                    visible_count[j] += 1.0

        # p_D = base_p_D × max(visibility_ratio, MIN_VISIBILITY_RATIO)
        # Use per-class base detection probability
        p_D = np.array(
            [self._get_class_detection_prob(c) for c in self.components],
            dtype=np.float32,
        )
        for j in range(J):
            if self.components[j].track_state in (TrackState.CONFIRMED, TrackState.LOST):
                vis_ratio = visible_count[j] / max(n_cams, 1)
                p_D[j] = p_D[j] * max(vis_ratio, MIN_VISIBILITY_RATIO)

        return p_D

    def _update_clutter_estimate(
        self,
        n_measurements: int,
        likelihoods: np.ndarray,
        weights: np.ndarray,
        p_D_per_component: np.ndarray,
    ):
        if n_measurements == 0:
            return

        # Compute per-measurement max association strength
        # association_strength[i] = max_j(p_D_j * w_j * likelihood[i,j])
        if likelihoods.shape[1] > 0:
            weighted_likelihoods = likelihoods * (p_D_per_component * weights)[None, :]
            max_assoc = weighted_likelihoods.max(axis=1)
        else:
            max_assoc = np.zeros(n_measurements)

        # Measurements with low max association are likely clutter
        CLUTTER_ASSOC_THRESHOLD = 0.01
        n_clutter = int(np.sum(max_assoc < CLUTTER_ASSOC_THRESHOLD))
        clutter_fraction = n_clutter / n_measurements

        # Adaptive κ: scale base intensity by clutter fraction
        # More clutter → higher κ → measurements absorbed as noise instead of birthing ghosts
        # Less clutter → lower κ → measurements contribute more to track updates/births
        measured_kappa = self._base_clutter_intensity * (1.0 + 2.0 * clutter_fraction)
        measured_kappa = float(np.clip(measured_kappa, 0.1, 2.0))

        # EMA update
        self.clutter_intensity = (
            (1 - self._clutter_ema_alpha) * self.clutter_intensity
            + self._clutter_ema_alpha * measured_kappa
        )

    def manage_components(self):
        """
        Component management: pruning, merging, mode selection, capping.
        """
        # Step 1: Pruning
        # Keep CONFIRMED/LOST components regardless of weight; lifecycle transitions
        # control their removal. Otherwise, low weights can delete tracks before
        # LOST recovery gets a chance.
        pruned = []
        for comp in self.components:
            if comp.track_state in (TrackState.CONFIRMED, TrackState.LOST):
                pruned.append(comp)
            elif comp.weight >= WEIGHT_PRUNE_THRESHOLD:
                pruned.append(comp)

        # Step 2: Identity-based and distance-based merging
        merged = []
        processed = set()

        for i, comp_i in enumerate(pruned):
            if i in processed:
                continue

            # Find a cluster of nearby components with the same identity
            to_merge_indices = [i]
            for j, comp_j in enumerate(pruned):
                if i == j or j in processed:
                    continue

                if comp_i.identity is not None and comp_i.identity == comp_j.identity:
                    # Check Mahalanobis distance on BEV position
                    delta_m = comp_i.mean[:2] - comp_j.mean[:2]
                    try:
                        P_pos = comp_i.covariance[:2, :2] + comp_j.covariance[:2, :2]
                        dist_sq = float(delta_m @ np.linalg.solve(P_pos, delta_m))
                    except np.linalg.LinAlgError:
                        dist_sq = float('inf')

                    if dist_sq < WEIGHT_MERGE_THRESHOLD:
                        to_merge_indices.append(j)

            # If a cluster of 2 or more components is found, merge them
            if len(to_merge_indices) > 1:
                total_weight = sum(pruned[j].weight for j in to_merge_indices)

                mean_num = np.sum(
                    [pruned[j].weight * pruned[j].mean for j in to_merge_indices], axis=0
                )
                merged_mean = mean_num / (total_weight + 1e-10)

                # PAPER EQ 49-50: P_merged = (1/w_merged) * Σ(w_i * (P_i + m_i*m_i^T)) - m_merged*m_merged^T
                cov_num = np.sum(
                    [pruned[j].weight * (pruned[j].covariance + np.outer(pruned[j].mean, pruned[j].mean)) for j in to_merge_indices],
                    axis=0
                )
                merged_cov = cov_num / (total_weight + 1e-10) - np.outer(merged_mean, merged_mean)
                merged_cov = self._clamp_state_covariance(merged_cov)

                # Find the mode of the highest-weight component in the cluster
                highest_weight_comp = max((pruned[j] for j in to_merge_indices), key=lambda c: c.weight)

                # Cap merged weight to preserve PHD intensity semantics
                capped_weight = min(total_weight, 1.5)

                merged_comp = GMPHDComponent(
                    weight=capped_weight,
                    mean=merged_mean,
                    covariance=merged_cov,
                    identity=comp_i.identity,
                    motion_mode=highest_weight_comp.motion_mode,
                    object_class=comp_i.object_class,
                    feature=highest_weight_comp.feature,  # Keep feature from highest-weight
                    track_state=highest_weight_comp.track_state,
                    hit_count=max(pruned[j].hit_count for j in to_merge_indices),
                    miss_count=min(pruned[j].miss_count for j in to_merge_indices),
                )
                merged.append(merged_comp)

                for j in to_merge_indices:
                    processed.add(j)
            else:
                # Not part of any cluster, so just add it
                merged.append(comp_i)
                processed.add(i)

        cross_merged = []
        cross_suppressed = set()
        weight_order = sorted(range(len(merged)), key=lambda i: merged[i].weight, reverse=True)
        for idx in weight_order:
            if idx in cross_suppressed:
                continue
            comp_i = merged[idx]
            cross_merged.append(comp_i)
            if comp_i.track_state != TrackState.CONFIRMED:
                continue
            if comp_i.identity is None:
                continue
            for jdx in weight_order:
                if jdx == idx or jdx in cross_suppressed:
                    continue
                comp_j = merged[jdx]
                if comp_j.track_state != TrackState.CONFIRMED:
                    continue
                # Only suppress if SAME identity (different identities may legitimately be close)
                if comp_j.identity != comp_i.identity:
                    continue
                dist = float(np.linalg.norm(comp_i.mean[:2] - comp_j.mean[:2]))
                if dist < 0.5:  # strict 0.5m hard gate
                    cross_suppressed.add(jdx)
        merged = cross_merged

        identity_to_comp = {}
        for comp in merged:
            if comp.identity is not None:
                if comp.identity not in identity_to_comp:
                    identity_to_comp[comp.identity] = comp
                else:
                    if comp.weight > identity_to_comp[comp.identity].weight:
                        identity_to_comp[comp.identity] = comp

        mode_selected = list(identity_to_comp.values())

        mode_selected.extend([c for c in merged if c.identity is None])

        if len(mode_selected) > MAX_COMPONENTS:
            mode_selected = sorted(mode_selected, key=lambda c: c.weight, reverse=True)[
                :MAX_COMPONENTS
            ]

        self.components = mode_selected

        self._update_track_states()

    def _update_track_states(self):
        """Apply track lifecycle state transitions.
        """
        surviving = []
        transition_counts = {
            "tent_to_conf": 0,
            "tent_deleted": 0,
            "conf_to_lost": 0,
            "lost_to_conf": 0,
            "lost_deleted": 0,
        }
        transition_examples = {
            "conf_to_lost": [],
            "lost_to_conf": [],
            "tent_deleted": [],
            "lost_deleted": [],
        }

        for comp in self.components:
            state = comp.track_state

            if state == TrackState.TENTATIVE:
                if comp.hit_count >= N_INIT:
                    # Promote to confirmed
                    comp.track_state = TrackState.CONFIRMED
                    transition_counts["tent_to_conf"] += 1
                    surviving.append(comp)
                elif comp.miss_count > self.tentative_miss_tolerance:
                    # False positive birth — delete
                    transition_counts["tent_deleted"] += 1
                    if len(transition_examples["tent_deleted"]) < ASSOC_DIAG_MAX_DETAILS:
                        transition_examples["tent_deleted"].append(comp.identity)
                    continue
                else:
                    # Still tentative, waiting for more hits
                    surviving.append(comp)

            elif state == TrackState.CONFIRMED:
                if comp.miss_count > self.confirmed_miss_tolerance:
                    comp.track_state = TrackState.LOST
                    comp.hit_count = 0
                    transition_counts["conf_to_lost"] += 1
                    if len(transition_examples["conf_to_lost"]) < ASSOC_DIAG_MAX_DETAILS:
                        transition_examples["conf_to_lost"].append(comp.identity)
                    surviving.append(comp)
                else:
                    # Still confirmed
                    surviving.append(comp)

            elif state == TrackState.LOST:
                if comp.hit_count >= 1:
                    # Re-identified! Promote back to confirmed
                    comp.track_state = TrackState.CONFIRMED
                    transition_counts["lost_to_conf"] += 1
                    if len(transition_examples["lost_to_conf"]) < ASSOC_DIAG_MAX_DETAILS:
                        transition_examples["lost_to_conf"].append(comp.identity)
                    surviving.append(comp)
                elif comp.miss_count > self.max_lost_frames:
                    # Too long without detection — delete
                    transition_counts["lost_deleted"] += 1
                    if len(transition_examples["lost_deleted"]) < ASSOC_DIAG_MAX_DETAILS:
                        transition_examples["lost_deleted"].append(comp.identity)
                    continue
                else:
                    # Still lost, keep predicting
                    surviving.append(comp)

            else:
                surviving.append(comp)

        # Log state transitions
        state_counts = {TrackState.TENTATIVE: 0, TrackState.CONFIRMED: 0, TrackState.LOST: 0}
        for comp in surviving:
            state_counts[comp.track_state] = state_counts.get(comp.track_state, 0) + 1
        deleted = len(self.components) - len(surviving)

        self.components = surviving

    def extract_tracks(self) -> List[Dict]:
        """
        Extract track estimates from PHD intensity.
        """
        tracks = []

        for component in self.components:
            # Extract all CONFIRMED tracks (lifecycle manages output, not weight)
            if component.track_state == TrackState.CONFIRMED:
                track = {
                    "position": component.mean[:2].copy(),
                    "velocity": component.mean[2:].copy(),
                    "covariance": component.covariance.copy(),
                    "identity": component.identity,
                    "weight": component.weight,
                    "motion_mode": component.motion_mode,
                    "object_class": component.object_class,
                }
                tracks.append(track)

        self.extracted_tracks = tracks
        return tracks

    def get_existing_track_info(self, weight_threshold: float = 0.15) -> Dict[int, Dict]:
        """
        Return existing track positions/IDs for identity matching in fusion stage.

        Args:
            weight_threshold: Minimum weight for non-LOST states.
                LOST states use LOST_POOL_WEIGHT_THRESHOLD.

        Returns:
            Dict mapping identity -> {pos, cov, weight} for matching
        """
        tracks = {}
        for comp in self.components:
            if comp.identity is None:
                continue
            effective_threshold = (
                LOST_POOL_WEIGHT_THRESHOLD
                if comp.track_state == TrackState.LOST
                else weight_threshold
            )
            if comp.weight >= effective_threshold:
                if comp.identity not in tracks or comp.weight > tracks[comp.identity]["weight"]:
                    tracks[comp.identity] = {
                        "pos": comp.mean[:2].copy(),
                        "cov": comp.covariance[:2, :2].copy(),
                        "weight": comp.weight,
                        "feature": comp.feature.copy() if comp.feature is not None else None,
                        "track_state": comp.track_state,
                    }
        return tracks

    def expected_number_of_targets(self) -> float:
        return float(np.sum([c.weight for c in self.components]))

# Define motion models dict for easier access
MOTION_MODELS = {
    "stationary": MOTION_MODES["stationary"],
    "constant_velocity": MOTION_MODES["constant_velocity"],
    "maneuvering": MOTION_MODES["maneuvering"],
}
