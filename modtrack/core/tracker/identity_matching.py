"""Dual-Modal Identity Matching for ModTrack
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.special import softmax


class DualModalMatcher:
    """Matches detections to identities using spatial and/or semantic modalities."""

    def __init__(
        self,
        matching_mode: str = "joint",
        sigma_spatial: float = 1.0,
        tau_geo: float = 5.0,
        tau_sem: float = 2.0,
        lambda_kl: float = 1.0,
        mahal_thresh: float = 9.21,
    ):
        """
        Initialize dual-modal matcher.

        Args:
            matching_mode: "spatial_only", "semantic_only", or "joint"
            sigma_spatial: Bandwidth parameter applied to the Mahalanobis-squared distance (Eq. 14)
            tau_geo: Temperature parameter for spatial softmax (Eq. 16)
            tau_sem: Temperature parameter for semantic softmax (Eq. 17)
            lambda_kl: KL divergence penalty weight (Eq. 22, joint mode only)
            mahal_thresh: Mahalanobis distance gate (χ²₂,₀.₉₉ = 9.21 for 99% confidence, 2 DOF)
        """
        assert matching_mode in [
            "spatial",
            "semantic",
            "joint",
        ], f"Invalid matching mode: {matching_mode}"

        self.matching_mode = matching_mode
        self.sigma_spatial = sigma_spatial
        self.tau_geo = tau_geo
        self.tau_sem = tau_sem
        self.lambda_kl = lambda_kl
        self.mahal_thresh = mahal_thresh

    def compute_spatial_similarity(
        self, det1: Dict, det2: Dict
    ) -> Tuple[float, float]:
        """
        Compute spatial similarity between two detections via Mahalanobis distance.

        Args:
            det1, det2: Detection dicts with 'z_bev' (BEV position) and 'R_bev' (covariance)

        Returns:
            (mahal_dist_sq, similarity_score)
        """
        z1 = det1["z_bev"]  # [2,] BEV position
        z2 = det2["z_bev"]
        R1 = det1["R_bev"]  # [2,2] covariance
        R2 = det2["R_bev"]

        delta_z = z2 - z1

        R_sum = R1 + R2

        try:
            inv_R = np.linalg.inv(R_sum)
            mahal_sq = float(delta_z @ inv_R @ delta_z)
        except np.linalg.LinAlgError:
            return np.inf, 0.0

        if mahal_sq > self.mahal_thresh:
            return mahal_sq, 0.0

        similarity = np.exp(-mahal_sq / (2.0 * self.sigma_spatial ** 2))

        return mahal_sq, similarity

    def compute_semantic_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute semantic similarity via cosine distance of feature vectors.

        Args:
            vec1, vec2: Feature vectors (typically 256-dim SWIN embeddings)

        Returns:
            Cosine similarity in [0, 1]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        # Cosine similarity
        sim = float((vec1 @ vec2) / (norm1 * norm2))

        # Clamp to [0, 1]
        return max(0.0, min(1.0, sim))

    def compute_kl_divergence(
        self, p: np.ndarray, q: np.ndarray
    ) -> float:
        """
        Compute KL divergence D_KL(p || q).

        Args:
            p, q: Probability distributions (should sum to 1)

        Returns:
            KL divergence value
        """
        # Ensure distributions are normalized
        p = p / (p.sum() + 1e-8)
        q = q / (q.sum() + 1e-8)

        # Avoid log(0) by clipping
        p = np.clip(p, 1e-8, 1.0)
        q = np.clip(q, 1e-8, 1.0)

        kl = float(np.sum(p * (np.log(p) - np.log(q))))

        # KL is non-negative by definition; warn if significantly negative (numerical issue)
        if kl < -1e-6:
            import warnings
            warnings.warn(
                f"Negative KL divergence ({kl:.6f}) indicates numerical instability. "
                f"p_sum={p.sum():.6f}, q_sum={q.sum():.6f}",
                RuntimeWarning,
                stacklevel=2
            )
        return max(0.0, kl)

    def compute_spatial_distribution(
        self,
        detection: Dict,
        existing_identities: Dict[int, List[Dict]],
    ) -> np.ndarray:
        """
        Compute probability distribution over identities using spatial matching only.

        Args:
            detection: Current detection
            existing_identities: Dict mapping identity ID -> list of recent detections

        Returns:
            Array of probabilities, one per identity
        """
        identity_ids = list(existing_identities.keys())

        if not identity_ids:
            return np.array([])

        scores = []
        for identity_id in identity_ids:
            recent_dets = existing_identities[identity_id]

            # Average spatial similarity to recent detections
            sims = []
            for recent_det in recent_dets:
                _, sim = self.compute_spatial_similarity(detection, recent_det)
                sims.append(sim)

            avg_sim = np.mean(sims) if sims else 0.0
            scores.append(avg_sim)

        # Softmax with temperature scaling
        # Use max(tau, 0.1) to prevent numerical explosion from very small temperatures
        scores = np.array(scores, dtype=np.float32)
        tau_safe = max(self.tau_geo, 0.1)
        probs = softmax(scores / tau_safe)

        return probs

    def compute_semantic_distribution(
        self,
        detection: Dict,
        existing_identities: Dict[int, List[Dict]],
    ) -> np.ndarray:
        """
        Compute probability distribution over identities using semantic matching only.

        Args:
            detection: Current detection (must have 'feature' key with SWIN embedding)
            existing_identities: Dict mapping identity ID -> list of recent detections

        Returns:
            Array of probabilities, one per identity
        """
        identity_ids = list(existing_identities.keys())

        if not identity_ids:
            return np.array([])

        if "feature" not in detection:
            # No feature available, return uniform distribution
            return np.ones(len(identity_ids)) / len(identity_ids)

        scores = []
        for identity_id in identity_ids:
            recent_dets = existing_identities[identity_id]

            # Average semantic similarity to recent detections
            sims = []
            for recent_det in recent_dets:
                if "feature" in recent_det:
                    sim = self.compute_semantic_similarity(
                        detection["feature"], recent_det["feature"]
                    )
                    sims.append(sim)

            avg_sim = np.mean(sims) if sims else 0.0
            scores.append(avg_sim)

        # Softmax with temperature scaling
        # Use max(tau, 0.1) to prevent numerical explosion from very small temperatures
        scores = np.array(scores, dtype=np.float32)
        tau_safe = max(self.tau_sem, 0.1)
        probs = softmax(scores / tau_safe)

        return probs

    def compute_joint_distribution(
        self,
        detection: Dict,
        existing_identities: Dict[int, List[Dict]],
    ) -> Tuple[np.ndarray, float]:
        """
        Compute probability distribution using joint spatial and semantic matching.

        Args:
            detection: Current detection
            existing_identities: Dict mapping identity ID -> list of recent detections

        Returns:
            (probability_distribution, kl_confidence)
            where kl_confidence = exp(-λ_KL * D_KL) ∈ (0, 1]
        """
        # Compute separate distributions
        q_spatial = self.compute_spatial_distribution(detection, existing_identities)
        q_sem = self.compute_semantic_distribution(detection, existing_identities)

        if len(q_spatial) == 0:
            return np.array([]), 1.0

        kl_per_id = q_spatial * np.log((q_spatial + 1e-8) / (q_sem + 1e-8))

        kl_div = float(np.sum(np.maximum(kl_per_id, 0)))  # KL is non-negative

        kl_confidence = np.exp(-self.lambda_kl * kl_div)

        q_avg = 0.5 * (q_spatial + q_sem)
        probs = q_avg * np.exp(-self.lambda_kl * np.abs(kl_per_id))

        probs = probs / (probs.sum() + 1e-8)

        return probs, float(kl_confidence)

    def match_detection_to_identities(
        self, detection: Dict, existing_identities: Dict[int, List[Dict]]
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Match a detection to existing identities using configured matching mode.

        Args:
            detection: Current detection dict
            existing_identities: Dict mapping identity ID -> list of recent detections

        Returns:
            (probability_distribution, kl_confidence or None)
        """
        if self.matching_mode == "spatial":
            probs = self.compute_spatial_distribution(detection, existing_identities)
            return probs, None

        elif self.matching_mode == "semantic":
            probs = self.compute_semantic_distribution(detection, existing_identities)
            return probs, None

        elif self.matching_mode == "joint":
            probs, kl_conf = self.compute_joint_distribution(
                detection, existing_identities
            )
            return probs, kl_conf

        else:
            raise ValueError(f"Unknown matching mode: {self.matching_mode}")
