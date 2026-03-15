# Calibration Analysis for ModTrack.

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import linear_sum_assignment


class CalibrationCollector:

    def __init__(self):
        self.records: List[Dict] = []

    @property
    def n_samples(self) -> int:
        return len(self.records)

    def add(
        self,
        pred_pos: np.ndarray,
        pred_cov: np.ndarray,
        gt_pos: np.ndarray,
        frame_idx: int = 0,
        track_id: Optional[int] = None,
    ):

        pred_pos = np.asarray(pred_pos, dtype=np.float64).flatten()[:2]
        gt_pos = np.asarray(gt_pos, dtype=np.float64).flatten()[:2]
        pred_cov = np.asarray(pred_cov, dtype=np.float64)

        # Extract 2x2 position covariance if 4x4 state covariance provided
        if pred_cov.shape == (4, 4):
            pred_cov = pred_cov[:2, :2]

        assert pred_pos.shape == (2,), f"pred_pos shape: {pred_pos.shape}"
        assert gt_pos.shape == (2,), f"gt_pos shape: {gt_pos.shape}"
        assert pred_cov.shape == (2, 2), f"pred_cov shape: {pred_cov.shape}"

        # Compute error and NEES
        error = gt_pos - pred_pos
        error_norm = float(np.linalg.norm(error))

        try:
            cov_inv = np.linalg.inv(pred_cov)
            nees = float(error @ cov_inv @ error)
        except np.linalg.LinAlgError:
            nees = float("inf")

        # Predicted standard deviation: marginal std = sqrt(trace(P)/2)
        pred_std_marginal = float(np.sqrt(np.trace(pred_cov) / 2.0))

        self.records.append({
            "pred_pos": pred_pos,
            "gt_pos": gt_pos,
            "pred_cov": pred_cov,
            "error": error,
            "error_norm": error_norm,
            "nees": nees,
            "pred_std_marginal": pred_std_marginal,
            "frame_idx": frame_idx,
            "track_id": track_id,
        })

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #

    def get_nees_values(self) -> np.ndarray:
        return np.array([r["nees"] for r in self.records if np.isfinite(r["nees"])])

    def get_errors(self) -> np.ndarray:
        return np.array([r["error_norm"] for r in self.records])

    def get_pred_stds(self) -> np.ndarray:
        return np.array([r["pred_std_marginal"] for r in self.records])

    # ------------------------------------------------------------------ #
    #  NEES Statistics                                                    #
    # ------------------------------------------------------------------ #

    def compute_nees_statistics(self) -> Dict:
        """Compute NEES consistency statistics."""
        nees = self.get_nees_values()
        if len(nees) == 0:
            return {"error": "No valid NEES samples"}

        dof = 2  # 2D position
        alpha = 0.05

        mean_nees = float(np.mean(nees))
        median_nees = float(np.median(nees))

        # 95% chi-squared acceptance interval
        chi2_lower = scipy_stats.chi2.ppf(0.025, dof)  # ~0.051
        chi2_upper = scipy_stats.chi2.ppf(0.975, dof)  # ~7.378
        pct_in_95 = float(np.mean((nees >= chi2_lower) & (nees <= chi2_upper))) * 100

        # Kolmogorov-Smirnov test against chi-squared(2)
        ks_stat, ks_pvalue = scipy_stats.kstest(nees, lambda x: scipy_stats.chi2.cdf(x, dof))

        # Mean NEES consistency CI for N matched samples (total DOF = 2N)
        n = len(nees)
        total_dof = dof * n
        mean_nees_lower = float(scipy_stats.chi2.ppf(alpha / 2.0, total_dof) / n)
        mean_nees_upper = float(scipy_stats.chi2.ppf(1.0 - (alpha / 2.0), total_dof) / n)
        avg_nees_consistent = bool(mean_nees_lower <= mean_nees <= mean_nees_upper)

        # Primary verdict (CI-based 3-way classification)
        if mean_nees <= mean_nees_upper and mean_nees >= mean_nees_lower:
            interpretation = "CALIBRATED"
        elif mean_nees > mean_nees_upper:
            interpretation = "OVERCONFIDENT"
        else:
            interpretation = "CONSERVATIVE"

        return {
            "n_samples": int(len(nees)),
            "mean_nees": mean_nees,
            "expected_nees": float(dof),
            "median_nees": median_nees,
            "mean_nees_ci_alpha": float(alpha),
            "mean_nees_ci_95": [mean_nees_lower, mean_nees_upper],
            # Additional statistics (non-primary verdict criteria)
            "supporting_pct_in_chi2_95": pct_in_95,
            "supporting_ks_statistic": float(ks_stat),
            "supporting_ks_pvalue": float(ks_pvalue),
            # Backward compatibility keys
            "pct_in_chi2_95": pct_in_95,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "avg_nees_95_interval": [mean_nees_lower, mean_nees_upper],
            "avg_nees_consistent": avg_nees_consistent,
            "interpretation": interpretation,
        }


def match_tracks_to_gt(
    tracks: List[Dict],
    gt_world_pts: np.ndarray,
    max_distance: float = 1.0,
) -> List[tuple]:
    """Match predicted tracks to GT positions using Hungarian algorithm.

    Args:
        tracks: List of track dicts with "position" key ([2,] BEV)
        gt_world_pts: (N_gt, 2+) array of GT positions; uses first 2 columns
        max_distance: Gating threshold (meters)

    Returns:
        List of (track_idx, gt_idx) matched pairs within the gate.
    """
    if not tracks or len(gt_world_pts) == 0:
        return []

    gt_xy = np.asarray(gt_world_pts)[:, :2]
    n_pred = len(tracks)
    n_gt = len(gt_xy)

    cost = np.full((n_pred, n_gt), 1e6, dtype=np.float64)
    for i, track in enumerate(tracks):
        pred_pos = np.asarray(track["position"]).flatten()[:2]
        for j in range(n_gt):
            cost[i, j] = np.linalg.norm(pred_pos - gt_xy[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    matched = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < max_distance:
            matched.append((i, j))

    return matched
