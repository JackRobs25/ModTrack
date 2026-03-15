"""Radar MOS NN helpers used by the RadarScenes adapter.

This module intentionally includes only inference-time utilities needed by
`modtrack.data.adapters.radarscenes`:
- k-NN feature construction
- multi-task model class factory for checkpoint loading
"""

from __future__ import annotations

import numpy as np


def compute_knn_features(
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    vr_comp: np.ndarray,
    rcs: np.ndarray,
    range_sc: np.ndarray,
    timestamps: np.ndarray,
    k: int = 16,
    dt_us: float = 50_000.0,
) -> np.ndarray:
    """Compute neighborhood statistics per radar point."""
    from scipy.spatial import cKDTree

    n_points = len(x_seq)
    features = np.zeros((n_points, 12), dtype=np.float32)

    t0 = timestamps.min() if len(timestamps) else 0.0
    frame_ids = ((timestamps - t0) / dt_us).astype(np.int64) if len(timestamps) else np.zeros(0, dtype=np.int64)
    unique_frames = np.unique(frame_ids)

    for frame_id in unique_frames:
        mask = frame_ids == frame_id
        indices = np.where(mask)[0]
        n_frame = len(indices)
        if n_frame < 2:
            continue

        xy = np.column_stack([x_seq[indices], y_seq[indices]])
        tree = cKDTree(xy)
        k_query = min(k + 1, n_frame)
        dists, nbr_local = tree.query(xy, k=k_query)

        nbr_dists = dists[:, 1:]
        nbr_idx_local = nbr_local[:, 1:]
        n_nbrs = k_query - 1
        if n_nbrs == 0:
            continue

        nbr_idx_global = indices[nbr_idx_local]
        nbr_vr = vr_comp[nbr_idx_global]
        nbr_rcs = rcs[nbr_idx_global]
        nbr_range = range_sc[nbr_idx_global]
        self_vr = vr_comp[indices]

        features[indices, 0] = (nbr_dists <= 3.0).sum(axis=1) / max(float(k), 1.0)
        features[indices, 1] = nbr_vr.mean(axis=1)
        features[indices, 2] = nbr_vr.std(axis=1) if n_nbrs > 1 else 0.0
        features[indices, 3] = nbr_rcs.mean(axis=1)
        features[indices, 4] = nbr_rcs.std(axis=1) if n_nbrs > 1 else 0.0
        features[indices, 5] = nbr_range.mean(axis=1)
        features[indices, 6] = nbr_range.std(axis=1) if n_nbrs > 1 else 0.0
        features[indices, 7] = np.abs(self_vr[:, None] - nbr_vr).mean(axis=1)
        features[indices, 8] = nbr_dists.mean(axis=1)
        features[indices, 9] = nbr_rcs.max(axis=1)
        features[indices, 10] = (np.abs(nbr_vr) > 0.5).mean(axis=1)

        frame_area = max(1.0, np.ptp(xy, axis=0).max() ** 2)
        frame_mean_density = n_frame / frame_area
        max_nbr_dist = nbr_dists.max(axis=1)
        local_area = np.maximum(1e-6, max_nbr_dist ** 2) * np.pi
        local_density = n_nbrs / local_area
        features[indices, 11] = local_density / max(1e-8, frame_mean_density)

    return features


def get_multitask_model_class():
    """Return the radar multi-task MLP class used for checkpoint loading."""
    import torch.nn as nn

    class RadarMultiTaskMLP(nn.Module):
        def __init__(self, input_dim=17, hidden1=64, hidden2=32, num_classes=5, dropout=0.3):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.mos_head = nn.Linear(hidden2, 1)
            self.class_head = nn.Linear(hidden2, num_classes)

        def forward(self, x):
            features = self.backbone(x)
            mos_logits = self.mos_head(features).squeeze(-1)
            class_logits = self.class_head(features)
            return mos_logits, class_logits

    return RadarMultiTaskMLP
