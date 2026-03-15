# ModTrack: Multi-View Multi-Camera Tracking Pipeline

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import time

import cv2
import numpy as np
from scipy.stats import chi2

# Spatial probability gate: detections with P(same) below this are flagged in stats
SPATIAL_GATE = 0.01  # Matches default prob_thresh in graph_clustering

# Import configuration system for mode management
try:
    from .config import (
        ModTrackConfig,
        DEFAULT_TRACKER_CONFIG,
        MODE_SPATIAL_ONLY,
        MODE_JOINT,
    )
except ImportError:
    # Fallback for legacy imports
    DEFAULT_TRACKER_CONFIG = {
        "mode": "joint",
        "tau_geo": 5.0,
        "tau_sem": 2.0,
        "lambda_kl": 1.0,
        "mahal_thresh": 9.21,
        "sem_thresh": 0.6,
        "max_iter": 100,
        "min_support": 2,
        "alpha": 2.0,
        "tau_new": 0.3,
        "allow_single_cam": True,
        "spatial_only": False,
    }
    MODE_SPATIAL_ONLY = "spatial"
    MODE_JOINT = "joint"


def geometric_similarity(
    kp1: np.ndarray,
    desc1: np.ndarray,
    kp2: np.ndarray,
    desc2: np.ndarray,
    H: np.ndarray,
    reproj_thresh: float = 5.0,
) -> float:
    """
    Geometric similarity using precomputed ground-plane homography validation.

    Args:
        kp1, desc1: Keypoints and descriptors from camera 1
        kp2, desc2: Keypoints and descriptors from camera 2
        H: Precomputed ground-plane homography matrix (3x3) from camera 1 to camera 2
        reproj_thresh: Reprojection error threshold for consistency check (pixels)

    Returns:
        Normalized geometric similarity score in [0, 1]
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 4:
        return 0.0

    pts1 = np.float32([kp1[m.queryIdx] for m in good])
    pts2 = np.float32([kp2[m.trainIdx] for m in good])

    try:
        # PAPER: Validate against precomputed ground-plane homography
        # Homography consistency: H @ p1 should project near p2
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])  # Homogeneous coords
        pts2_proj = (H @ pts1_h.T).T  # Project via precomputed H
        pts2_proj = pts2_proj[:, :2] / (pts2_proj[:, 2:3] + 1e-8)  # Normalize

        # Compute reprojection errors
        errors = np.linalg.norm(pts2 - pts2_proj, axis=1)
        inliers = np.sum(errors < reproj_thresh)

        # Similarity = fraction of consistent keypoints
        return float(inliers) / np.sqrt(max(len(kp1) * len(kp2), 1))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return 0.0


def semantic_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Semantic similarity using cosine similarity of normalized vectors.
    """
    v1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
    v2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
    return float(max(0.0, np.dot(v1, v2)))


def _vectorized_mahalanobis_spatial(
    positions: np.ndarray,
    covariances: np.ndarray,
    cam_ids: np.ndarray,
) -> np.ndarray:
    """
    Vectorized computation of pairwise Mahalanobis distances (spatial only).

    Args:
        positions: [N, 2] array of BEV positions
        covariances: [N, 2, 2] array of covariance matrices
        cam_ids: [N,] array of camera IDs

    Returns:
        [N, N] matrix of Mahalanobis distances squared (inf for same-camera pairs)
    """
    N = len(positions)

    # Pairwise position differences: [N, N, 2]
    diff = positions[:, None, :] - positions[None, :, :]

    # Pairwise covariance sums: [N, N, 2, 2]
    cov_sum = covariances[:, None, :, :] + covariances[None, :, :, :]

    # Compute Mahalanobis distance for each pair
    # d² = diff @ inv(cov_sum) @ diff
    mahal_sq = np.full((N, N), np.inf, dtype=np.float64)

    for i in range(N):
        for j in range(i + 1, N):
            # Skip same-camera pairs
            if cam_ids[i] == cam_ids[j]:
                continue
            try:
                d_sq = diff[i, j] @ np.linalg.solve(cov_sum[i, j], diff[i, j])
                mahal_sq[i, j] = d_sq
                mahal_sq[j, i] = d_sq
            except np.linalg.LinAlgError:
                pass

    return mahal_sq


def _vectorized_semantic_similarity(
    features: np.ndarray,
    cam_ids: np.ndarray,
) -> np.ndarray:
    """
    Vectorized computation of pairwise cosine similarities.

    Args:
        features: [N, D] array of L2-normalized feature vectors (or None entries)
        cam_ids: [N,] array of camera IDs

    Returns:
        [N, N] matrix of cosine similarities (0 for same-camera or missing features)
    """
    N = len(features)

    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    features_norm = features / norms

    # Compute all pairwise cosine similarities: [N, N]
    sim_matrix = features_norm @ features_norm.T

    # Zero out same-camera pairs
    same_cam = cam_ids[:, None] == cam_ids[None, :]
    sim_matrix[same_cam] = 0.0

    # Ensure non-negative
    sim_matrix = np.maximum(sim_matrix, 0.0)

    return sim_matrix


def _split_component_by_camera(
    detections: List[Dict],
    members: List[int],
) -> List[List[int]]:
    """
    Partition a connected component so each resulting group has <=1 detection per camera.
    """
    if len(members) <= 1:
        return [members]

    cam_to_members: Dict[int, List[int]] = {}
    for idx in members:
        cam_id = detections[idx].get("cam_id", -1)
        cam_to_members.setdefault(cam_id, []).append(idx)

    max_per_cam = max(len(idxs) for idxs in cam_to_members.values())
    if max_per_cam <= 1:
        return [members]

    # Sort duplicates deterministically: higher-confidence detections first.
    for cam_id, idxs in cam_to_members.items():
        idxs.sort(key=lambda i: (-float(detections[i].get("conf", 0.0)), i))

    # Round-robin across cameras: kth detection per camera goes to kth split group.
    split_groups: List[List[int]] = [[] for _ in range(max_per_cam)]
    for cam_id in sorted(cam_to_members.keys()):
        idxs = cam_to_members[cam_id]
        for k, det_idx in enumerate(idxs):
            split_groups[k].append(det_idx)

    return [group for group in split_groups if group]


def graph_clustering_fast(
    detections: List[Dict],
    prob_thresh: float = 0.01,
    sem_thresh: float = 0.6,
    mode: str = "spatial",
    allow_single_cam: bool = True,
    spatial_weight: float = 0.7,
    semantic_weight: float = 0.3,
) -> Tuple[List[List[int]], Dict]:
    start_time = time.time()
    stats_info = {
        "n_detections": len(detections),
        "mode": mode,
        "prob_thresh": prob_thresh,
        "timing": {},
    }

    N = len(detections)
    if N == 0:
        return [], stats_info

    # =========================================================================
    # Phase 1: Extract arrays from detections
    # =========================================================================
    extract_start = time.time()

    positions = np.array([d["z_bev"] for d in detections], dtype=np.float64)  # [N, 2]
    covariances = np.array([d["R_bev"] for d in detections], dtype=np.float64)  # [N, 2, 2]
    cam_ids = np.array([d.get("cam_id", -1) for d in detections], dtype=np.int32)  # [N,]

    # Extract features if needed for semantic/joint mode
    has_features = mode in ("semantic", "joint")
    if has_features:
        feature_dim = None
        for d in detections:
            if d.get("vec") is not None:
                feature_dim = len(d["vec"])
                break

        if feature_dim is None:
            has_features = False
            features = None
        else:
            features = np.zeros((N, feature_dim), dtype=np.float32)
            for i, d in enumerate(detections):
                if d.get("vec") is not None:
                    features[i] = d["vec"]
    else:
        features = None

    stats_info["timing"]["extract"] = time.time() - extract_start

    # =========================================================================
    # Phase 2: Compute pairwise probabilities
    # =========================================================================
    prob_start = time.time()

    # Initialize probability matrix
    prob_matrix = np.zeros((N, N), dtype=np.float64)

    if mode in ("spatial", "joint"):
        # Compute Mahalanobis distances
        mahal_sq = _vectorized_mahalanobis_spatial(positions, covariances, cam_ids)

        # Convert to chi-squared probabilities: P(same) = 1 - CDF(d², df=2)
        # Use survival function for numerical stability
        spatial_prob = np.zeros_like(mahal_sq)
        finite_mask = np.isfinite(mahal_sq)
        spatial_prob[finite_mask] = chi2.sf(mahal_sq[finite_mask], df=2)

        if mode == "spatial":
            prob_matrix = spatial_prob

    if mode in ("semantic", "joint") and has_features:
        # Compute cosine similarities
        sem_sim = _vectorized_semantic_similarity(features, cam_ids)

        # Apply threshold
        sem_prob = np.where(sem_sim >= sem_thresh, sem_sim, 0.0)

        if mode == "semantic":
            prob_matrix = sem_prob

    if mode == "joint":
        # Weighted geometric mean fusion: P_joint = P_spatial^w_geo × P_semantic^w_sem
        # Treats both modalities as independent evidence sources

        if has_features:
            has_feat_per_det = (features != 0).any(axis=1)  # [N,]
            both_have_feat = has_feat_per_det[:, None] & has_feat_per_det[None, :]  # [N,N]

            # Adaptive weights based on spatial confidence (covariance trace)
            # Compute trace of summed covariances for each pair
            cov_traces = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(i + 1, N):
                    cov_traces[i, j] = np.trace(covariances[i] + covariances[j])
                    cov_traces[j, i] = cov_traces[i, j]

            # Logistic weight: uncertain depth → trust semantic more
            w_geo_base = 0.6
            confidence_factor = 1.0 / (1.0 + cov_traces)
            w_geo = np.clip(w_geo_base * (0.5 + confidence_factor), 0.3, 0.7)
            w_sem = 1.0 - w_geo

            # Use raw semantic similarity (not thresholded) as P(same|appearance)
            sem_prob_raw = np.where(both_have_feat, sem_sim, 0.5)  # 0.5 = uninformative

            # Geometric mean in log space
            eps = 1e-10
            log_spatial = np.log(np.maximum(spatial_prob, eps))
            log_semantic = np.log(np.maximum(sem_prob_raw, eps))
            log_joint = w_geo * log_spatial + w_sem * log_semantic
            prob_matrix = np.clip(np.exp(log_joint), 0.0, 1.0)

            # Semantic veto: pairs below sem_thresh get zeroed out
            veto_mask = both_have_feat & (sem_sim < sem_thresh)
            prob_matrix = np.where(veto_mask, 0.0, prob_matrix)

            stats_info["w_geo_mean"] = float(np.mean(w_geo[w_geo > 0]))
            stats_info["n_vetoed"] = int(np.sum(veto_mask) / 2)  # Symmetric, divide by 2
        else:
            prob_matrix = spatial_prob

    stats_info["timing"]["probability"] = time.time() - prob_start

    # =========================================================================
    # Phase 3: Build Union-Find from edges above threshold
    # =========================================================================
    union_start = time.time()

    uf = UnionFind(N)
    n_edges = 0

    # Only check upper triangle (symmetric matrix)
    for i in range(N):
        for j in range(i + 1, N):
            if prob_matrix[i, j] >= prob_thresh:
                uf.union(i, j)
                n_edges += 1

    stats_info["n_edges_created"] = n_edges
    stats_info["timing"]["union_find"] = time.time() - union_start

    # =========================================================================
    # Phase 4: Extract connected components
    # =========================================================================
    component_start = time.time()

    components = uf.get_components()

    # Separate multi-camera and single-camera clusters
    clusters: List[List[int]] = []
    single_cam_clusters: List[List[int]] = []

    n_same_cam_splits = 0
    for _root, members in components.items():
        split_members = _split_component_by_camera(detections, members)
        n_same_cam_splits += max(0, len(split_members) - 1)

        for group in split_members:
            cameras_in_cluster = set(cam_ids[idx] for idx in group)

            if len(cameras_in_cluster) >= 2:
                # Multi-camera cluster
                cluster_id = len(clusters)
                for idx in group:
                    detections[idx]["id"] = cluster_id
                clusters.append(group)
            else:
                single_cam_clusters.append(group)

    stats_info["n_multi_cam_clusters"] = len(clusters)
    stats_info["n_single_cam_pending"] = len(single_cam_clusters)
    stats_info["n_same_cam_splits"] = n_same_cam_splits

    # =========================================================================
    # Phase 5: Handle single-camera detections
    # =========================================================================
    if allow_single_cam and single_cam_clusters:
        SINGLE_CAM_FILTER_THRESH_FAST = 0.3  # Raised from prob_thresh for single-cam

        # Compute cluster centroids, covariances, and pooled features
        cluster_centroids_list = []
        cluster_covs_list = []
        cluster_feats_list = []
        cluster_cam_sets = []
        if clusters:
            for members in clusters:
                cluster_centroids_list.append(np.mean(positions[members], axis=0))
                cluster_covs_list.append(np.mean(covariances[members], axis=0))
                cluster_cam_sets.append(set(cam_ids[m] for m in members))
                # Pool features for cluster
                if has_features:
                    member_feats = [features[m] for m in members if np.any(features[m] != 0)]
                    if member_feats:
                        pooled = np.mean(member_feats, axis=0)
                        pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
                        cluster_feats_list.append(pooled)
                    else:
                        cluster_feats_list.append(None)
                else:
                    cluster_feats_list.append(None)

        for single_cluster in single_cam_clusters:
            for idx in single_cluster:
                det_pos = positions[idx]
                det_cov = covariances[idx]
                det_feat = features[idx] if has_features else None
                det_has_feat = det_feat is not None and np.any(det_feat != 0)
                det_cam = cam_ids[idx]

                # Check if statistically distinct from all existing clusters
                too_close = False
                for c_idx in range(len(cluster_centroids_list)):
                    # Camera-consistency rule:
                    # A cluster containing this camera cannot filter this detection.
                    # Same-camera detections should not suppress each other.
                    if det_cam in cluster_cam_sets[c_idx]:
                        continue

                    # MODE-AWARE FILTERING: Respect the clustering mode
                    if mode == "semantic":
                        # SEMANTIC MODE: Use only appearance similarity
                        if det_has_feat and cluster_feats_list[c_idx] is not None:
                            s_sim = float(max(0.0, np.dot(
                                det_feat / (np.linalg.norm(det_feat) + 1e-8),
                                cluster_feats_list[c_idx]
                            )))
                            # Merge if semantically similar (use consistent threshold)
                            if s_sim >= sem_thresh:
                                too_close = True
                                break
                    else:
                        # SPATIAL/JOINT MODE: Use spatial distance first
                        diff = det_pos - cluster_centroids_list[c_idx]
                        cov_sum = det_cov + cluster_covs_list[c_idx]
                        try:
                            d_sq = diff @ np.linalg.solve(cov_sum, diff)
                            prob = chi2.sf(d_sq, df=2)
                            if prob >= SINGLE_CAM_FILTER_THRESH_FAST:
                                # Spatially close — check semantic veto in joint mode
                                if mode == "joint" and det_has_feat and cluster_feats_list[c_idx] is not None:
                                    s_sim = float(max(0.0, np.dot(
                                        det_feat / (np.linalg.norm(det_feat) + 1e-8),
                                        cluster_feats_list[c_idx]
                                    )))
                                    # Semantic veto: keep separate if very dissimilar
                                    if s_sim < 0.3:
                                        continue  # Different person, don't filter
                                too_close = True
                                break
                        except np.linalg.LinAlgError:
                            pass

                if not too_close:
                    cluster_id = len(clusters)
                    detections[idx]["id"] = cluster_id
                    clusters.append([idx])
                    cluster_centroids_list.append(det_pos)
                    cluster_covs_list.append(det_cov)
                    cluster_cam_sets.append({int(det_cam)})
                    det_feat_norm = (det_feat / (np.linalg.norm(det_feat) + 1e-8)) if det_has_feat else None
                    cluster_feats_list.append(det_feat_norm)

    stats_info["timing"]["components"] = time.time() - component_start
    stats_info["n_final_clusters"] = len(clusters)
    stats_info["timing"]["total"] = time.time() - start_time

    return clusters, stats_info


class UnionFind:
    """
    Disjoint set data structure for efficient connected component identification.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if merge occurred, False if already same set."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def get_components(self) -> Dict[int, List[int]]:
        """Return all connected components as {root: [members]}."""
        components: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        return components


def compute_edge_probability(
    det_i: Dict,
    det_j: Dict,
    mode: str = "spatial",
    sem_thresh: float = 0.6,  # Consistent with JOINT_CONFIG; caller can override for semantic mode (0.85)
) -> Tuple[float, Dict]:
    """
    Compute edge probability between two detections using chi-squared CDF.

    Args:
        det_i, det_j: Detection dictionaries with z_bev, R_bev, vec (optional)
        mode: "spatial", "semantic", or "joint"
        sem_thresh: Semantic similarity threshold for joint mode

    Returns:
        Tuple of (probability, stats_info_dict)
    """
    stats_info = {}

    # Compute Mahalanobis distance (using covariance sum for joint uncertainty)
    diff = det_i["z_bev"] - det_j["z_bev"]
    cov_sum = det_i["R_bev"] + det_j["R_bev"]

    try:
        mahal_sq = float(diff @ np.linalg.solve(cov_sum, diff))
    except np.linalg.LinAlgError:
        mahal_sq = np.inf

    stats_info["mahal_sq"] = mahal_sq
    stats_info["euclidean_dist"] = float(np.linalg.norm(diff))
    stats_info["cov_det"] = float(np.linalg.det(cov_sum))

    # Chi-squared probability: P(same identity) = 1 - CDF(mahal_sq, df=2)
    # This gives high probability for low Mahalanobis distance (consistent positions)
    # and low probability for high Mahalanobis distance (inconsistent positions)
    if not np.isfinite(mahal_sq):
        spatial_prob = 0.0
    else:
        # Use chi2.sf (survival function) for better numerical stability
        spatial_prob = float(chi2.sf(mahal_sq, df=2))

    stats_info["spatial_prob"] = spatial_prob

    if mode == "spatial":
        return spatial_prob, stats_info

    # Semantic similarity
    vec_i = det_i.get("vec")
    vec_j = det_j.get("vec")

    if vec_i is None or vec_j is None:
        sem_sim = 0.0
    else:
        sem_sim = semantic_similarity(vec_i, vec_j)

    stats_info["sem_sim"] = sem_sim

    if mode == "semantic":
        sem_prob = sem_sim if sem_sim >= sem_thresh else 0.0
        return sem_prob, stats_info

    has_features = (vec_i is not None and vec_j is not None and
                    np.any(vec_i != 0) and np.any(vec_j != 0))

    sem_prob = sem_sim if has_features else 0.5  # 0.5 = uninformative prior

    cov_trace = np.trace(cov_sum)

    w_geo_base = 0.6  # Base weight for spatial
    confidence_factor = 1.0 / (1.0 + cov_trace)  # Higher trace → lower confidence
    w_geo = w_geo_base * (0.5 + confidence_factor)  # Range: [0.3, 0.6]
    w_geo = np.clip(w_geo, 0.3, 0.7)
    w_sem = 1.0 - w_geo

    stats_info["w_geo"] = w_geo
    stats_info["w_sem"] = w_sem
    stats_info["cov_trace"] = cov_trace

    eps = 1e-10
    log_spatial = np.log(max(spatial_prob, eps))
    log_semantic = np.log(max(sem_prob, eps))

    log_joint = w_geo * log_spatial + w_sem * log_semantic
    joint_prob = np.exp(log_joint)
    joint_prob = float(np.clip(joint_prob, 0.0, 1.0))

    if has_features and sem_sim < sem_thresh:
        joint_prob = 0.0
        stats_info["semantic_action"] = "veto"
    elif has_features:
        stats_info["semantic_action"] = "agree"
    else:
        stats_info["semantic_action"] = "no_features"

    stats_info["boost_applied"] = has_features

    stats_info["joint_prob"] = joint_prob
    stats_info["sem_contrib"] = sem_sim
    # stats_info["spatial_gated"] = spatial_prob < SPATIAL_GATE

    return joint_prob, stats_info


def graph_clustering(
    detections: List[Dict],
    homographies: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    prob_thresh: float = 0.01,  # P(same|d²) > 1% → create edge (χ²: d² < 9.21)
    sem_thresh: float = 0.6,  # Consistent with JOINT_CONFIG; caller can override for semantic mode (0.85)
    mode: str = "spatial",
    allow_single_cam: bool = True,
    max_euclidean_dist: float = 0.5,  # Hard cutoff - keep tight for dense scenes like WildTrack
) -> Tuple[List[List[int]], Dict]:
    """
    Graph-based clustering with chi-squared probability edge weights.

    Args:
        detections: List of detection dicts with z_bev, R_bev, cam_id, vec
        homographies: Ground-plane homographies (unused, kept for API compatibility)
        prob_thresh: Minimum association probability to create edge
                    Default 0.01 corresponds to χ²(2)=9.21 threshold (99% conf)
        sem_thresh: Semantic similarity threshold for joint mode
        mode: "spatial", "semantic", or "joint"
        allow_single_cam: Whether to include single-detection clusters
        max_euclidean_dist: Hard Euclidean distance cutoff (meters) to prevent
                           transitive closure chaining across scene. Default 0.5m.
                           Only applies to "spatial" and "joint" modes; ignored in "semantic" mode.

    Returns:
        Tuple of (clusters, stats_info)
        - clusters: List of clusters (each cluster is list of detection indices)
        - stats_info: Dict with timing and edge statistics
    """
    start_time = time.time()
    stats_info = {
        "n_detections": len(detections),
        "mode": mode,
        "prob_thresh": prob_thresh,
        "max_euclidean_dist": max_euclidean_dist,
        "edges": [],
        "timing": {},
    }

    if len(detections) == 0:
        return [], stats_info

    N = len(detections)
    uf = UnionFind(N)

    # Phase 1: Build graph edges based on chi-squared probability
    edge_build_start = time.time()
    n_edges_created = 0
    n_pairs_checked = 0
    n_same_cam_skipped = 0
    # Hard Euclidean distance cutoff to prevent transitive closure across scene.
    n_euclidean_rejected = 0
    accepted_edge_count = np.zeros(N, dtype=np.int32)
    euclidean_reject_count = np.zeros(N, dtype=np.int32)

    positions = np.asarray([det["z_bev"] for det in detections], dtype=np.float64)
    cam_ids = np.asarray([int(det.get("cam_id", -1)) for det in detections], dtype=np.int32)

    upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    same_cam_mask = cam_ids[:, None] == cam_ids[None, :]
    cross_cam_upper = upper_mask & (~same_cam_mask)
    n_pairs_checked = int(upper_mask.sum())
    n_same_cam_skipped = int((upper_mask & same_cam_mask).sum())

    diff = positions[:, None, :] - positions[None, :, :]
    euclidean_dist = np.linalg.norm(diff, axis=2)

    if mode != "semantic":
        too_far_upper = cross_cam_upper & (euclidean_dist > max_euclidean_dist)
        n_euclidean_rejected = int(too_far_upper.sum())
        too_far_sym = too_far_upper | too_far_upper.T
        euclidean_reject_count = too_far_sym.sum(axis=1).astype(np.int32, copy=False)
        valid_pair_upper = cross_cam_upper & (~too_far_upper)
    else:
        valid_pair_upper = cross_cam_upper

    prob_matrix = np.zeros((N, N), dtype=np.float64)
    sem_sim_matrix = np.zeros((N, N), dtype=np.float64)
    spatial_prob = np.zeros((N, N), dtype=np.float64)
    mahal_sq = np.full((N, N), np.inf, dtype=np.float64)
    cov_det_matrix = np.zeros((N, N), dtype=np.float64)
    cov_trace_matrix = np.zeros((N, N), dtype=np.float64)
    w_geo_matrix = np.zeros((N, N), dtype=np.float64)
    w_sem_matrix = np.zeros((N, N), dtype=np.float64)

    has_feat_per_det = np.zeros((N,), dtype=bool)
    feature_dim = 0
    for det in detections:
        vec = det.get("vec")
        if vec is not None:
            arr = np.asarray(vec).reshape(-1)
            if arr.size > 0:
                feature_dim = int(arr.size)
                break
    if feature_dim > 0:
        features = np.zeros((N, feature_dim), dtype=np.float32)
        for idx, det in enumerate(detections):
            vec = det.get("vec")
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size != feature_dim:
                continue
            features[idx] = arr
            has_feat_per_det[idx] = bool(np.any(arr != 0))
        sem_sim_matrix = _vectorized_semantic_similarity(features, cam_ids)

    if mode in ("spatial", "joint"):
        covariances = np.asarray([det["R_bev"] for det in detections], dtype=np.float64)
        cov_sum = covariances[:, None, :, :] + covariances[None, :, :, :]
        a = cov_sum[:, :, 0, 0]
        b = cov_sum[:, :, 0, 1]
        c = cov_sum[:, :, 1, 0]
        d = cov_sum[:, :, 1, 1]
        cov_det_matrix = a * d - b * c
        valid_cov = np.abs(cov_det_matrix) > 1e-12
        inv00 = np.zeros_like(a)
        inv01 = np.zeros_like(a)
        inv10 = np.zeros_like(a)
        inv11 = np.zeros_like(a)
        inv00[valid_cov] = d[valid_cov] / cov_det_matrix[valid_cov]
        inv01[valid_cov] = -b[valid_cov] / cov_det_matrix[valid_cov]
        inv10[valid_cov] = -c[valid_cov] / cov_det_matrix[valid_cov]
        inv11[valid_cov] = a[valid_cov] / cov_det_matrix[valid_cov]

        dx = diff[:, :, 0]
        dy = diff[:, :, 1]
        mahal_all = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy)
        mahal_sq[valid_cov] = mahal_all[valid_cov]

        valid_spatial = valid_cov & np.isfinite(mahal_sq)
        spatial_prob[valid_spatial] = chi2.sf(mahal_sq[valid_spatial], df=2)
        cov_trace_matrix = np.trace(cov_sum, axis1=2, axis2=3)

        if mode == "spatial":
            prob_matrix = spatial_prob

    if mode == "semantic":
        prob_matrix = np.where(sem_sim_matrix >= sem_thresh, sem_sim_matrix, 0.0)

    if mode == "joint":
        pair_has_feat = has_feat_per_det[:, None] & has_feat_per_det[None, :]
        sem_prob = np.where(pair_has_feat, sem_sim_matrix, 0.5)

        w_geo_base = 0.6
        confidence_factor = 1.0 / (1.0 + cov_trace_matrix)
        w_geo_matrix = np.clip(w_geo_base * (0.5 + confidence_factor), 0.3, 0.7)
        w_sem_matrix = 1.0 - w_geo_matrix

        eps = 1e-10
        log_spatial = np.log(np.maximum(spatial_prob, eps))
        log_semantic = np.log(np.maximum(sem_prob, eps))
        prob_matrix = np.exp(w_geo_matrix * log_spatial + w_sem_matrix * log_semantic)
        prob_matrix = np.clip(prob_matrix, 0.0, 1.0)

        # Semantic veto: if both have features but are dissimilar, suppress edge.
        prob_matrix[pair_has_feat & (sem_sim_matrix < sem_thresh)] = 0.0

    edge_upper = valid_pair_upper & (prob_matrix >= prob_thresh)
    edge_indices = np.transpose(np.nonzero(edge_upper))
    n_edges_created = int(edge_indices.shape[0])
    edge_sym = edge_upper | edge_upper.T
    accepted_edge_count = edge_sym.sum(axis=1).astype(np.int32, copy=False)

    for i, j in edge_indices:
        i_int = int(i)
        j_int = int(j)
        uf.union(i_int, j_int)

    n_phase1b_candidate_dets = 0
    n_phase1b_pairs_checked = 0
    n_phase1b_edges_created = 0
    if mode == "joint":
        PHASE1B_MAX_EUCLIDEAN_DIST = 0.8
        PHASE1B_SEM_THRESH = 0.5

        phase1b_candidates = [
            idx for idx in range(N)
            if accepted_edge_count[idx] == 0 and euclidean_reject_count[idx] > 0
        ]
        n_phase1b_candidate_dets = len(phase1b_candidates)

        if n_phase1b_candidate_dets >= 2:
            for a_pos in range(n_phase1b_candidate_dets):
                i = phase1b_candidates[a_pos]
                cam_i = detections[i].get("cam_id")
                vec_i = detections[i].get("vec")
                has_i_feat = vec_i is not None and np.linalg.norm(vec_i) > 1e-8
                if not has_i_feat:
                    continue

                for b_pos in range(a_pos + 1, n_phase1b_candidate_dets):
                    j = phase1b_candidates[b_pos]
                    cam_j = detections[j].get("cam_id")
                    if cam_i == cam_j:
                        continue

                    diff = detections[i]["z_bev"] - detections[j]["z_bev"]
                    eucl_dist = float(np.linalg.norm(diff))
                    # Rescue band: pairs rejected by 0.5m gate, but not too far.
                    if eucl_dist <= max_euclidean_dist or eucl_dist > PHASE1B_MAX_EUCLIDEAN_DIST:
                        continue

                    n_phase1b_pairs_checked += 1
                    vec_j = detections[j].get("vec")
                    has_j_feat = vec_j is not None and np.linalg.norm(vec_j) > 1e-8
                    if not has_j_feat:
                        continue

                    sem_sim = semantic_similarity(vec_i, vec_j)
                    if sem_sim >= PHASE1B_SEM_THRESH:
                        uf.union(i, j)
                        n_phase1b_edges_created += 1
                        accepted_edge_count[i] += 1
                        accepted_edge_count[j] += 1


    stats_info["n_phase1b_candidate_dets"] = n_phase1b_candidate_dets
    stats_info["n_phase1b_pairs_checked"] = n_phase1b_pairs_checked
    stats_info["n_phase1b_edges_created"] = n_phase1b_edges_created

    stats_info["timing"]["edge_build"] = time.time() - edge_build_start
    stats_info["n_pairs_checked"] = n_pairs_checked
    stats_info["n_same_cam_skipped"] = n_same_cam_skipped
    stats_info["n_edges_created"] = n_edges_created


    # Phase 2: Extract connected components
    component_start = time.time()
    components = uf.get_components()
    stats_info["timing"]["component_extract"] = time.time() - component_start

    n_clusters_split = 0
    n_phase2b_ejected_members = 0
    if mode in ("semantic", "joint"):
        validated_components: Dict[int, List[int]] = {}
        for root, members in components.items():
            if len(members) <= 2:
                validated_components[root] = members
                continue

            # Build cross-camera similarity matrix for this cluster
            n = len(members)
            # For each member, count how many cross-camera pairs fail the threshold
            fail_count = [0] * n
            for ii in range(n):
                for jj in range(ii + 1, n):
                    if detections[members[ii]].get("cam_id") == detections[members[jj]].get("cam_id"):
                        continue
                    vec_a = detections[members[ii]].get("vec")
                    vec_b = detections[members[jj]].get("vec")
                    if vec_a is None or vec_b is None or semantic_similarity(vec_a, vec_b) < sem_thresh:
                        fail_count[ii] += 1
                        fail_count[jj] += 1

            if max(fail_count) == 0:
                # All pairs pass — cluster is valid
                validated_components[root] = members
                continue

            remaining = list(range(n))
            ejected = []
            while remaining:
                # Recompute fail counts for remaining members
                r_fail = {idx: 0 for idx in remaining}
                all_pass = True
                for ii_pos, ii in enumerate(remaining):
                    for jj in remaining[ii_pos + 1:]:
                        if detections[members[ii]].get("cam_id") == detections[members[jj]].get("cam_id"):
                            continue
                        vec_a = detections[members[ii]].get("vec")
                        vec_b = detections[members[jj]].get("vec")
                        if vec_a is None or vec_b is None or semantic_similarity(vec_a, vec_b) < sem_thresh:
                            r_fail[ii] += 1
                            r_fail[jj] += 1
                            all_pass = False
                if all_pass:
                    break
                # Eject member with most failures
                worst = max(remaining, key=lambda idx: r_fail[idx])
                remaining.remove(worst)
                ejected.append(worst)

            # Add the validated core cluster
            if remaining:
                core = [members[i] for i in remaining]
                validated_components[core[0]] = core
            # Add ejected members as singletons
            for idx in ejected:
                det_idx = members[idx]
                validated_components[det_idx] = [det_idx]
            if ejected:
                n_clusters_split += 1
                n_phase2b_ejected_members += len(ejected)

        components = validated_components

    stats_info["n_clusters_split"] = n_clusters_split
    stats_info["n_phase2b_ejected_members"] = n_phase2b_ejected_members

    # Phase 3: Convert to cluster list and assign IDs
    clusters: List[List[int]] = []
    single_cam_clusters: List[List[int]] = []

    n_same_cam_splits = 0
    for _root, members in components.items():
        split_members = _split_component_by_camera(detections, members)
        n_same_cam_splits += max(0, len(split_members) - 1)

        for group in split_members:
            # Check camera diversity within cluster
            cameras_in_cluster = set(detections[idx].get("cam_id") for idx in group)

            if len(cameras_in_cluster) >= 2:
                # Multi-camera cluster: assign same ID to all members
                cluster_id = len(clusters)
                for idx in group:
                    detections[idx]["id"] = cluster_id
                    if "probs" not in detections[idx]:
                        detections[idx]["probs"] = {}
                clusters.append(group)
            else:
                # Single-camera: defer to pass 2
                single_cam_clusters.append(group)

    stats_info["n_euclidean_rejected"] = n_euclidean_rejected
    stats_info["n_multi_cam_clusters"] = len(clusters)
    stats_info["n_single_cam_pending"] = len(single_cam_clusters)
    stats_info["n_same_cam_splits"] = n_same_cam_splits

    # Phase 4: Handle single-camera detections
    if allow_single_cam and single_cam_clusters:
        pass2_start = time.time()

        # Build centroids, covariances, and pooled features for existing multi-camera clusters
        cluster_stats = []
        for cluster in clusters:
            centroid = np.mean([detections[i]["z_bev"] for i in cluster], axis=0)
            cov = np.mean([detections[i]["R_bev"] for i in cluster], axis=0)
            cam_set = set(detections[i].get("cam_id", -1) for i in cluster)
            # Pool semantic features for cluster (confidence-weighted average)
            cluster_vecs = [detections[i].get("vec") for i in cluster
                           if detections[i].get("vec") is not None and np.any(detections[i]["vec"] != 0)]
            if cluster_vecs:
                pooled_feat = np.mean(cluster_vecs, axis=0)
                pooled_feat = pooled_feat / (np.linalg.norm(pooled_feat) + 1e-8)
            else:
                pooled_feat = None
            cluster_stats.append((centroid, cov, pooled_feat, cam_set))

        n_pass2_added = 0
        n_pass2_filtered = 0
        n_pass2_filtered_semantic = 0
        n_pass2_filtered_spatial = 0

        for single_cluster in single_cam_clusters:
            # For each detection in the single-camera cluster
            for idx in single_cluster:
                det_pos = detections[idx]["z_bev"]
                det_cov = detections[idx]["R_bev"]
                det_vec = detections[idx].get("vec")
                has_det_feat = det_vec is not None and np.any(det_vec != 0)
                det_cam = detections[idx].get("cam_id", -1)

                # Check if statistically distinct from all existing clusters
                SINGLE_CAM_FILTER_THRESH = 0.3  # Spatial probability threshold for spatial/joint modes
                too_close = False
                filter_reason = None
                for cluster_pos, cluster_cov, cluster_feat, cluster_cam_set in cluster_stats:
                    # Camera-consistency rule:
                    # A cluster containing this camera cannot filter this detection.
                    # Same-camera detections should not suppress each other.
                    if det_cam in cluster_cam_set:
                        continue

                    # MODE-AWARE FILTERING: Respect the clustering mode
                    if mode == "semantic":
                        # SEMANTIC MODE: Use only appearance similarity
                        if has_det_feat and cluster_feat is not None:
                            sem_sim_val = semantic_similarity(det_vec, cluster_feat)
                            # Merge if semantically similar (use consistent threshold)
                            if sem_sim_val >= sem_thresh:
                                too_close = True
                                filter_reason = "semantic"
                                break
                    else:
                        # SPATIAL/JOINT MODE: Use spatial distance with semantic veto in joint mode
                        prob, _ = compute_edge_probability(
                            {"z_bev": det_pos, "R_bev": det_cov, "vec": det_vec},
                            {"z_bev": cluster_pos, "R_bev": cluster_cov, "vec": cluster_feat},
                            mode=mode,
                            sem_thresh=sem_thresh,
                        )
                        if prob >= SINGLE_CAM_FILTER_THRESH:
                            too_close = True
                            filter_reason = "spatial"
                            break

                if not too_close:
                    # Add as new cluster
                    cluster_id = len(clusters)
                    detections[idx]["id"] = cluster_id
                    if "probs" not in detections[idx]:
                        detections[idx]["probs"] = {}
                    clusters.append([idx])
                    det_feat_pooled = (det_vec / (np.linalg.norm(det_vec) + 1e-8)) if has_det_feat else None
                    cluster_stats.append((det_pos, det_cov, det_feat_pooled, {int(det_cam)}))
                    n_pass2_added += 1
                else:
                    n_pass2_filtered += 1
                    if filter_reason == "semantic":
                        n_pass2_filtered_semantic += 1
                    elif filter_reason == "spatial":
                        n_pass2_filtered_spatial += 1

        stats_info["timing"]["pass2"] = time.time() - pass2_start
        stats_info["n_pass2_added"] = n_pass2_added
        stats_info["n_pass2_filtered"] = n_pass2_filtered
        stats_info["n_pass2_filtered_semantic"] = n_pass2_filtered_semantic
        stats_info["n_pass2_filtered_spatial"] = n_pass2_filtered_spatial

    stats_info["timing"]["total"] = time.time() - start_time
    stats_info["n_final_clusters"] = len(clusters)

    return clusters, stats_info

def compute_identity_distribution(
    detection_idx: int,
    all_detections: List[Dict],
    # homographies: Dict[Tuple[int, int], np.ndarray],
    mode: str = "joint",
    tau_geo: float = 5.0,
    tau_sem: float = 2.0,
    lambda_kl: float = 1.0,
) -> Dict[int, float]:
    """
    Compute P(ID_k | detection) for all existing identities.
    """
    # detection in consideration
    det = all_detections[detection_idx]

    id_groups: Dict[int, List[Dict]] = {}
    for i, d in enumerate(all_detections):
        # skip the detection in consideration and any detections from the same camera
        if i == detection_idx or d["cam_id"] == det["cam_id"]:
            continue
        if d.get("id") is not None:
            # update cluster id dict with this detection
            id_groups.setdefault(d["id"], []).append(d)

    # id_groups is a mapping: cluster_id --> detections within cluster

    if not id_groups:
        return {}

    geo_sims: Dict[int, float] = {}
    sem_sims: Dict[int, float] = {}

    # how likely is our candidate detection to belong to one of the existing clusters?
    for id_k, group in id_groups.items():
        g_list: List[float] = []
        s_list: List[float] = []

        for d in group:
            diff = det["z_bev"] - d["z_bev"]
            # PAPER EQ 11: Mahalanobis distance with covariance fusion
            # d² = Δ^T (Σ₁ + Σ₂)⁻¹ Δ  [chi-squared distribution, no sqrt]
            cov_sum = det["R_bev"] + d["R_bev"]
            try:
                # OPTIMIZATION: Use solve instead of inv for Mahalanobis distance
                mahal_sq = float(diff @ np.linalg.solve(cov_sum, diff))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                mahal_sq = np.inf

            # joint mode
            g = np.exp(-(mahal_sq/2))
            s = semantic_similarity(det["vec"], d["vec"])
            g_list.append(g)
            s_list.append(s)

        geo_sims[id_k] = float(np.mean(g_list)) if g_list else 0.0
        sem_sims[id_k] = float(np.mean(s_list)) if s_list else 0.0

    ids = list(id_groups.keys())
    if not ids:
        return {}

    def _softmax(scores: np.ndarray, tau: float) -> np.ndarray:
        if scores.size == 0:
            return scores
        logits = scores / max(tau, 1e-6)
        logits -= logits.max()
        exp_scores = np.exp(logits)
        denom = exp_scores.sum() + 1e-8
        return exp_scores / denom

    if mode == "spatial":
        scores = np.array([geo_sims[k] for k in ids], dtype=np.float32)
        probs = _softmax(scores, tau_geo)
        return {ids[i]: float(probs[i]) for i in range(len(ids))}

    if mode == "semantic":
        scores = np.array([sem_sims[k] for k in ids], dtype=np.float32)
        probs = _softmax(scores, tau_sem)
        return {ids[i]: float(probs[i]) for i in range(len(ids))}

    # joint: combine both distributions w/ KL penalty
    geo_scores = np.array([geo_sims[k] for k in ids], dtype=np.float32)
    sem_scores = np.array([sem_sims[k] for k in ids], dtype=np.float32)

    q_geo = _softmax(geo_scores, tau_geo)
    q_sem = _softmax(sem_scores, tau_sem)

    kl_per_id = q_geo * np.log((q_geo + 1e-8) / (q_sem + 1e-8))
    q_avg = 0.5 * (q_geo + q_sem)
    probs = q_avg * np.exp(-lambda_kl * np.abs(kl_per_id))
    probs /= probs.sum() + 1e-8
    return {ids[i]: float(probs[i]) for i in range(len(ids))}


def merge_nearby_clusters(
    clusters: List[List[int]],
    detections: List[Dict],
    mahal_thresh: float = 9.21,
) -> List[List[int]]:
    """
    Merge clusters whose fused positions are statistically close.

    Args:
        clusters: List of clusters (each cluster is list of detection indices)
        detections: List of detection dicts with z_bev, R_bev
        mahal_thresh: Mahalanobis distance threshold for merging (default χ²(2,0.99)=9.21)

    Returns:
        Merged clusters list
    """
    if len(clusters) <= 1:
        return clusters

    # Compute fused position for each cluster
    cluster_positions = []
    cluster_covariances = []

    for cluster in clusters:
        if len(cluster) == 0:
            continue

        measurements = [
            {
                "z_bev": detections[i]["z_bev"],
                "R_bev": detections[i]["R_bev"],
                "R_common_bev": detections[i].get("R_common_bev"),
            }
            for i in cluster
        ]
        z_fused, P_fused = _precision_fuse_measurements(measurements)

        cluster_positions.append(z_fused)
        cluster_covariances.append(P_fused)

    # Build merge graph using Union-Find
    n_clusters = len(clusters)
    uf = UnionFind(n_clusters)
    n_merges = 0

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Mahalanobis distance between cluster centers
            diff = cluster_positions[i] - cluster_positions[j]
            cov_sum = cluster_covariances[i] + cluster_covariances[j]

            try:
                mahal_sq = diff @ np.linalg.solve(cov_sum, diff)
            except np.linalg.LinAlgError:
                continue

            if mahal_sq < mahal_thresh:
                uf.union(i, j)
                n_merges += 1

    # Extract merged clusters
    merged_map = {}
    for i in range(n_clusters):
        root = uf.find(i)
        if root not in merged_map:
            merged_map[root] = []
        merged_map[root].extend(clusters[i])

    return list(merged_map.values())


def _stabilize_covariance(cov: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Symmetrize and clamp a 2x2 covariance matrix to be PSD."""
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, float(min_eig))
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _get_shared_bev_covariance(measurements: List[Dict]) -> np.ndarray:
    """Return the shared covariance term only when every measurement agrees on it."""
    if not measurements:
        return np.zeros((2, 2), dtype=np.float64)

    shared_terms = []
    for meas in measurements:
        shared = meas.get("R_common_bev")
        if shared is None:
            return np.zeros((2, 2), dtype=np.float64)
        shared_arr = np.asarray(shared, dtype=np.float64)
        if shared_arr.shape != (2, 2) or not np.isfinite(shared_arr).all():
            return np.zeros((2, 2), dtype=np.float64)
        shared_terms.append(0.5 * (shared_arr + shared_arr.T))

    ref = shared_terms[0]
    if any(not np.allclose(shared, ref, rtol=1e-5, atol=1e-6) for shared in shared_terms[1:]):
        return np.zeros((2, 2), dtype=np.float64)
    return ref


def _precision_fuse_measurements(measurements: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse BEV measurements while preserving any explicitly marked shared covariance."""
    if not measurements:
        return np.zeros(2, dtype=np.float32), np.eye(2, dtype=np.float32)

    shared_cov = _get_shared_bev_covariance(measurements)
    precision_sum = np.zeros((2, 2), dtype=np.float64)
    weighted_sum = np.zeros(2, dtype=np.float64)
    positions = []
    full_covariances = []

    for meas in measurements:
        z = np.asarray(meas["z_bev"], dtype=np.float64).reshape(2)
        R_full = np.asarray(meas["R_bev"], dtype=np.float64).reshape(2, 2)
        R_indep = _stabilize_covariance(R_full - shared_cov)
        positions.append(z)
        full_covariances.append(R_full)

        try:
            R_inv = np.linalg.inv(R_indep)
        except np.linalg.LinAlgError:
            R_inv = np.eye(2, dtype=np.float64)
        precision_sum += R_inv
        weighted_sum += R_inv @ z

    try:
        P_indep = np.linalg.inv(precision_sum)
        x = P_indep @ weighted_sum
        P = _stabilize_covariance(P_indep + shared_cov)
    except np.linalg.LinAlgError:
        x = np.mean(positions, axis=0)
        P = _stabilize_covariance(np.mean(full_covariances, axis=0))

    return x.astype(np.float32), P.astype(np.float32)



def kalman_fusion(
    cluster_indices: List[int],
    detections: List[Dict],
    homographies: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    mode: str = "joint",
    tau_geo: float = 5.0,
    tau_sem: float = 2.0,
    lambda_kl: float = 1.0,
    alpha: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precision-weighted batch fusion for a cluster.

    Args:
        cluster_indices: Indices of detections in this cluster
        detections: All detections from all cameras
        homographies: Precomputed ground-plane homographies H_ij (unused, kept for API)
        mode: Matching mode (unused in fusion, kept for API compatibility)
        tau_geo, tau_sem, lambda_kl, alpha: Kept for API compatibility

    Returns:
        Tuple of (fused_position [2,], fused_covariance [2,2])
    """
    # Group detections by camera, keeping highest-confidence detection per camera
    camera_measurements: Dict[int, Dict] = {}
    for idx in cluster_indices:
        det = detections[idx]
        cam_id = det.get("cam_id")
        conf = det.get("conf", 0.5)
        if cam_id not in camera_measurements or conf > camera_measurements[cam_id]["conf"]:
            # Preserve legacy camera behavior (use only R_common_bev).
            # RadarScenes detections carry R_pose (and no R_common_bev), so
            # allow fallback only in that radar-specific shape.
            shared_cov = det.get("R_common_bev")
            if shared_cov is None and "R_pose" in det and "R_common_bev" not in det:
                shared_cov = det.get("R_pose")
            camera_measurements[cam_id] = {
                "z_bev": det["z_bev"],
                "R_bev": det["R_bev"],
                "R_common_bev": shared_cov,
                "conf": conf,
            }

    x, P = _precision_fuse_measurements(list(camera_measurements.values()))

    # POST-FUSION SAFEGUARDS
    if not np.isfinite(P).all() or not np.isfinite(x).all():
        positions = [m["z_bev"] for m in camera_measurements.values()]
        x = np.mean(positions, axis=0).astype(np.float32)
        P = np.eye(2, dtype=np.float32) * 1.0

    eigvals = np.linalg.eigvalsh(P)
    max_std = 10.0
    if eigvals.max() > max_std**2:
        P = P * (max_std**2 / eigvals.max())

    return x, P


def pool_cluster_features(
    cluster_indices: List[int],
    detections: List[Dict],
    feature_dim: int = 256,
) -> np.ndarray:
    """
    Pool SWIN features across cameras in a cluster (confidence-weighted).

    Args:
        cluster_indices: Indices of detections in this cluster
        detections: All detections
        feature_dim: Expected feature dimension

    Returns:
        Pooled feature vector (L2-normalized)
    """
    features = []
    weights = []

    for idx in cluster_indices:
        det = detections[idx]
        feat = det.get("vec")
        if feat is None:
            continue

        feat_norm = np.linalg.norm(feat)
        if feat_norm < 1e-8:
            continue

        # L2-normalize before pooling
        features.append(feat / feat_norm)
        # Weight by detection confidence
        weights.append(det.get("conf", 0.5))

    if not features:
        return np.zeros(feature_dim, dtype=np.float32)

    # Confidence-weighted average
    w = np.array(weights, dtype=np.float32)
    w /= w.sum() + 1e-8
    pooled = sum(wi * fi for wi, fi in zip(w, features))

    # Re-normalize
    pooled_norm = np.linalg.norm(pooled)
    if pooled_norm > 1e-8:
        pooled = pooled / pooled_norm

    return pooled.astype(np.float32)


def create_occupancy_map(
    detections: List[Dict],
    homographies: Dict[Tuple[int, int], np.ndarray],
    tracker_params: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Complete pipeline: Graph clustering + Precision-weighted fusion.

    Args:
        detections: List of detection dicts with z_bev, R_bev, cam_id, etc.
        homographies: Ground-plane homographies (kept for API compatibility)
        tracker_params: Configuration dict

    Returns:
        Dict with:
        - positions: [N, 2] fused BEV positions
        - covariances: List of [2, 2] fused covariances (uncertainty propagated!)
        - clusters: List of detection index lists
        - features: List of pooled SWIN features (if available)
        - stats_info: Clustering statistics
    """
    if not detections:
        return {
            "positions": np.zeros((0, 2), dtype=np.float32),
            "covariances": [],
            "clusters": [],
            "features": [],
            "stats_info": {},
        }

    cfg = DEFAULT_TRACKER_CONFIG.copy()
    if tracker_params:
        cfg.update({k: v for k, v in tracker_params.items() if v is not None})

    mode = cfg.get("mode", "joint")

    prob_thresh = chi2.sf(cfg["mahal_thresh"], df=2)

    clusters, cluster_stats_info = graph_clustering(
        detections,
        homographies,
        prob_thresh=prob_thresh,
        sem_thresh=cfg["sem_thresh"],
        mode=mode,
        allow_single_cam=cfg.get("allow_single_cam", True),
    )

    # Fusion stage: Precision-weighted batch fusion (preserves covariances!)
    positions: List[np.ndarray] = []
    covariances: List[np.ndarray] = []
    features: List[np.ndarray] = []

    for cluster in clusters:
        # Position and covariance fusion
        pos, cov = kalman_fusion(
            cluster,
            detections,
            homographies,
            mode=mode,
            tau_geo=cfg["tau_geo"],
            tau_sem=cfg["tau_sem"],
            lambda_kl=cfg["lambda_kl"],
            alpha=cfg["alpha"],
        )
        positions.append(pos)
        covariances.append(cov)

        # Feature pooling (for re-ID in PHD filter)
        pooled_feat = pool_cluster_features(cluster, detections)
        features.append(pooled_feat)

    result = {
        "positions": np.stack(positions, axis=0) if positions else np.zeros((0, 2), dtype=np.float32),
        "covariances": covariances,
        "clusters": clusters,
        "features": features,
        "stats_info": cluster_stats_info,
    }

    return result
