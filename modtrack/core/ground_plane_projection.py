"""Ground-plane projection and uncertainty propagation utilities."""

import numpy as np


def _symmetrize_covariance(cov: np.ndarray) -> np.ndarray:
    """Return a numerically symmetric covariance matrix."""
    return 0.5 * (cov + cov.T)


def _floor_covariance_eigenvalues(cov: np.ndarray, min_eig: float) -> np.ndarray:
    """Clamp covariance eigenvalues to keep the matrix PSD."""
    cov = _symmetrize_covariance(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, float(min_eig))
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def bev_project(u, v, d_hat, sigma2_d, K, R, t, n=np.array([0.,0.,1.]), d_plane=0.0, R_pose=None, min_variance=None):
    """
    Project pixel coordinates with depth to BEV with uncertainty propagation.

    Args:
        min_variance: Minimum eigenvalue for covariance matrices (m²).
                      If None (default), uses a conservative fixed floor (0.21 m²).
                      Most callers pass a dataset-specific value.
        R_pose: Camera pose/calibration uncertainty (2x2 matrix). If None, uses
                default 0.012 m² diagonal to account for calibration imprecision.
    """
    u = np.atleast_1d(u).astype(float)
    v = np.atleast_1d(v).astype(float)
    d_hat = np.atleast_1d(d_hat).astype(float)
    sigma2_d = np.atleast_1d(sigma2_d).astype(float)
    N = u.shape[0]

    p = np.stack([u, v, np.ones_like(u)], axis=-1)                     # (N,3)
    K_inv = np.linalg.inv(K)
    r_c = (K_inv @ p.T).T                                              # (N,3)
    X_c = d_hat[:, None] * r_c                                         # (N,3)
    X_w = (R @ X_c.T).T + t.reshape(1,3)                               # (N,3)

    n = n / np.linalg.norm(n)
    if np.allclose(n, [0,0,1.]) and np.isclose(d_plane, 0.0):
        X_proj = X_w
    else:
        s = (d_plane - (X_w @ n))                                      # (N,)
        X_proj = X_w + s[:, None] * n                                  # (N,3)

    z_bev = X_proj[:, :2]                                              # (N,2)

    if R_pose is None:
        R_pose = np.eye(2) * 0.012
    R_pose = _symmetrize_covariance(np.asarray(R_pose, dtype=float))
    pose_min_eig = float(np.linalg.eigvalsh(R_pose).min())

    J_base = (R[:2, :] @ K_inv)                                        # (2,3)
    J = (J_base @ p.T).T                                               # (N,2) each row = R_{1:2,:} K^{-1} [u v 1]
    cov = np.zeros((N, 2, 2))
    for i in range(N):
        j = J[i].reshape(2,1)                                          # (2,1)
        cov_depth = sigma2_d[i] * (j @ j.T)  # Depth contribution to BEV covariance

        if min_variance is None:
            adaptive_min = 0.21
        else:
            adaptive_min = min_variance

        indep_min = max(0.0, float(adaptive_min) - pose_min_eig)
        cov_depth = _floor_covariance_eigenvalues(cov_depth, indep_min)
        cov[i] = cov_depth + R_pose

    return z_bev, cov, R_pose


def bev_covariance_from_depth(
    u,
    v,
    sigma2_d,
    K,
    R,
    R_pose=None,
    min_variance=None,
):
    """
    Compute BEV covariance from depth variance using Jacobian propagation.

    Args:
        u, v: Pixel coordinates (arrays or scalars).
        sigma2_d: Depth variance per pixel (arrays or scalars).
        K: 3x3 camera intrinsics.
        R: 3x3 rotation (camera to world).
        R_pose: Optional 2x2 pose/calibration covariance.
        min_variance: Minimum eigenvalue for covariance matrices.

    Returns:
        cov: (N,2,2) BEV covariance matrices.
    """
    u = np.atleast_1d(u).astype(float)
    v = np.atleast_1d(v).astype(float)
    sigma2_d = np.atleast_1d(sigma2_d).astype(float)
    N = u.shape[0]

    p = np.stack([u, v, np.ones_like(u)], axis=-1)                     # (N,3)
    K_inv = np.linalg.inv(K)

    if R_pose is None:
        R_pose = np.eye(2) * 0.012
    R_pose = _symmetrize_covariance(np.asarray(R_pose, dtype=float))
    pose_min_eig = float(np.linalg.eigvalsh(R_pose).min())

    J_base = (R[:2, :] @ K_inv)                                        # (2,3)
    J = (J_base @ p.T).T                                               # (N,2)
    cov = np.zeros((N, 2, 2))

    for i in range(N):
        j = J[i].reshape(2, 1)                                         # (2,1)
        cov_depth = sigma2_d[i] * (j @ j.T)

        if min_variance is None:
            adaptive_min = 0.21
        else:
            adaptive_min = min_variance

        indep_min = max(0.0, float(adaptive_min) - pose_min_eig)
        cov_depth = _floor_covariance_eigenvalues(cov_depth, indep_min)
        cov[i] = cov_depth + R_pose

    return cov


def footpoint_ray_plane(
    bbox,
    K,
    R,
    t,
    n=np.array([0.0, 0.0, 1.0]),
    d_plane=0.0,
    allow_negative_depth=False,
):
    """
    Intersect the footpoint ray with the ground plane.

    Args:
        bbox: (x1, y1, x2, y2) bounding box.
        K: 3x3 camera intrinsics.
        R: 3x3 rotation (camera to world).
        t: 3x1 translation (camera to world).
        n: Plane normal in world coordinates.
        d_plane: Plane offset in world coordinates (n dot X = d_plane).
        allow_negative_depth: If True, accept negative depth intersections
                              (useful for datasets with inverted camera Z).

    Returns:
        Tuple of (u_foot, v_foot, depth_cam, X_w) or (None, None, None, None) on failure.
    """
    x1, y1, x2, y2 = bbox
    u_foot = (x1 + x2) / 2.0
    v_foot = y2

    K_inv = np.linalg.inv(K)
    p_foot = np.array([u_foot, v_foot, 1.0], dtype=float)
    r_c = K_inv @ p_foot

    n = n / np.linalg.norm(n)
    denom = n @ (R @ r_c)
    if abs(denom) < 1e-8:
        return None, None, None, None

    numer = d_plane - (n @ t.reshape(3))
    lam = numer / denom

    if not allow_negative_depth and lam <= 0.0:
        return None, None, None, None
    if allow_negative_depth and lam == 0.0:
        return None, None, None, None

    X_c = lam * r_c
    X_w = R @ X_c + t.reshape(3)

    return float(u_foot), float(v_foot), float(lam), X_w
