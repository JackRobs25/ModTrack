"""
Depth estimation using Lift CamEncode.
"""

import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

from modtrack.core.model import CamEncode
from modtrack.core.lift_utils import (
    expected_depth_from_probs,
    sample_depth_probs_at_uv,
    gaussian_sample_points_with_weights,
    load_camencode_from_lss_ckpt,
    frame_to_tensor,
)
from modtrack.core.tracker.config import get_footpoint_min_depth_var


class CamEncodeDepthEstimator:

    def __init__(
        self,
        checkpoint_path: str,
        depth_bins: torch.Tensor,
        ctx_dim: int = 64,
        depth_sample_count: int = 16,
        depth_sample_sigma_scale: float = 0.2,
        depth_prob_power: float = 1.0,
        pedestrian_height_m: Optional[float] = None,
        depth_prior_min_range_m: Optional[float] = None,
        depth_prior_sigma_scale: float = 0.05,
        depth_var_cutoff: Optional[float] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.depth_bins = depth_bins.to(self.device)
        self.D = len(depth_bins)
        self.depth_sample_count = int(depth_sample_count)
        self.depth_sample_sigma_scale = float(depth_sample_sigma_scale)
        self.depth_prob_power = float(depth_prob_power)
        self.pedestrian_height_m = pedestrian_height_m
        self.depth_prior_min_range_m = depth_prior_min_range_m
        self.depth_prior_sigma_scale = float(depth_prior_sigma_scale)
        self.depth_var_cutoff = depth_var_cutoff

        # Initialize CamEncode model
        self.model = CamEncode(D=self.D, C=ctx_dim, downsample=16).to(self.device)

        # Load checkpoint
        if Path(checkpoint_path).exists():
            load_camencode_from_lss_ckpt(self.model, checkpoint_path, self.device)
            print(f"[DepthEstimator] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[DepthEstimator] Warning: checkpoint not found, using ImageNet init")

        self.model.eval()

    @torch.no_grad()
    def run_inference(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        img_tensor = frame_to_tensor(image, self.device, imagenet_norm=True)
        depth_probs, ctx = self.model(img_tensor)
        return depth_probs, ctx

    @torch.no_grad()
    def run_inference_batch(self, images: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not images:
            raise ValueError("run_inference_batch requires at least one image")

        img_batch = torch.cat(
            [frame_to_tensor(image, self.device, imagenet_norm=True) for image in images],
            dim=0,
        )
        depth_probs, ctx = self.model(img_batch)
        return depth_probs, ctx

    def estimate_depth_at_detections(
        self,
        depth_probs: torch.Tensor,
        detections: List[Dict],
        image_hw: Tuple[int, int],
        K: Optional[np.ndarray] = None,
        cutoff_dets: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        B, D, _Hf, _Wf = depth_probs.shape
        assert B == 1, "Batch size must be 1"
        if D != self.D:
            raise ValueError(f"Depth bin mismatch: depth_probs has D={D}, expected {self.D}")

        H, W = image_hw
        depth_bins = self.depth_bins.view(1, D)  # [1, D]
        n = max(self.depth_sample_count, 1)
        sigma_scale = float(self.depth_sample_sigma_scale)
        prob_power = float(self.depth_prob_power)
        log_d = float(np.log(float(D)))
        trim_quantile = 0.75  
        min_keep = max(4, n // 4)  
        min_depth_var = get_footpoint_min_depth_var()
        filtered: List[Dict] = []
        det_entries: List[Tuple[Dict, Tuple[float, float, float, float], torch.Tensor, torch.Tensor]] = []
        all_coords: List[torch.Tensor] = []

        for det in detections:
            bbox = det.get("bbox")
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = map(float, bbox)
            else:
                u = float(det["pixel_u"])
                v = float(det["pixel_v"])
                bw = float(det["box_width"])
                bh = float(det["box_height"])
                x1, y1, x2, y2 = u - bw / 2.0, v - bh / 2.0, u + bw / 2.0, v + bh / 2.0

            x1 = float(np.clip(x1, 0.0, float(W - 1)))
            x2 = float(np.clip(x2, 0.0, float(W - 1)))
            y1 = float(np.clip(y1, 0.0, float(H - 1)))
            y2 = float(np.clip(y2, 0.0, float(H - 1)))
            if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                continue

            box_t = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=self.device)
            coords, w_norm = gaussian_sample_points_with_weights(
                box_t,
                n,
                sigma_scale=sigma_scale,
                device=self.device,
            )  # coords: [n,2], w_norm: [n], sum(w_norm)=1
            if coords.numel() == 0:
                continue

            det_entries.append((det, (x1, y1, x2, y2), coords, w_norm))
            all_coords.append(coords)

        if not det_entries:
            return filtered

        sampled_probs_all = sample_depth_probs_at_uv(depth_probs, torch.cat(all_coords, dim=0), image_hw)
        if sampled_probs_all.numel() == 0:
            return filtered
        p_sum = sampled_probs_all.sum(dim=1, keepdim=True).clamp_min(1e-8)
        sampled_probs_all = sampled_probs_all / p_sum
        if prob_power != 1.0:
            sampled_probs_all = sampled_probs_all.clamp_min(1e-12).pow(prob_power)
            sampled_probs_all = sampled_probs_all / sampled_probs_all.sum(dim=1, keepdim=True).clamp_min(1e-8)

        k_det = len(det_entries)
        sampled_probs = sampled_probs_all.view(k_det, n, D)  # [K, n, D]
        coords = torch.stack([entry[2] for entry in det_entries], dim=0)  # [K, n, 2]
        w_norm = torch.stack([entry[3] for entry in det_entries], dim=0)  # [K, n]

        depth_bins_batched = depth_bins.view(1, 1, D)  # [1, 1, D]
        d_hat_raw = (sampled_probs * depth_bins_batched).sum(dim=2)  # [K, n]
        d_hat = d_hat_raw
        sigma2 = (sampled_probs * (depth_bins_batched - d_hat[:, :, None]) ** 2).sum(dim=2)  # [K, n]
        sample_max_prob = sampled_probs.max(dim=2).values  # [K, n]
        sample_entropy = -(sampled_probs * (sampled_probs + 1e-8).log()).sum(dim=2)  # [K, n]
        entropy_norm = sample_entropy / log_d
        conf_weight = sample_max_prob * (1.0 - entropy_norm.clamp(0.0, 1.0))

        keep_thresh = torch.quantile(sigma2, trim_quantile, dim=1, keepdim=True)  # [K,1]
        keep_mask = sigma2 <= keep_thresh
        keep_mask[keep_mask.sum(dim=1) < min_keep] = True

        # Robust aggregation: downweight uncertain samples and trim high-variance outliers.
        w = w_norm * conf_weight
        w = w * keep_mask.to(dtype=w.dtype)
        wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        invalid_rows = (wsum <= 0.0).squeeze(1)
        if torch.any(invalid_rows):
            w[invalid_rows] = w_norm[invalid_rows]
            wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        w = w / wsum

        det_depth_lift = (w * d_hat).sum(dim=1)  # [K]
        within_var = (w * sigma2).sum(dim=1)
        between_var = (w * (d_hat - det_depth_lift[:, None]) ** 2).sum(dim=1)
        det_var_lift = (within_var + between_var).clamp_min(min_depth_var)

        p_lift = (w[:, :, None] * sampled_probs).sum(dim=1)  # [K, D]
        p_lift = p_lift / (p_lift.sum(dim=1, keepdim=True) + 1e-8)
        det_depth_raw = (w_norm * d_hat_raw).sum(dim=1)  # [K]
        det_u = (w * coords[:, :, 0]).sum(dim=1)  # [K]
        det_v = (w * coords[:, :, 1]).sum(dim=1)  # [K]

        for det_idx, (det, (x1, y1, x2, y2), _coords, _w_norm) in enumerate(det_entries):
            det_depth_lift_i = det_depth_lift[det_idx]
            det_var_lift_i = det_var_lift[det_idx]
            p_lift_i = p_lift[det_idx]

            det_depth = det_depth_lift_i
            det_var = det_var_lift_i
            p_bar = p_lift_i

            det["depth_lift"] = float(det_depth_lift_i.item())
            det["depth_var_lift"] = float(det_var_lift_i.item())

            if (
                self.pedestrian_height_m is not None
                and self.depth_prior_min_range_m is not None
                and K is not None
            ):
                det_depth_raw_i = float(det_depth_raw[det_idx].item())
                if det_depth_raw_i >= float(self.depth_prior_min_range_m):
                    h_px = float(det.get("box_height", y2 - y1))
                    h_px = max(h_px, 1.0)
                    fy = float(K[1, 1])
                    if fy > 0.0:
                        proxy_depth = fy * float(self.pedestrian_height_m) / h_px
                        if proxy_depth > 0.0:
                            det["bbox_proxy_depth_m"] = float(proxy_depth)
                            sigma = max(self.depth_prior_sigma_scale * proxy_depth, 1e-3)
                            p_proxy = torch.exp(-0.5 * ((depth_bins - proxy_depth) / sigma) ** 2)
                            p_proxy = p_proxy / (p_proxy.sum() + 1e-8)
                            proxy_var = float(max(sigma ** 2, min_depth_var))
                            det["bbox_proxy_depth_var"] = float(proxy_var)
                            lift_var = float(max(float(det_var_lift_i.item()), min_depth_var))
                            lift_conf = 1.0 / lift_var
                            proxy_conf = 1.0 / proxy_var
                            alpha = lift_conf / (lift_conf + proxy_conf)
                            p_bar = p_lift_i * float(alpha) + p_proxy * float(1.0 - alpha)
                            p_bar = p_bar / (p_bar.sum() + 1e-8)
                            det_depth = (p_bar * depth_bins).sum()
                            det_var = (p_bar * (depth_bins - det_depth) ** 2).sum().clamp_min(min_depth_var)
            max_prob = float(p_bar.max().item())
            entropy = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())

            det_u_i = det_u[det_idx]
            det_v_i = det_v[det_idx]

            det["pixel_u"] = float(det_u_i.item())
            det["pixel_v"] = float(det_v_i.item())
            det["depth"] = float(det_depth.item())
            det["depth_var"] = float(max(det_var.item(), min_depth_var))
            det["depth_cutoff"] = False

            if (
                self.depth_var_cutoff is not None
                and det["depth_var"] > self.depth_var_cutoff
            ):
                det["depth_cutoff"] = True
                if cutoff_dets is not None:
                    cutoff_dets.append(det)
                cutoff_parts = []
                det_label = det.get("det_label")
                if det_label is not None:
                    cutoff_parts.append(f"id={det_label}")
                cam_id = det.get("cam_id")
                if cam_id is not None:
                    cutoff_parts.append(f"cam={cam_id}")
                frame_idx = det.get("frame_idx")
                if frame_idx is not None:
                    cutoff_parts.append(f"frame={frame_idx}")
                cutoff_suffix = f" {' '.join(cutoff_parts)}" if cutoff_parts else ""
                continue

            filtered.append(det)

        return filtered


def evaluate_depth_prediction(
    detections: List[Dict[str, Any]],
    gt_points_by_cam: Dict[int, np.ndarray],
    *,
    max_match_dist_m: float = 1.0,
    require_covariance_for_nll: bool = False,
) -> Dict[str, Any]:

    def _as_xy(arr: Any) -> Optional[np.ndarray]:
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            return a.reshape(0, 2)
        if a.ndim == 1:
            if a.shape[0] < 2:
                return None
            return a[:2].reshape(1, 2)
        if a.ndim == 2:
            if a.shape[1] < 2:
                return None
            return a[:, :2]
        return None

    def _try_hungarian(cost: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore

            return linear_sum_assignment(cost)
        except Exception:
            return None

    def _greedy_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if cost.size == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
        det_n, gt_n = cost.shape
        flat = np.argsort(cost.reshape(-1))
        used_det = np.zeros(det_n, dtype=bool)
        used_gt = np.zeros(gt_n, dtype=bool)
        det_idx: List[int] = []
        gt_idx: List[int] = []
        for k in flat:
            i = int(k // gt_n)
            j = int(k % gt_n)
            if used_det[i] or used_gt[j]:
                continue
            used_det[i] = True
            used_gt[j] = True
            det_idx.append(i)
            gt_idx.append(j)
            if used_det.all() or used_gt.all():
                break
        return np.asarray(det_idx, dtype=np.int64), np.asarray(gt_idx, dtype=np.int64)

    def _safe_inv_2x2(mat: np.ndarray) -> Optional[np.ndarray]:
        m = np.asarray(mat, dtype=np.float64).reshape(2, 2)
        det = float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
        if not np.isfinite(det) or abs(det) < 1e-12:
            return None
        inv = np.array([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]], dtype=np.float64) / det
        return inv

    def _gaussian_nll_2d(delta: np.ndarray, cov: np.ndarray) -> Optional[float]:
        inv = _safe_inv_2x2(cov)
        if inv is None:
            return None
        cov64 = np.asarray(cov, dtype=np.float64).reshape(2, 2)
        det = float(cov64[0, 0] * cov64[1, 1] - cov64[0, 1] * cov64[1, 0])
        if not np.isfinite(det) or det <= 0.0:
            return None
        d = np.asarray(delta, dtype=np.float64).reshape(2, 1)
        maha2 = float((d.T @ inv @ d).squeeze())
        if not np.isfinite(maha2):
            return None
        return 0.5 * (maha2 + np.log(det) + 2.0 * np.log(2.0 * np.pi))

    dets_by_cam: Dict[int, List[Dict[str, Any]]] = {}
    for det in detections:
        cam_id = det.get("cam_id")
        if cam_id is None:
            continue
        z_bev = det.get("z_bev")
        if z_bev is None:
            continue
        z_arr = np.asarray(z_bev).reshape(-1)
        if z_arr.size < 2:
            continue
        dets_by_cam.setdefault(int(cam_id), []).append(det)

    per_det_rows: List[Dict[str, Any]] = []
    per_cam_summary: Dict[int, Dict[str, Any]] = {}

    total_dets = 0
    total_gt = 0
    total_matched = 0
    matched_errs: List[float] = []
    matched_maha: List[float] = []
    matched_nll: List[float] = []

    for cam_id, cam_dets in sorted(dets_by_cam.items()):
        gt_xy = _as_xy(gt_points_by_cam.get(cam_id))
        det_xy = np.asarray(
            [np.asarray(d.get("z_bev"), dtype=np.float32).reshape(-1)[:2] for d in cam_dets],
            dtype=np.float32,
        )
        if det_xy.ndim != 2 or det_xy.shape[1] != 2:
            continue

        cam_total_dets = int(det_xy.shape[0])
        total_dets += cam_total_dets

        if gt_xy is None:
            for det in cam_dets:
                pred = np.asarray(det.get("z_bev"), dtype=np.float32).reshape(-1)[:2]
                per_det_rows.append(
                    {
                        "cam_id": cam_id,
                        "det_label": det.get("det_label"),
                        "matched": False,
                        "gt_missing_for_cam": True,
                        "pred_x_m": float(pred[0]),
                        "pred_y_m": float(pred[1]),
                    }
                )
            per_cam_summary[cam_id] = {
                "cam_id": cam_id,
                "num_dets": cam_total_dets,
                "num_gt": 0,
                "num_matched": 0,
                "mean_err_dist_m": None,
            }
            continue

        cam_total_gt = int(gt_xy.shape[0])
        total_gt += cam_total_gt

        if cam_total_gt == 0 or cam_total_dets == 0:
            for det in cam_dets:
                pred = np.asarray(det.get("z_bev"), dtype=np.float32).reshape(-1)[:2]
                per_det_rows.append(
                    {
                        "cam_id": cam_id,
                        "det_label": det.get("det_label"),
                        "matched": False,
                        "gt_missing_for_cam": cam_total_gt == 0,
                        "pred_x_m": float(pred[0]),
                        "pred_y_m": float(pred[1]),
                    }
                )
            per_cam_summary[cam_id] = {
                "cam_id": cam_id,
                "num_dets": cam_total_dets,
                "num_gt": cam_total_gt,
                "num_matched": 0,
                "mean_err_dist_m": None,
            }
            continue

        cost = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=2)
        match = _try_hungarian(cost)
        if match is None:
            det_idx, gt_idx = _greedy_assignment(cost)
        else:
            det_idx, gt_idx = match

        cam_matched = 0
        cam_errs: List[float] = []
        cam_maha: List[float] = []
        cam_nll: List[float] = []

        used_det = set()
        used_gt = set()

        for i_det, i_gt in zip(det_idx, gt_idx):
            dist = float(cost[int(i_det), int(i_gt)])
            if dist > float(max_match_dist_m):
                continue
            det = cam_dets[int(i_det)]
            pred = det_xy[int(i_det)]
            gt = gt_xy[int(i_gt)]
            err = pred - gt
            err_dist = float(np.linalg.norm(err))
            row: Dict[str, Any] = {
                "cam_id": cam_id,
                "det_label": det.get("det_label"),
                "matched": True,
                "gt_missing_for_cam": False,
                "pred_x_m": float(pred[0]),
                "pred_y_m": float(pred[1]),
                "gt_x_m": float(gt[0]),
                "gt_y_m": float(gt[1]),
                "err_x_m": float(err[0]),
                "err_y_m": float(err[1]),
                "err_dist_m": err_dist,
            }
            maha_val = None
            nll_val = None
            R_bev = det.get("R_bev")
            if R_bev is not None:
                cov = np.asarray(R_bev, dtype=np.float64).reshape(2, 2)
                inv = _safe_inv_2x2(cov)
                if inv is not None:
                    delta = err.reshape(2, 1).astype(np.float64)
                    maha_val = float((delta.T @ inv @ delta).squeeze())
                nll_val = _gaussian_nll_2d(err, cov)
            if maha_val is not None:
                row["mahalanobis"] = float(maha_val)
                cam_maha.append(float(maha_val))
                matched_maha.append(float(maha_val))
            if nll_val is not None:
                row["nll"] = float(nll_val)
                cam_nll.append(float(nll_val))
                matched_nll.append(float(nll_val))
            per_det_rows.append(row)
            cam_matched += 1
            total_matched += 1
            cam_errs.append(err_dist)
            matched_errs.append(err_dist)
            used_det.add(int(i_det))
            used_gt.add(int(i_gt))

        for i_det, det in enumerate(cam_dets):
            if i_det in used_det:
                continue
            pred = det_xy[i_det]
            per_det_rows.append(
                {
                    "cam_id": cam_id,
                    "det_label": det.get("det_label"),
                    "matched": False,
                    "gt_missing_for_cam": False,
                    "pred_x_m": float(pred[0]),
                    "pred_y_m": float(pred[1]),
                }
            )

        per_cam_summary[cam_id] = {
            "cam_id": cam_id,
            "num_dets": cam_total_dets,
            "num_gt": cam_total_gt,
            "num_matched": cam_matched,
            "mean_err_dist_m": float(np.mean(cam_errs)) if cam_errs else None,
            "rmse_err_dist_m": float(np.sqrt(np.mean(np.square(cam_errs)))) if cam_errs else None,
            "mean_mahalanobis": float(np.mean(cam_maha)) if cam_maha else None,
            "mean_nll": float(np.mean(cam_nll)) if cam_nll else None,
        }

    if matched_errs:
        mean_err = float(np.mean(matched_errs))
        rmse_err = float(np.sqrt(np.mean(np.square(matched_errs))))
    else:
        mean_err = None
        rmse_err = None

    summary: Dict[str, Any] = {
        "num_dets": int(total_dets),
        "num_gt": int(total_gt),
        "num_matched": int(total_matched),
        "mean_err_dist_m": mean_err,
        "rmse_err_dist_m": rmse_err,
        "per_cam": per_cam_summary,
    }

    if not require_covariance_for_nll:
        if matched_maha:
            summary["mean_mahalanobis"] = float(np.mean(matched_maha))
        if matched_nll:
            summary["mean_nll"] = float(np.mean(matched_nll))

    return {
        "per_detection": per_det_rows,
        "summary": summary,
    }
