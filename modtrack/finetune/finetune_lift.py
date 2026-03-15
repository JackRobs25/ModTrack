#!/usr/bin/env python3
"""Train CamEncode/LIFT with WildTrack GT bounding boxes."""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

from modtrack.data.adapters.multiviewx import MULTIVIEWX_CAMERAS, iter_multiviewx_frames
from modtrack.data.adapters.wildtrack import WILDTRACK_CAMERAS, iter_wildtrack_frames
from modtrack.core.lift_utils import frame_to_tensor as lift_frame_to_tensor, load_camencode_from_lss_ckpt, make_bins as lift_make_bins
from modtrack.core.model import CamEncode
from modtrack.core.paths import ensure_file, weights_root


def boxes_to_depth_map(boxes: np.ndarray, depth: np.ndarray, hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize sparse per-box depths into dense depth and valid-mask maps."""
    H, W = hw
    depth_map = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    if boxes is None or depth is None:
        return depth_map, mask
    boxes = np.asarray(boxes, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    n = min(len(boxes), len(depth))
    if n == 0:
        return depth_map, mask
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        x1_i = int(max(0, np.floor(x1)))
        y1_i = int(max(0, np.floor(y1)))
        x2_i = int(min(W, np.ceil(x2)))
        y2_i = int(min(H, np.ceil(y2)))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue
        depth_map[y1_i:y2_i, x1_i:x2_i] = depth[i]
        mask[y1_i:y2_i, x1_i:x2_i] = True
    return depth_map, mask


def project_world_points(
    K: np.ndarray,
    Tcw: np.ndarray,
    points_world: np.ndarray,
    hw: Tuple[int, int],
    depth_sign: float = 1.0,
    return_indices: bool = False,
):
    """Project world points into image space and return visible depth samples."""
    if points_world is None or len(points_world) == 0:
        return (None, None) if return_indices else None
    if isinstance(hw, np.ndarray):
        if hw.ndim == 1 and hw.size >= 2:
            H, W = int(hw[0]), int(hw[1])
        else:
            raise ValueError(f"Unexpected hw shape for project_world_points: {hw}")
    else:
        H, W = hw
    pts = np.asarray(points_world, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    Twc_inv = np.linalg.inv(Tcw)
    cam = (Twc_inv @ pts_h.T).T[:, :3]
    idxs = np.arange(cam.shape[0], dtype=np.int64)
    depth = cam[:, 2] * float(depth_sign)
    depth_mask = depth > 0
    if not np.any(depth_mask):
        return (None, None) if return_indices else None
    cam = cam[depth_mask]
    depth = depth[depth_mask]
    idxs = idxs[depth_mask]
    K0 = np.asarray(K, dtype=np.float32)
    u = (K0[0, 0] * cam[:, 0] / cam[:, 2]) + K0[0, 2]
    v = (K0[1, 1] * cam[:, 1] / cam[:, 2]) + K0[1, 2]
    iu = np.round(u).astype(np.int32)
    iv = np.round(v).astype(np.int32)
    in_img = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    if not np.any(in_img):
        return (None, None) if return_indices else None
    depth = depth[in_img]
    idxs = idxs[in_img]
    if return_indices:
        return depth, idxs
    return depth


def gaussian_sample_points(box: torch.Tensor, num_samples: int, sigma_scale: float, device: torch.device) -> torch.Tensor:
    """Sample image points from a Gaussian centered inside a bounding box."""
    if num_samples <= 0:
        return torch.zeros((0, 2), device=device, dtype=torch.float32)
    x1, y1, x2, y2 = box.tolist()
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    sigma_x = max(width * sigma_scale, 1e-3)
    sigma_y = max(height * sigma_scale, 1e-3)
    xs = torch.randn(num_samples, device=device) * sigma_x + cx
    ys = torch.randn(num_samples, device=device) * sigma_y + cy
    xs = xs.clamp(min=x1, max=x2)
    ys = ys.clamp(min=y1, max=y2)
    return torch.stack([xs, ys], dim=-1)


def expected_depth_from_logits(logits: torch.Tensor, depth_bins: torch.Tensor) -> torch.Tensor:
    """Compute expected depth from per-bin logits for sampled rays."""
    probs = logits.softmax(dim=-1)
    return torch.sum(probs * depth_bins.view(1, -1), dim=-1)


class LiftSparseDepthDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_kind: str,
        root: str,
        camera: str,
        hw: Tuple[int, int],
        device: torch.device,
        imagenet_norm: bool,
        every_n: int = 1,
        max_frames: int = 0,
    ) -> None:
        super().__init__()
        self.device = device
        self.imagenet_norm = imagenet_norm
        self.H, self.W = hw
        self.dataset_kind = dataset_kind
        if isinstance(camera, str):
            camera_list = [c.strip() for c in camera.split(',') if c.strip()]
        else:
            camera_list = list(camera)
        if not camera_list:
            raise ValueError('Provide at least one camera')
        if dataset_kind == 'wildtrack' and any(c.lower() == 'all' for c in camera_list):
            camera_list = WILDTRACK_CAMERAS[:]
        if dataset_kind == 'multiviewx' and any(c.lower() == 'all' for c in camera_list):
            camera_list = MULTIVIEWX_CAMERAS[:]

        # Eager loading for WildTrack/MultiviewX
        self.samples: List[Dict] = []
        for cam in camera_list:
            if dataset_kind in ('wildtrack', 'multiviewx'):
                iterator_fn = iter_wildtrack_frames if dataset_kind == 'wildtrack' else iter_multiviewx_frames
                iterator = iterator_fn(
                    root,
                    cam,
                    hw,
                    every_n=every_n,
                    max_frames=max_frames,
                    return_token=True,
                )
            else:
                raise ValueError(f'Unsupported dataset: {dataset_kind}')
            for payload in iterator:
                if not payload:
                    continue
                if len(payload) < 9:
                    raise RuntimeError(
                        'Multiview dataset iterator must include boxes, world points, and token for training'
                    )
                *core, token = payload
                img_path, frame_rgb, hw_np, K_img_np, Tcw_np, Tec_np, world_pts, boxes_raw, person_ids = core[:9]
                if world_pts is None or boxes_raw is None or person_ids is None:
                    continue
                projected_depth, keep_idx = project_world_points(
                    K_img_np,
                    Tcw_np,
                    world_pts,
                    hw_np,
                    depth_sign=-1.0 if dataset_kind == "multiviewx" else 1.0,
                    return_indices=True,
                )
                if projected_depth is None or keep_idx is None:
                    continue
                boxes_np = np.asarray(boxes_raw, dtype=np.float32)
                box_ids = np.asarray(person_ids, dtype=np.int32)
                keep_idx = np.asarray(keep_idx, dtype=np.int64)
                boxes_np = boxes_np[keep_idx]
                box_ids = box_ids[keep_idx]
                if boxes_np.shape[0] != projected_depth.shape[0]:
                    raise RuntimeError(
                        f"GT boxes/world points mismatch for {cam} frame {getattr(img_path, 'stem', 'unknown')}"
                    )
                depth_np, mask_np = boxes_to_depth_map(boxes_np, projected_depth, hw_np)
                box_depth_values = projected_depth.astype(np.float32)

                sample: Dict[str, object] = {
                    'rgb': frame_rgb.astype(np.uint8),
                    'hw': hw_np,
                    'depth': depth_np,
                    'mask': mask_np,
                    'K_img': torch.tensor(K_img_np, dtype=torch.float32),
                    'Tcw': torch.tensor(Tcw_np, dtype=torch.float32),
                    'Tec': torch.tensor(Tec_np, dtype=torch.float32) if Tec_np is not None else torch.eye(4, dtype=torch.float32),
                    'token': token,
                    'camera': cam,
                    'boxes': torch.from_numpy(boxes_np.astype(np.float32)),
                    'box_depths': torch.from_numpy(box_depth_values),
                    'box_person_ids': torch.from_numpy(box_ids.astype(np.int32)),
                }
                self.samples.append(sample)
        if not self.samples:
            raise RuntimeError("No samples found for training; check dataset path/camera selection")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_t = lift_frame_to_tensor(sample["rgb"], self.device, imagenet_norm=self.imagenet_norm).squeeze(0)
        boxes = sample.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
        box_depths = sample.get("box_depths", torch.zeros((0,), dtype=torch.float32))
        box_person_ids = sample.get("box_person_ids", torch.zeros((0,), dtype=torch.int32))
        return {
            "image": img_t,
            "rgb": sample["rgb"],
            "depth": torch.from_numpy(sample["depth"]),
            "mask": torch.from_numpy(sample["mask"]),
            "boxes": boxes,
            "box_depths": box_depths,
            "box_person_ids": box_person_ids,
            "K_img": sample["K_img"],
            "Tcw": sample["Tcw"],
            "Tec": sample["Tec"],
            "token": sample["token"],
            "camera": sample["camera"],
            "hw": sample["hw"],
        }

def collate_sparse(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate sparse depth samples while keeping variable-length metadata lists."""
    images = torch.stack([item["image"] for item in batch])
    depth_maps = torch.stack([item["depth"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    return {
        "image": images,
        "depth": depth_maps,
        "mask": masks,
        "rgb": [item["rgb"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "box_depths": [item["box_depths"] for item in batch],
        "box_person_ids": [item["box_person_ids"] for item in batch],
        "K_img": [item["K_img"] for item in batch],
        "Tcw": [item["Tcw"] for item in batch],
        "Tec": [item["Tec"] for item in batch],
        "token": [item["token"] for item in batch],
        "camera": [item["camera"] for item in batch],
        "hw": [item["hw"] for item in batch],
    }


def aggregate_expected_depth(logits: torch.Tensor, points: torch.Tensor, depth_bins: torch.Tensor, img_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    """Aggregate expected depth at sampled points for loss supervision."""
    if points is None or points.numel() == 0:
        return None
    sampled = CamEncode.sample_depth_logits(logits, points, img_hw)
    if sampled.numel() == 0:
        return None
    expected_per_ray = expected_depth_from_logits(sampled, depth_bins)
    return expected_per_ray.mean()


def main() -> None:
    """Train CamEncode with sparse depth supervision and export checkpoints."""
    ap = argparse.ArgumentParser("Train CamEncode on sparse depth cues using GT bounding boxes or projected depth points")
    ap.add_argument("--dataset", choices=["wildtrack", "multiviewx"], required=True)
    ap.add_argument("--wildtrack_root", type=str, default=None)
    ap.add_argument("--multiviewx_root", type=str, default=None)
    ap.add_argument(
        "--camera",
        type=str,
        default='all',
        help="Camera(s) to use (comma-separated or 'all'); default uses all cameras for the selected dataset.",
    )
    ap.add_argument("--every_n", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--depth_bins", type=int, default=41)
    ap.add_argument("--z_min", type=float, default=1.0)
    ap.add_argument("--z_max", type=float, default=36.0)
    ap.add_argument("--bin_scheme", choices=["linear", "log", "inv"], default="linear")
    ap.add_argument("--ctx_dim", type=int, default=64)
    ap.add_argument("--imagenet_norm", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="lift_train_ckpts")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--points_per_box", type=int, default=16, help="Gaussian samples per detected bounding box")
    ap.add_argument("--gaussian_sigma_scale", type=float, default=0.2, help="Gaussian sigma as fraction of box size")
    args = ap.parse_args()

    ckpt_path = (weights_root() / "lss" / "model525000.pt").expanduser().resolve()
    ensure_file(
        ckpt_path,
        "Lift finetune base checkpoint (expected at weights/lss/model525000.pt by default)",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    hw = (args.height, args.width)

    if args.dataset == "wildtrack":
        if not args.wildtrack_root:
            raise SystemExit("--wildtrack_root required for WildTrack training")
        dataset = LiftSparseDepthDataset(
            "wildtrack",
            args.wildtrack_root,
            args.camera,
            hw,
            device,
            args.imagenet_norm,
            every_n=args.every_n,
            max_frames=args.max_frames,
        )
    elif args.dataset == "multiviewx":
        if not args.multiviewx_root:
            raise SystemExit("--multiviewx_root required for MultiviewX training")
        dataset = LiftSparseDepthDataset(
            "multiviewx",
            args.multiviewx_root,
            args.camera,
            hw,
            device,
            args.imagenet_norm,
            every_n=args.every_n,
            max_frames=args.max_frames,
        )
    else:
        raise SystemExit(f"Unsupported dataset: {args.dataset}")

    camera_to_indices: Dict[str, List[int]] = {}
    for idx, sample in enumerate(dataset.samples):
        camera = sample["camera"]
        camera_to_indices.setdefault(camera, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for camera, idxs in camera_to_indices.items():
        if not idxs:
            continue
        if args.val_split <= 0.0:
            split_idx = len(idxs)
        elif args.val_split >= 1.0:
            split_idx = 0
        else:
            num_val = max(1, int(np.ceil(len(idxs) * args.val_split)))
            if len(idxs) > 1:
                num_val = min(num_val, len(idxs) - 1)
            split_idx = len(idxs) - num_val
        train_indices.extend(idxs[:split_idx])
        val_indices.extend(idxs[split_idx:])
        print(f"[split] {camera}: {len(idxs[:split_idx])} train / {len(idxs[split_idx:])} val samples")

    if not val_indices and train_indices:
        val_size = max(1, len(train_indices) // 10)
        val_indices = train_indices[-val_size:]
        train_indices = train_indices[:-val_size]

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    print(f"[data] train dataset size: {len(train_ds)}")
    print(f"[data] val dataset size: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=collate_sparse)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_sparse)

    camenc = CamEncode(D=args.depth_bins, C=args.ctx_dim, downsample=16).to(device)
    load_camencode_from_lss_ckpt(camenc, str(ckpt_path), device)

    params = list(camenc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    depth_bins = lift_make_bins(args.depth_bins, args.z_min, args.z_max, args.bin_scheme, device)

    # --- wandb init ---
    if _HAS_WANDB and os.environ.get("WANDB_API_KEY"):
        wandb.init(
            project="modtrack-lift",
            name=f"lift-{args.dataset}-z{args.z_max}-lr{args.lr}",
            config=vars(args),
        )
    _use_wandb = _HAS_WANDB and wandb.run is not None

    best_val = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    match_log_path = Path(args.out_dir) / "box_supervision.csv"
    match_log_file = match_log_path.open("w")
    match_log_file.write("phase,epoch,batch_idx,num_boxes,num_supervised\n")

    try:
        for epoch in range(args.epochs):
            camenc.train()
            running_abs = 0.0
            running_cnt = 0
            train_iter = tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}", leave=False)
            for batch_idx, batch in enumerate(train_iter):
                x = batch["image"].to(device)
                gt_boxes_list = batch["boxes"]
                gt_box_depths_list = batch["box_depths"]
                img_hw_list = [tuple(int(v) for v in hw_pair) for hw_pair in batch["hw"]]

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    depth_logits_batch, _ = camenc.forward_depth_head(x)
                    preds: List[torch.Tensor] = []
                    targets: List[torch.Tensor] = []
                    batch_box_total = 0
                    batch_supervised = 0
                    for b_idx, logits_b in enumerate(depth_logits_batch):
                        boxes = gt_boxes_list[b_idx]
                        depths = gt_box_depths_list[b_idx]
                        if boxes is None or boxes.numel() == 0 or depths is None or depths.numel() == 0:
                            continue
                        src_hw = img_hw_list[b_idx]
                        batch_box_total += boxes.shape[0]
                        for box_tensor, depth_val in zip(boxes, depths):
                            coords = gaussian_sample_points(
                                box_tensor.to(device),
                                args.points_per_box,
                                args.gaussian_sigma_scale,
                                device,
                            )
                            aggregated = aggregate_expected_depth(logits_b, coords, depth_bins, src_hw)
                            if aggregated is None:
                                continue
                            preds.append(aggregated)
                            targets.append(depth_val.to(device))
                            batch_supervised += 1
                match_log_file.write(f"train,{epoch + 1},{batch_idx},{batch_box_total},{batch_supervised}\n")
                match_log_file.flush()

                if not preds:
                    continue
                preds_t = torch.cat(preds) if preds[0].dim() > 0 else torch.stack(preds)
                targets_t = torch.cat(targets) if targets[0].dim() > 0 else torch.stack(targets)
                loss = F.l1_loss(preds_t, targets_t, reduction="mean")
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(params, args.grad_clip)
                scaler.step(opt)
                scaler.update()
                diff = torch.abs(preds_t.detach() - targets_t.detach())
                running_abs += diff.sum().item()
                running_cnt += diff.numel()
            avg_loss = running_abs / max(1, running_cnt)
            train_loss_history.append(avg_loss)
            print(f"[train] epoch {epoch+1}/{args.epochs} l1={avg_loss:.4f}")
            if _use_wandb:
                wandb.log({"train/l1_loss": avg_loss, "epoch": epoch + 1})

            camenc.eval()
            with torch.no_grad():
                val_abs = 0.0
                val_cnt = 0
                val_iter = tqdm(val_loader, desc=f"val {epoch+1}/{args.epochs}", leave=False)
                for batch_idx, batch in enumerate(val_iter):
                    x = batch["image"].to(device)
                    gt_boxes_list = batch["boxes"]
                    gt_box_depths_list = batch["box_depths"]
                    img_hw_list = [tuple(int(v) for v in hw_pair) for hw_pair in batch["hw"]]

                    depth_logits_batch, _ = camenc.forward_depth_head(x)
                    batch_box_total = 0
                    batch_supervised = 0
                    for b_idx, logits_b in enumerate(depth_logits_batch):
                        boxes = gt_boxes_list[b_idx]
                        depths = gt_box_depths_list[b_idx]
                        if boxes is None or boxes.numel() == 0 or depths is None or depths.numel() == 0:
                            continue
                        src_hw = img_hw_list[b_idx]
                        batch_box_total += boxes.shape[0]
                        for box_tensor, depth_val in zip(boxes, depths):
                            coords = gaussian_sample_points(
                                box_tensor.to(device),
                                args.points_per_box,
                                args.gaussian_sigma_scale,
                                device,
                            )
                            aggregated = aggregate_expected_depth(logits_b, coords, depth_bins, src_hw)
                            if aggregated is None:
                                continue
                            diff = torch.abs(aggregated - depth_val.to(device))
                            val_abs += diff.item()
                            val_cnt += 1
                            batch_supervised += 1

                    match_log_file.write(f"val,{epoch + 1},{batch_idx},{batch_box_total},{batch_supervised}\n")
                    match_log_file.flush()
                val_loss = val_abs / max(1, val_cnt)
            val_loss_history.append(val_loss)
            print(f"[val]   epoch {epoch+1}/{args.epochs} l1={val_loss:.4f}")
            if _use_wandb:
                wandb.log({"val/l1_loss": val_loss, "epoch": epoch + 1})

            torch.save(
                {"camencode": camenc.state_dict(), "args": vars(args), "epoch": epoch + 1, "val_loss": val_loss},
                os.path.join(args.out_dir, f"last_{args.dataset}.pt"),
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"camencode": camenc.state_dict(), "args": vars(args), "epoch": epoch + 1, "val_loss": val_loss},
                    os.path.join(args.out_dir, f"best_{args.dataset}.pt"),
                )
    finally:
        match_log_file.close()

    print(f"Done. Best val l1={best_val:.4f}")
    if _use_wandb:
        wandb.log({"best_val_l1": best_val})
        wandb.finish()


if __name__ == "__main__":
    main()
