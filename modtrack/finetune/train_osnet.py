"""OSNet-based semantic encoder training entrypoint for ModTrack."""

import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from modtrack.finetune.semantics_dataset_registry import dataset_choices, resolve_crops_root


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_frame_index(path: Path) -> Optional[int]:
    stem = path.stem
    token = None
    if stem.startswith("frame_"):
        token = stem[len("frame_") :]
    else:
        matches = re.findall(r"\d+", stem)
        if matches:
            token = matches[-1]
    if not token:
        return None
    try:
        return int(token)
    except (TypeError, ValueError):
        return None


def _collect_frame_indices(root_dir: Path) -> List[int]:
    indices = set()
    for person_dir in sorted(root_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for cam_dir in sorted(person_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            for img_path in cam_dir.glob("*.jpg"):
                idx = _extract_frame_index(img_path)
                if idx is not None:
                    indices.add(idx)
    return sorted(indices)


def _parse_cam_id(cam_name: str) -> int:
    digits = re.findall(r"\d+", cam_name)
    if digits:
        return int(digits[-1])
    return 0


class OSNetPersonDataset(Dataset):
    """Supervised person ReID dataset from person_crops/person_id/camera/*.jpg."""

    def __init__(
        self,
        root_dir: Path,
        split: str,
        train_ratio: float,
        image_height: int,
        image_width: int,
        train_transform: bool,
        min_cameras: int = 2,
        max_images_per_pid: int = 0,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.train_ratio = float(train_ratio)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.min_cameras = int(min_cameras)
        self.max_images_per_pid = int(max_images_per_pid)

        if train_transform:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_height, self.image_width)),
                    transforms.Pad(10),
                    transforms.RandomCrop((self.image_height, self.image_width)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_height, self.image_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        self.all_frames = _collect_frame_indices(self.root_dir)
        self.allowed_frames = self._build_split_frames()

        self.records: List[Tuple[str, int, int]] = []
        self.pid_to_indices: Dict[int, List[int]] = defaultdict(list)
        self.pid_to_cam_to_indices: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

        self._build_records()

    def _build_split_frames(self) -> Optional[set]:
        if not self.all_frames or self.split == "all":
            return None
        split_idx = max(1, int(len(self.all_frames) * self.train_ratio))
        if self.split == "train":
            return set(self.all_frames[:split_idx])
        if self.split == "val":
            return set(self.all_frames[split_idx:])
        raise ValueError(f"Unknown split: {self.split}")

    def _build_records(self) -> None:
        pid_names = [p for p in sorted(self.root_dir.iterdir()) if p.is_dir()]
        pid_name_to_label = {pid_dir.name: idx for idx, pid_dir in enumerate(pid_names)}

        for pid_dir in pid_names:
            pid_label = pid_name_to_label[pid_dir.name]
            cam_images: Dict[int, List[str]] = defaultdict(list)

            for cam_dir in sorted(pid_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue
                cam_id = _parse_cam_id(cam_dir.name)
                for img_path in sorted(cam_dir.glob("*.jpg")):
                    frame_idx = _extract_frame_index(img_path)
                    if self.allowed_frames is not None:
                        if frame_idx is None or frame_idx not in self.allowed_frames:
                            continue
                    cam_images[cam_id].append(str(img_path))

            if len(cam_images) < self.min_cameras:
                continue

            candidate = []
            for cam_id, paths in cam_images.items():
                for p in paths:
                    candidate.append((p, pid_label, cam_id))

            if len(candidate) < 2:
                continue

            if self.max_images_per_pid > 0 and len(candidate) > self.max_images_per_pid:
                candidate = random.sample(candidate, self.max_images_per_pid)

            for rec in candidate:
                index = len(self.records)
                self.records.append(rec)
                self.pid_to_indices[pid_label].append(index)
                self.pid_to_cam_to_indices[pid_label][rec[2]].append(index)

        # Filter out IDs with <2 samples after split
        keep_indices = []
        valid_pids = {pid for pid, idxs in self.pid_to_indices.items() if len(idxs) >= 2}
        for idx, (_, pid, _) in enumerate(self.records):
            if pid in valid_pids:
                keep_indices.append(idx)

        if len(keep_indices) != len(self.records):
            new_records = []
            new_pid_to_indices: Dict[int, List[int]] = defaultdict(list)
            new_pid_to_cam_to_indices: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
            for old_idx in keep_indices:
                path, pid, cam = self.records[old_idx]
                new_idx = len(new_records)
                new_records.append((path, pid, cam))
                new_pid_to_indices[pid].append(new_idx)
                new_pid_to_cam_to_indices[pid][cam].append(new_idx)
            self.records = new_records
            self.pid_to_indices = new_pid_to_indices
            self.pid_to_cam_to_indices = new_pid_to_cam_to_indices

        # Reindex PID labels to contiguous [0, num_ids-1]. This is required
        # for CE-based methods (e.g., BoT) where labels must be < num_classes.
        if self.records:
            uniq_pids = sorted(self.pid_to_indices.keys())
            pid_remap = {old_pid: new_pid for new_pid, old_pid in enumerate(uniq_pids)}

            if any(pid_remap[old_pid] != old_pid for old_pid in uniq_pids):
                remapped_records = []
                remapped_pid_to_indices: Dict[int, List[int]] = defaultdict(list)
                remapped_pid_to_cam_to_indices: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

                for path, old_pid, cam in self.records:
                    new_pid = pid_remap[old_pid]
                    new_idx = len(remapped_records)
                    remapped_records.append((path, new_pid, cam))
                    remapped_pid_to_indices[new_pid].append(new_idx)
                    remapped_pid_to_cam_to_indices[new_pid][cam].append(new_idx)

                self.records = remapped_records
                self.pid_to_indices = remapped_pid_to_indices
                self.pid_to_cam_to_indices = remapped_pid_to_cam_to_indices

    @property
    def num_ids(self) -> int:
        return len(self.pid_to_indices)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, pid, camid = self.records[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, pid, camid


class PKSampler(Sampler[List[int]]):
    """Sample P identities x K images each per batch."""

    def __init__(
        self,
        dataset: OSNetPersonDataset,
        batch_size: int,
        instances_per_pid: int,
        steps_per_epoch: int,
        cross_camera_only: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.instances_per_pid = int(instances_per_pid)
        self.steps_per_epoch = int(steps_per_epoch)
        self.cross_camera_only = bool(cross_camera_only)

        if self.instances_per_pid < 2:
            raise ValueError("instances_per_pid must be >= 2 for contrastive learning")

        self.pids = list(dataset.pid_to_indices.keys())
        self.num_pids_per_batch = max(1, self.batch_size // self.instances_per_pid)

        if len(self.pids) < 2:
            raise ValueError("Need at least 2 identities for supervised ReID training")
        if self.cross_camera_only:
            insufficient = [
                pid for pid in self.pids
                if len(self.dataset.pid_to_cam_to_indices[pid]) < self.instances_per_pid
            ]
            if insufficient:
                raise ValueError(
                    "cross_camera_only=True requires each PID to have at least "
                    f"instances_per_pid={self.instances_per_pid} distinct cameras. "
                    f"Found {len(insufficient)} insufficient PIDs."
                )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            if len(self.pids) >= self.num_pids_per_batch:
                batch_pids = random.sample(self.pids, self.num_pids_per_batch)
            else:
                batch_pids = random.choices(self.pids, k=self.num_pids_per_batch)

            batch = []
            for pid in batch_pids:
                if self.cross_camera_only:
                    cam_to_indices = self.dataset.pid_to_cam_to_indices[pid]
                    cam_ids = list(cam_to_indices.keys())
                    chosen_cams = random.sample(cam_ids, self.instances_per_pid)
                    chosen = [random.choice(cam_to_indices[cam]) for cam in chosen_cams]
                else:
                    idxs = self.dataset.pid_to_indices[pid]
                    if len(idxs) >= self.instances_per_pid:
                        chosen = random.sample(idxs, self.instances_per_pid)
                    else:
                        chosen = random.choices(idxs, k=self.instances_per_pid)
                batch.extend(chosen)

            yield batch


def _import_torchreid_or_exit() -> object:
    try:
        import torchreid
    except ImportError as exc:
        raise SystemExit(
            "Failed to import torchreid (or one of its dependencies).\n"
            f"Import error: {exc}\n"
            "Install/repair dependencies, e.g.\n"
            "  source Lift/.venv/bin/activate\n"
            "  pip install torchreid\n"
            "  pip install tensorboard\n"
            "or install from the repo: https://github.com/KaiyangZhou/deep-person-reid"
        ) from exc
    return torchreid


def _build_osnet_model(torchreid_module, arch: str, pretrained: bool, num_classes: int) -> nn.Module:
    build_fn = getattr(torchreid_module.models, "build_model", None)
    if build_fn is None:
        raise RuntimeError("torchreid.models.build_model not found")

    trials = [
        {"name": arch, "num_classes": int(num_classes), "loss": "triplet", "pretrained": pretrained, "use_gpu": torch.cuda.is_available()},
        {"name": arch, "num_classes": int(num_classes), "loss": "triplet", "pretrained": pretrained},
        {"name": arch, "num_classes": int(num_classes), "loss": "triplet"},
    ]

    last_error: Optional[Exception] = None
    for kwargs in trials:
        try:
            model = build_fn(**kwargs)
            return model
        except Exception as exc:  # pragma: no cover - fallback trials
            last_error = exc
            continue

    raise RuntimeError(f"Could not build OSNet model '{arch}'. Last error: {last_error}")


def _extract_logits_and_feature_tensors(model_out: object, batch_size: int) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if torch.is_tensor(model_out):
        if model_out.ndim < 2:
            raise RuntimeError(f"Model output tensor has invalid shape: {tuple(model_out.shape)}")
        return None, model_out

    if isinstance(model_out, (tuple, list)):
        tensors = [x for x in model_out if torch.is_tensor(x) and x.ndim == 2 and x.shape[0] == batch_size]
        if len(tensors) >= 2:
            # torchreid triplet models return (logits, features) during training.
            return tensors[0], tensors[1]
        if len(tensors) == 1:
            return None, tensors[0]

    raise RuntimeError(f"Unsupported model output type: {type(model_out)}")


def cross_camera_triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    camids: torch.Tensor,
    margin: float = 0.3,
    cross_camera_positives: bool = True,
    cross_camera_negatives: bool = True,
) -> torch.Tensor:
    """Compute batch-hard triplet loss with optional cross-camera constraints."""
    n = features.shape[0]
    if n < 2:
        return features.sum() * 0.0

    dists = torch.cdist(features, features, p=2)
    labels_col = labels.view(-1, 1)
    camids_col = camids.view(-1, 1)

    is_pos = labels_col.eq(labels_col.t())
    is_neg = ~is_pos
    eye = torch.eye(n, dtype=torch.bool, device=features.device)
    is_pos = is_pos & ~eye
    is_neg = is_neg & ~eye

    is_cross_cam = ~camids_col.eq(camids_col.t())
    if cross_camera_positives:
        is_pos = is_pos & is_cross_cam
    if cross_camera_negatives:
        is_neg = is_neg & is_cross_cam

    pos_dists = dists.masked_fill(~is_pos, -1.0)
    hardest_pos = pos_dists.max(dim=1).values
    neg_exists = is_neg.any(dim=1)
    neg_dists = dists.masked_fill(~is_neg, float("inf"))
    hardest_neg = neg_dists.min(dim=1).values

    valid = (hardest_pos > -0.5) & neg_exists
    if not torch.any(valid):
        return features.sum() * 0.0

    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return loss.mean()


def cross_camera_infonce_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    camids: torch.Tensor,
    temperature: float = 0.07,
    cross_camera_positives: bool = True,
    cross_camera_negatives: bool = True,
) -> torch.Tensor:
    """Compute an InfoNCE objective with camera-aware positive/negative masks."""
    n = features.shape[0]
    if n < 2:
        return features.sum() * 0.0

    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be > 0")

    labels_col = labels.view(-1, 1)
    is_pos = labels_col.eq(labels_col.t())
    is_neg = ~is_pos

    eye = torch.eye(n, dtype=torch.bool, device=features.device)
    is_pos = is_pos & ~eye
    is_neg = is_neg & ~eye

    camids_col = camids.view(-1, 1)
    is_cross_cam = ~camids_col.eq(camids_col.t())

    if cross_camera_positives:
        is_pos = is_pos & is_cross_cam
    # Optionally enforce strict cross-camera negatives.
    if cross_camera_negatives:
        is_neg = is_neg & is_cross_cam

    valid = is_pos.any(dim=1) & is_neg.any(dim=1)
    if not torch.any(valid):
        return features.sum() * 0.0

    sim = torch.matmul(features, features.t()) / float(temperature)
    den_mask = is_pos | is_neg

    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    sim_valid = sim[valid_idx]
    pos_mask_valid = is_pos[valid_idx]
    den_mask_valid = den_mask[valid_idx]

    # Stable masked log-sum-exp (masked-out entries set to very negative).
    masked_den = sim_valid.masked_fill(~den_mask_valid, -1e9)
    masked_pos = sim_valid.masked_fill(~pos_mask_valid, -1e9)

    log_den = torch.logsumexp(masked_den, dim=1)
    log_num = torch.logsumexp(masked_pos, dim=1)

    loss = -(log_num - log_den)
    return loss.mean()


@torch.no_grad()
def compute_pair_metrics(
    model: nn.Module,
    dataset: OSNetPersonDataset,
    device: torch.device,
    thresholds: List[float],
    max_pairs: int,
    batch_size: int,
    cross_camera_only: bool = True,
) -> Dict[str, object]:
    """Evaluate pairwise similarity metrics on validation features."""
    thresholds = [float(t) for t in thresholds]
    threshold_keys = [f"{t:.2f}" for t in thresholds]

    if len(dataset) == 0:
        return {
            "same_above_by_threshold": {k: 0.0 for k in threshold_keys},
            "diff_above_by_threshold": {k: 0.0 for k in threshold_keys},
            "pair_gap_by_threshold": {k: 0.0 for k in threshold_keys},
            "pair_gap_mean": 0.0,
            "same_mean": 0.0,
            "diff_mean": 0.0,
            "same_count": 0,
            "diff_count": 0,
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    all_features = []
    all_labels = []
    all_camids = []

    model.eval()
    for images, pids, camids in loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        _, feats = _extract_logits_and_feature_tensors(output, batch_size=images.shape[0])
        feats = F.normalize(feats, dim=1)
        all_features.append(feats.cpu())
        all_labels.append(pids.cpu())
        all_camids.append(camids.cpu())

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    camids = torch.cat(all_camids, dim=0)

    labels_list = [int(x) for x in labels.tolist()]
    camids_list = [int(x) for x in camids.tolist()]

    pid_to_indices: Dict[int, List[int]] = defaultdict(list)
    pid_to_cam_to_indices: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, (pid, cam) in enumerate(zip(labels_list, camids_list)):
        pid_to_indices[pid].append(idx)
        pid_to_cam_to_indices[pid][cam].append(idx)

    if cross_camera_only:
        valid_pids = [pid for pid, cam_map in pid_to_cam_to_indices.items() if len(cam_map) >= 2]
    else:
        valid_pids = [pid for pid, idxs in pid_to_indices.items() if len(idxs) >= 2]
    if len(valid_pids) < 2:
        return {
            "same_above_by_threshold": {k: 0.0 for k in threshold_keys},
            "diff_above_by_threshold": {k: 0.0 for k in threshold_keys},
            "pair_gap_by_threshold": {k: 0.0 for k in threshold_keys},
            "pair_gap_mean": 0.0,
            "same_mean": 0.0,
            "diff_mean": 0.0,
            "same_count": 0,
            "diff_count": 0,
        }

    same_sims = []
    diff_sims = []

    for _ in range(max_pairs):
        pid = random.choice(valid_pids)
        if cross_camera_only:
            cam_to_indices = pid_to_cam_to_indices[pid]
            if len(cam_to_indices) < 2:
                continue
            cam_a, cam_b = random.sample(list(cam_to_indices.keys()), 2)
            i = random.choice(cam_to_indices[cam_a])
            j = random.choice(cam_to_indices[cam_b])
        else:
            i, j = random.sample(pid_to_indices[pid], 2)
        sim = float(torch.dot(features[i], features[j]).item())
        same_sims.append(sim)

    all_pid_labels = list(pid_to_indices.keys())
    all_indices = list(range(features.shape[0]))
    if cross_camera_only:
        for _ in range(max_pairs):
            found = False
            for _ in range(64):
                ia, ib = random.sample(all_indices, 2)
                if labels_list[ia] != labels_list[ib] and camids_list[ia] != camids_list[ib]:
                    found = True
                    break
            if not found:
                continue
            sim = float(torch.dot(features[ia], features[ib]).item())
            diff_sims.append(sim)
    else:
        for _ in range(max_pairs):
            pid_a, pid_b = random.sample(all_pid_labels, 2)
            ia = random.choice(pid_to_indices[pid_a])
            ib = random.choice(pid_to_indices[pid_b])
            sim = float(torch.dot(features[ia], features[ib]).item())
            diff_sims.append(sim)

    same_arr = np.asarray(same_sims, dtype=np.float32)
    diff_arr = np.asarray(diff_sims, dtype=np.float32)

    same_above_by_threshold = {}
    diff_above_by_threshold = {}
    pair_gap_by_threshold = {}
    for t, key in zip(thresholds, threshold_keys):
        same_above = float((same_arr >= t).mean()) if same_arr.size else 0.0
        diff_above = float((diff_arr >= t).mean()) if diff_arr.size else 0.0
        same_above_by_threshold[key] = same_above
        diff_above_by_threshold[key] = diff_above
        pair_gap_by_threshold[key] = same_above - diff_above

    pair_gap_mean = float(np.mean(list(pair_gap_by_threshold.values()))) if pair_gap_by_threshold else 0.0

    return {
        "same_above_by_threshold": same_above_by_threshold,
        "diff_above_by_threshold": diff_above_by_threshold,
        "pair_gap_by_threshold": pair_gap_by_threshold,
        "pair_gap_mean": pair_gap_mean,
        "same_mean": float(same_arr.mean()) if same_arr.size else 0.0,
        "diff_mean": float(diff_arr.mean()) if diff_arr.size else 0.0,
        "same_count": int(same_arr.size),
        "diff_count": int(diff_arr.size),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    method: str,
    grad_clip_norm: float,
    infonce_temperature: float,
    triplet_margin: float,
    bot_ce_weight: float,
    bot_triplet_weight: float,
    bot_label_smoothing: float,
    cross_camera_positives: bool,
    cross_camera_negatives: bool,
) -> float:
    """Run one optimization epoch and return the average training loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, pids, camids in loader:
        images = images.to(device, non_blocking=True)
        pids = pids.to(device, non_blocking=True)
        camids = camids.to(device, non_blocking=True)

        output = model(images)
        logits, feats = _extract_logits_and_feature_tensors(output, batch_size=images.shape[0])
        feats = F.normalize(feats, dim=1)

        if method == "infonce":
            loss = cross_camera_infonce_loss(
                feats,
                pids,
                camids=camids,
                temperature=infonce_temperature,
                cross_camera_positives=cross_camera_positives,
                cross_camera_negatives=cross_camera_negatives,
            )
        elif method == "triplet":
            loss = cross_camera_triplet_loss(
                feats,
                pids,
                camids=camids,
                margin=triplet_margin,
                cross_camera_positives=cross_camera_positives,
                cross_camera_negatives=cross_camera_negatives,
            )
        elif method == "bot":
            if logits is None:
                raise RuntimeError("BoT method requires logits from the model forward pass")
            ce_loss = F.cross_entropy(logits, pids, label_smoothing=float(bot_label_smoothing))
            triplet_loss = cross_camera_triplet_loss(
                feats,
                pids,
                camids=camids,
                margin=triplet_margin,
                cross_camera_positives=cross_camera_positives,
                cross_camera_negatives=cross_camera_negatives,
            )
            loss = float(bot_ce_weight) * ce_loss + float(bot_triplet_weight) * triplet_loss
        else:
            raise ValueError(f"Unknown method: {method}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    if n_batches == 0:
        return 0.0
    return running_loss / n_batches


@torch.no_grad()
def infer_feature_dim(model: nn.Module, device: torch.device, image_height: int, image_width: int) -> int:
    """Infer embedding dimensionality from a dummy model forward pass."""
    model.eval()
    dummy = torch.randn(1, 3, int(image_height), int(image_width), device=device)
    out = model(dummy)
    _, feats = _extract_logits_and_feature_tensors(out, batch_size=1)
    return int(feats.shape[1])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for OSNet training."""
    parser = argparse.ArgumentParser(description="Train OSNet ReID encoder for ModTrack semantic mode")

    parser.add_argument("--dataset", type=str, choices=dataset_choices(), default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, required=True)

    parser.add_argument("--arch", type=str, default="osnet_ain_x1_0", help="OSNet architecture name from torchreid")
    parser.add_argument("--pretrained", action="store_true", help="Initialize OSNet from ImageNet-pretrained weights")

    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (P*K)")
    parser.add_argument("--instances_per_pid", type=int, default=4, help="K images per identity in each batch")
    parser.add_argument("--steps_per_epoch", type=int, default=300, help="Number of PK batches per epoch")
    parser.add_argument(
        "--method",
        type=str,
        choices=["infonce", "triplet", "bot"],
        default="infonce",
        help="Training objective: InfoNCE, batch-hard triplet, or BoT (cross-entropy + triplet)",
    )

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument(
        "--infonce_temperature",
        type=float,
        default=0.07,
        help="InfoNCE temperature for cosine similarity logits",
    )
    parser.add_argument("--triplet_margin", type=float, default=0.3, help="Margin for batch-hard triplet loss")
    parser.add_argument("--bot_ce_weight", type=float, default=1.0, help="BoT weight for cross-entropy term")
    parser.add_argument("--bot_triplet_weight", type=float, default=1.0, help="BoT weight for triplet term")
    parser.add_argument("--bot_label_smoothing", type=float, default=0.1, help="BoT cross-entropy label smoothing")

    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--max_images_per_pid", type=int, default=0, help="Optional cap to speed up experiments")

    parser.add_argument("--eval_threshold", type=float, default=0.67, help="Legacy single-threshold fallback")
    parser.add_argument(
        "--eval_thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Thresholds used for per-epoch pair metrics (e.g. 0.5 0.6 0.7 0.8)",
    )
    parser.add_argument("--eval_pairs", type=int, default=3000)
    parser.add_argument(
        "--eval_cross_camera_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, evaluation same/diff pair sampling is restricted to cross-camera pairs",
    )
    parser.add_argument(
        "--cross_camera_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, each PID contributes one sample per distinct camera in PK batches (100%% cross-camera positives)",
    )
    parser.add_argument(
        "--cross_camera_negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, training negatives are restricted to different-camera samples only",
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """Run end-to-end OSNet training/evaluation and checkpoint export."""
    args = parse_args()
    _set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    try:
        data_root = resolve_crops_root(args.dataset, args.dataset_root, args.data_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using crops: {data_root}")
    eval_thresholds = args.eval_thresholds if args.eval_thresholds else [args.eval_threshold]
    eval_thresholds = sorted({float(t) for t in eval_thresholds})

    train_min_cams = max(2, args.instances_per_pid) if args.cross_camera_only else 2

    train_ds = OSNetPersonDataset(
        root_dir=data_root,
        split="train",
        train_ratio=args.train_ratio,
        image_height=args.image_height,
        image_width=args.image_width,
        train_transform=True,
        min_cameras=train_min_cams,
        max_images_per_pid=args.max_images_per_pid,
    )
    eval_ds = OSNetPersonDataset(
        root_dir=data_root,
        split="val",
        train_ratio=args.train_ratio,
        image_height=args.image_height,
        image_width=args.image_width,
        train_transform=False,
        min_cameras=2,
        max_images_per_pid=0,
    )

    if train_ds.num_ids < 2:
        raise SystemExit(f"Not enough train identities: {train_ds.num_ids}")

    print(
        f"Train split: {len(train_ds)} images, {train_ds.num_ids} IDs | "
        f"Val split: {len(eval_ds)} images, {eval_ds.num_ids} IDs"
    )
    print(f"Method: {args.method}")
    print(
        f"Positive composition: {'100% cross-camera / 0% same-camera' if args.cross_camera_only else 'mixed cross/same-camera'}"
    )
    if args.method == "infonce":
        print(f"Loss setup: InfoNCE (temperature={args.infonce_temperature:.3f})")
    elif args.method == "triplet":
        print(f"Loss setup: batch-hard triplet (margin={args.triplet_margin:.3f})")
    else:
        print(
            "Loss setup: BoT "
            f"(ce_weight={args.bot_ce_weight:.3f}, triplet_weight={args.bot_triplet_weight:.3f}, "
            f"margin={args.triplet_margin:.3f}, label_smoothing={args.bot_label_smoothing:.3f})"
        )
    print(
        f"Negative composition: {'100% cross-camera' if args.cross_camera_negatives else 'mixed cross/same-camera'}"
    )
    print(
        "Eval thresholds: " + ", ".join(f"{t:.2f}" for t in eval_thresholds)
    )
    print(
        f"Eval pair mode: {'cross-camera only' if args.eval_cross_camera_only else 'mixed cross/same-camera'}"
    )

    sampler = PKSampler(
        dataset=train_ds,
        batch_size=args.batch_size,
        instances_per_pid=args.instances_per_pid,
        steps_per_epoch=args.steps_per_epoch,
        cross_camera_only=args.cross_camera_only,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    torchreid_module = _import_torchreid_or_exit()
    model = _build_osnet_model(
        torchreid_module,
        arch=args.arch,
        pretrained=args.pretrained,
        num_classes=train_ds.num_ids,
    )
    model = model.to(device)
    feature_dim = infer_feature_dim(model, device=device, image_height=args.image_height, image_width=args.image_width)
    if feature_dim <= 8:
        raise RuntimeError(
            f"Invalid feature_dim={feature_dim}. OSNet forward likely returned logits instead of embeddings. "
            "Verify torchreid model settings."
        )
    print(f"OSNet feature dimension: {feature_dim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    start_epoch = 1
    best_gap = -float("inf")

    dataset_name = args.dataset or "custom"

    for epoch in range(start_epoch, args.num_epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            method=args.method,
            grad_clip_norm=args.grad_clip_norm,
            infonce_temperature=args.infonce_temperature,
            triplet_margin=args.triplet_margin,
            bot_ce_weight=args.bot_ce_weight,
            bot_triplet_weight=args.bot_triplet_weight,
            bot_label_smoothing=args.bot_label_smoothing,
            cross_camera_positives=args.cross_camera_only,
            cross_camera_negatives=args.cross_camera_negatives,
        )
        scheduler.step()

        val_metrics = compute_pair_metrics(
            model=model,
            dataset=eval_ds,
            device=device,
            thresholds=eval_thresholds,
            max_pairs=args.eval_pairs,
            batch_size=max(1, args.batch_size // 2),
            cross_camera_only=args.eval_cross_camera_only,
        )

        threshold_parts = []
        for t in eval_thresholds:
            key = f"{t:.2f}"
            same_val = val_metrics["same_above_by_threshold"][key] * 100.0
            diff_val = val_metrics["diff_above_by_threshold"][key] * 100.0
            gap_val = val_metrics["pair_gap_by_threshold"][key]
            threshold_parts.append(f"t={t:.2f} same={same_val:.1f}% diff={diff_val:.1f}% gap={gap_val:.4f}")

        print(
            f"[epoch {epoch:03d}] loss={avg_loss:.4f} "
            f"pair_gap_mean={val_metrics['pair_gap_mean']:.4f} "
            + " | ".join(threshold_parts)
        )

        ckpt = {
            "epoch": epoch,
            "backend": "torchreid",
            "encoder_family": "osnet",
            "arch": args.arch,
            "input_size": [args.image_height, args.image_width],
            "feature_dim": feature_dim,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_pair_gap": best_gap,
            "eval_threshold": args.eval_threshold,
            "eval_thresholds": eval_thresholds,
            "val_metrics": val_metrics,
            "dataset": dataset_name,
            "config": vars(args),
        }

        last_path = Path(args.checkpoint_dir) / f"osnet_last_{dataset_name}.pt"
        torch.save(ckpt, last_path)

        curr_gap = float(val_metrics["pair_gap_mean"])
        if np.isfinite(curr_gap) and curr_gap > best_gap:
            best_gap = curr_gap
            ckpt["best_pair_gap"] = best_gap
            best_path = Path(args.checkpoint_dir) / f"best_osnet_{dataset_name}.pt"
            torch.save(ckpt, best_path)
            summary_key = f"{eval_thresholds[-1]:.2f}"
            print(
                f"  [best] saved {best_path} "
                f"(gap_mean={best_gap:.4f}, "
                f"same@{summary_key}={val_metrics['same_above_by_threshold'][summary_key]*100:.1f}%, "
                f"diff@{summary_key}={val_metrics['diff_above_by_threshold'][summary_key]*100:.1f}%)"
            )

        if epoch % 20 == 0:
            periodic_path = Path(args.checkpoint_dir) / f"osnet_{dataset_name}_epoch_{epoch:03d}.pt"
            torch.save(ckpt, periodic_path)
            print(f"  [ckpt] saved {periodic_path}")

    print("Training complete")


if __name__ == "__main__":
    main()
