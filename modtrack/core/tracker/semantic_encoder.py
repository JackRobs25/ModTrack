"""OSNet semantic feature extraction for ModTrack."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SemanticEncoder:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        feature_dim: int = 128,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = int(feature_dim)
        self.model = None
        self.checkpoint_path = checkpoint_path

        self.encoder_family = "osnet"
        self.arch = "osnet_ain_x1_0"
        self.input_size = (256, 128)  # (height, width)
        self._norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._checkpoint_obj = None
        if checkpoint_path and Path(checkpoint_path).is_file():
            self._checkpoint_obj = self._load_checkpoint_object(checkpoint_path)
            self._infer_checkpoint_metadata(self._checkpoint_obj)

        self._build_osnet_model()
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint_object(self, checkpoint_path: str):
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            return None

    def _infer_checkpoint_metadata(self, checkpoint_obj) -> None:
        if not isinstance(checkpoint_obj, dict):
            return

        family = str(checkpoint_obj.get("encoder_family", "")).lower().strip()
        if family and family != "osnet":
            raise ValueError(
                f"SemanticEncoder only supports OSNet checkpoints, got encoder_family='{family}'"
            )

        backend = str(checkpoint_obj.get("backend", "")).lower().strip()
        if backend and backend != "torchreid":
            raise ValueError(
                f"SemanticEncoder only supports torchreid OSNet checkpoints, got backend='{backend}'"
            )

        arch = checkpoint_obj.get("arch")
        if isinstance(arch, str) and arch.strip():
            self.arch = arch.strip()

        size = checkpoint_obj.get("input_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            self.input_size = (int(size[0]), int(size[1]))

        feature_dim = checkpoint_obj.get("feature_dim")
        if feature_dim is not None:
            self.feature_dim = int(feature_dim)

    def _build_osnet_model(self):
        try:
            import torchreid
        except ImportError as exc:
            print(f"[WARNING] Could not import torchreid for OSNet semantic encoder: {exc}")
            self.model = None
            return

        arch = self.arch or "osnet_ain_x1_0"
        build_fn = getattr(torchreid.models, "build_model", None)
        if build_fn is None:
            print("[WARNING] torchreid.models.build_model not available")
            self.model = None
            return

        trials = [
            {"name": arch, "num_classes": 1, "loss": "triplet", "pretrained": False, "use_gpu": torch.cuda.is_available()},
            {"name": arch, "num_classes": 1, "loss": "triplet", "pretrained": False},
            {"name": arch, "num_classes": 1, "loss": "triplet"},
            {"name": arch, "num_classes": 1, "pretrained": False},
            {"name": arch, "num_classes": 1},
        ]

        model = None
        last_error = None
        for kwargs in trials:
            try:
                model = build_fn(**kwargs)
                break
            except Exception as exc:  # pragma: no cover - compatibility fallback
                last_error = exc

        if model is None:
            print(f"[WARNING] Could not build OSNet model '{arch}': {last_error}")
            self.model = None
            return

        self.model = model.to(self.device)
        self.model.eval()
        print(f"[SemanticEncoder] Loaded OSNet backbone arch={arch}")

    def _state_dict_from_checkpoint(self, checkpoint_obj) -> Optional[Dict[str, torch.Tensor]]:
        if checkpoint_obj is None:
            return None

        if isinstance(checkpoint_obj, dict):
            if "model_state_dict" in checkpoint_obj:
                state_dict = checkpoint_obj["model_state_dict"]
            elif "state_dict" in checkpoint_obj:
                state_dict = checkpoint_obj["state_dict"]
            else:
                state_dict = checkpoint_obj
        else:
            state_dict = checkpoint_obj

        if not isinstance(state_dict, dict):
            return None

        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith("module.") else key
            cleaned[new_key] = value
        return cleaned

    def _filter_incompatible_keys(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], list[str]]:
        if self.model is None:
            return state_dict, []

        model_state = self.model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped: list[str] = []

        for key, value in state_dict.items():
            model_value = model_state.get(key)
            if model_value is None:
                filtered[key] = value
                continue

            if model_value.shape != value.shape:
                skipped.append(
                    f"{key}: checkpoint {tuple(value.shape)} != model {tuple(model_value.shape)}"
                )
                continue

            filtered[key] = value

        return filtered, skipped

    def _load_checkpoint(self, checkpoint_path: str):
        try:
            if self.model is None:
                print("[WARNING] Semantic model not initialized; skipping checkpoint load")
                return

            checkpoint = self._checkpoint_obj
            if checkpoint is None:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            state_dict = self._state_dict_from_checkpoint(checkpoint)
            if state_dict is None:
                raise RuntimeError("Checkpoint did not contain a valid state_dict")

            state_dict, skipped = self._filter_incompatible_keys(state_dict)
            if skipped:
                print(f"[SemanticEncoder] skipped incompatible keys: {len(skipped)}")
                for item in skipped[:5]:
                    print(f"[SemanticEncoder]   - {item}")
                if len(skipped) > 5:
                    print(f"[SemanticEncoder]   ... and {len(skipped) - 5} more")

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[SemanticEncoder] missing keys: {len(missing)}")
            if unexpected:
                print(f"[SemanticEncoder] unexpected keys: {len(unexpected)}")

            print(f"[SemanticEncoder] Loaded checkpoint from {checkpoint_path} (family=osnet)")
        except Exception as exc:
            print(f"[WARNING] Could not load semantic checkpoint {checkpoint_path}: {exc}")

    def _preprocess_image_np(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = self.input_size
        image_resized = cv2.resize(image_rgb, (w, h))

        image_float = image_resized.astype(np.float32) / 255.0
        image_norm = (image_float - self._norm_mean) / self._norm_std

        chw = np.transpose(image_norm, (2, 0, 1))
        return np.ascontiguousarray(chw, dtype=np.float32)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        image_chw = self._preprocess_image_np(image)
        tensor = torch.from_numpy(image_chw).unsqueeze(0)
        return tensor.to(self.device)

    def _preprocess_images_batch_np(self, images: List[np.ndarray]) -> np.ndarray:
        """Preprocess a crop list into a contiguous [N, 3, H, W] float32 batch."""
        n_images = len(images)
        if n_images == 0:
            h, w = self.input_size
            return np.empty((0, 3, h, w), dtype=np.float32)

        h, w = self.input_size
        batch_np = np.empty((n_images, 3, h, w), dtype=np.float32)
        for idx, image in enumerate(images):
            batch_np[idx] = self._preprocess_image_np(image)
        return batch_np

    def _extract_feature_tensor(self, model_out: object, batch_size: int) -> torch.Tensor:
        if torch.is_tensor(model_out):
            return model_out

        if isinstance(model_out, (tuple, list)):
            tensors = [x for x in model_out if torch.is_tensor(x)]
            candidates = [t for t in tensors if t.ndim == 2 and t.shape[0] == batch_size]
            if candidates:
                candidates = sorted(candidates, key=lambda t: t.shape[1], reverse=True)
                return candidates[0]
            if tensors:
                return tensors[0]

        raise RuntimeError(f"Unsupported semantic model output type: {type(model_out)}")

    @torch.no_grad()
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None

        try:
            batch = self._preprocess_image(image)
            model_out = self.model(batch)
            embeddings = self._extract_feature_tensor(model_out, batch_size=batch.shape[0])
            embeddings = F.normalize(embeddings, dim=1)

            feat_np = embeddings.cpu().numpy().astype(np.float32)
            if feat_np.ndim != 2 or feat_np.shape[0] == 0:
                return None

            self.feature_dim = int(feat_np.shape[1])
            return feat_np[0]
        except Exception as exc:
            print(f"[WARNING] Semantic feature extraction failed: {exc}")
            return None

    @torch.no_grad()
    def extract_features_batch(self, images: list[np.ndarray]) -> Optional[np.ndarray]:
        if self.model is None or len(images) == 0:
            return None

        try:
            batch_np = self._preprocess_images_batch_np(images)
            batch_cpu = torch.from_numpy(batch_np)

            if str(self.device).startswith("cuda"):
                batch = batch_cpu.pin_memory().to(self.device, non_blocking=True)
            else:
                batch = batch_cpu.to(self.device)

            model_out = self.model(batch)
            embeddings = self._extract_feature_tensor(model_out, batch_size=batch.shape[0])
            embeddings = F.normalize(embeddings, dim=1)

            feat_np = embeddings.cpu().numpy().astype(np.float32)
            if feat_np.ndim != 2:
                return None

            self.feature_dim = int(feat_np.shape[1])
            return feat_np
        except Exception as exc:
            print(f"[WARNING] Batch semantic feature extraction failed: {exc}")
            return None

    def extract_from_detection(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x_min, y_min, x_max, y_max = bbox

        h, w = image.shape[:2]
        x_min = max(0, min(x_min, w - 1))
        x_max = max(0, min(x_max, w))
        y_min = max(0, min(y_min, h - 1))
        y_max = max(0, min(y_max, h))

        if x_max <= x_min or y_max <= y_min:
            return None

        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None

        return self.extract_features(crop)


_global_encoder: Optional[SemanticEncoder] = None


def get_semantic_encoder(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Optional[SemanticEncoder]:
    """Return a cached semantic encoder instance, reinitializing when config changes."""
    global _global_encoder

    needs_reinit = (
        _global_encoder is None
        or (checkpoint_path is not None and _global_encoder.checkpoint_path != checkpoint_path)
        or (device is not None and _global_encoder.device != device)
    )

    if needs_reinit:
        try:
            _global_encoder = SemanticEncoder(checkpoint_path=checkpoint_path, device=device)
        except Exception as exc:
            print(f"[WARNING] Could not initialize global semantic encoder: {exc}")
            return None

    return _global_encoder
