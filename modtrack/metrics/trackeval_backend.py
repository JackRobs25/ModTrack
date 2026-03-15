#!/usr/bin/env python3
"""
TrackEval-backed MOT evaluation for BEV tracking.

This module intentionally keeps a drop-in compatible `MOTEvaluator` interface
for `evaluate_modtrack.py`, while delegating metric computation to the official
TrackEval library (CLEAR, Identity, HOTA).
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# TrackEval currently uses deprecated NumPy aliases (np.int/np.float/np.bool)
# in multiple modules. NumPy>=2 removed them, so restore compatibility here
# before importing trackeval.
if "int" not in np.__dict__:
    np.int = int  # type: ignore[attr-defined]
if "float" not in np.__dict__:
    np.float = float  # type: ignore[attr-defined]
if "bool" not in np.__dict__:
    np.bool = bool  # type: ignore[attr-defined]

try:
    import trackeval  # type: ignore
    from trackeval.datasets._base_dataset import _BaseDataset  # type: ignore
except Exception as exc:
    msg = (
        "[TrackEval] Failed to import 'trackeval'.\n"
        "Install it in this environment, for example:\n"
        "  pip install \"git+https://github.com/JonathonLuiten/TrackEval.git\"\n"
        f"Import error: {exc}"
    )
    raise RuntimeError(msg) from exc


def _import_trackeval_or_die():
    return trackeval


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float64)
        return float(np.mean(arr)) if arr.size > 0 else default
    if isinstance(value, np.ndarray):
        return float(np.mean(value)) if value.size > 0 else default
    try:
        return float(value)
    except Exception:
        return default


def _to_percent(value: float) -> float:
    """
    Normalize a metric to [0,100]-style display.
    TrackEval often returns percentages already, but some fields may be [0,1].
    """
    if np.isnan(value) or np.isinf(value):
        return 0.0
    if -1.0 <= value <= 1.0:
        return value * 100.0
    return value


class WildTrackBEVDataset(_BaseDataset):
    """
    Minimal custom TrackEval dataset for a single BEV sequence.

    Sequence:
        - Name: wildtrack_bev
        - Class: pedestrian
    """

    @staticmethod
    def get_default_dataset_config() -> Dict[str, Any]:
        return {
            "PRINT_CONFIG": True,
            "TRACKERS_TO_EVAL": ["bev_tracker"],
            "TRACKER_DISPLAY_NAMES": ["ModTrack"],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "OUTPUT_FOLDER": None,
            "OUTPUT_SUB_FOLDER": "",
            "FRAME_START": 0,
            "FRAME_END": 0,
            "MAX_DISTANCE": 1.0,
        }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        gt_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        pred_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> None:
        _import_trackeval_or_die()
        super().__init__()

        self.config = trackeval.utils.init_config(
            config,
            self.get_default_dataset_config(),
            self.get_name(),
        )

        self.output_fol = self.config["OUTPUT_FOLDER"] or os.getcwd()
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]
        self.should_classes_combine = False
        self.use_super_categories = False

        self.class_list = ["pedestrian"]
        self.seq_list = ["wildtrack_bev"]
        self.seq_lengths = {"wildtrack_bev": 0}

        trackers = self.config.get("TRACKERS_TO_EVAL") or ["bev_tracker"]
        self.tracker_list = list(trackers)
        disp = self.config.get("TRACKER_DISPLAY_NAMES")
        if disp is None:
            self.tracker_to_disp = {t: t for t in self.tracker_list}
        elif len(disp) == len(self.tracker_list):
            self.tracker_to_disp = {t: d for t, d in zip(self.tracker_list, disp)}
        else:
            raise ValueError("TRACKER_DISPLAY_NAMES must match TRACKERS_TO_EVAL length.")

        self.max_distance = float(self.config["MAX_DISTANCE"])
        self.frame_start = int(self.config["FRAME_START"])
        self.frame_end = int(self.config["FRAME_END"])
        self._gt_by_frame = gt_by_frame or {}
        self._pred_by_frame = pred_by_frame or {}

        if self.frame_end <= self.frame_start:
            if self._gt_by_frame or self._pred_by_frame:
                all_frames = sorted(set(self._gt_by_frame.keys()) | set(self._pred_by_frame.keys()))
                self.frame_start = all_frames[0]
                self.frame_end = all_frames[-1] + 1
            else:
                self.frame_start = 0
                self.frame_end = 1
        self.seq_lengths["wildtrack_bev"] = max(1, self.frame_end - self.frame_start)

    def get_display_name(self, tracker):
        return self.tracker_to_disp.get(tracker, tracker)

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        if gt_dets_t is None or tracker_dets_t is None:
            return np.zeros((0, 0), dtype=np.float64)
        gt = np.asarray(gt_dets_t, dtype=np.float64)
        tr = np.asarray(tracker_dets_t, dtype=np.float64)
        if gt.size == 0 or tr.size == 0:
            return np.zeros((gt.shape[0], tr.shape[0]), dtype=np.float64)

        diff = gt[:, None, :2] - tr[None, :, :2]
        dists = np.linalg.norm(diff, axis=2)
        sim = 1.0 - (dists / max(self.max_distance, 1e-12))
        sim = np.clip(sim, 0.0, 1.0)
        sim[dists > self.max_distance] = 0.0
        return sim

    def _load_raw_file(self, tracker, seq, is_gt):
        if seq != "wildtrack_bev":
            raise ValueError(f"Unknown sequence: {seq}")
        if tracker not in self.tracker_list:
            raise ValueError(f"Unknown tracker: {tracker}")

        num_timesteps = self.seq_lengths["wildtrack_bev"]

        if is_gt:
            gt_ids = []
            gt_dets = []
            gt_classes = []
            gt_extras = []
            for frame in range(self.frame_start, self.frame_end):
                rows = self._gt_by_frame.get(frame, [])
                ids = np.asarray([int(r["person_id"]) for r in rows], dtype=np.int64)
                dets = np.asarray([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)
                if dets.size == 0:
                    dets = np.empty((0, 2), dtype=np.float64)
                gt_ids.append(ids)
                gt_dets.append(dets)
                gt_classes.append(np.ones((len(ids),), dtype=np.int64))
                gt_extras.append({})
            return {
                "gt_ids": gt_ids,
                "gt_dets": gt_dets,
                "gt_classes": gt_classes,
                "gt_extras": gt_extras,
                "num_timesteps": num_timesteps,
                "seq": seq,
            }

        tracker_ids = []
        tracker_dets = []
        tracker_classes = []
        tracker_confidences = []
        for frame in range(self.frame_start, self.frame_end):
            rows = self._pred_by_frame.get(frame, [])
            valid_rows = [r for r in rows if int(r["track_id"]) >= 0]
            ids = np.asarray([int(r["track_id"]) for r in valid_rows], dtype=np.int64)
            dets = np.asarray([[float(r["x"]), float(r["y"])] for r in valid_rows], dtype=np.float64)
            conf = np.asarray([float(r.get("confidence", 1.0)) for r in valid_rows], dtype=np.float64)
            if dets.size == 0:
                dets = np.empty((0, 2), dtype=np.float64)
            tracker_ids.append(ids)
            tracker_dets.append(dets)
            tracker_classes.append(np.ones((len(ids),), dtype=np.int64))
            tracker_confidences.append(conf)

        return {
            "tracker_ids": tracker_ids,
            "tracker_dets": tracker_dets,
            "tracker_classes": tracker_classes,
            "tracker_confidences": tracker_confidences,
            "num_timesteps": num_timesteps,
            "seq": seq,
        }

    def get_preprocessed_seq_data(self, raw_data, cls):
        if cls.lower() != "pedestrian":
            raise ValueError(f"Unsupported class '{cls}'. Only 'pedestrian' is supported.")

        self._check_unique_ids(raw_data)
        keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
        ]
        data = {k: [None] * raw_data["num_timesteps"] for k in keys}
        unique_gt_ids: List[int] = []
        unique_tracker_ids: List[int] = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data["num_timesteps"]):
            data["gt_ids"][t] = raw_data["gt_ids"][t]
            data["gt_dets"][t] = raw_data["gt_dets"][t]
            data["similarity_scores"][t] = raw_data["similarity_scores"][t]
            data["tracker_ids"][t] = raw_data["tracker_ids"][t]
            data["tracker_dets"][t] = raw_data["tracker_dets"][t]
            data["tracker_confidences"][t] = raw_data["tracker_confidences"][t]

            unique_gt_ids.extend(list(np.unique(data["gt_ids"][t])))
            unique_tracker_ids.extend(list(np.unique(data["tracker_ids"][t])))
            num_gt_dets += len(data["gt_ids"][t])
            num_tracker_dets += len(data["tracker_ids"][t])

        if len(unique_gt_ids) > 0:
            unique_gt_ids = list(np.unique(np.asarray(unique_gt_ids, dtype=np.int64)))
            gt_id_map = {old: new for new, old in enumerate(unique_gt_ids)}
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = np.asarray([gt_id_map[int(v)] for v in data["gt_ids"][t]], dtype=np.int64)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = list(np.unique(np.asarray(unique_tracker_ids, dtype=np.int64)))
            tr_id_map = {old: new for new, old in enumerate(unique_tracker_ids)}
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = np.asarray(
                        [tr_id_map[int(v)] for v in data["tracker_ids"][t]], dtype=np.int64
                    )

        data["num_tracker_dets"] = int(num_tracker_dets)
        data["num_gt_dets"] = int(num_gt_dets)
        data["num_tracker_ids"] = int(len(unique_tracker_ids))
        data["num_gt_ids"] = int(len(unique_gt_ids))
        data["num_timesteps"] = int(raw_data["num_timesteps"])
        data["seq"] = raw_data["seq"]
        self._check_unique_ids(data, after_preproc=True)
        return data


class MOTEvaluator:
    """TrackEval-backed MOT evaluator with a legacy-compatible interface."""

    def __init__(self, max_distance: float = 1.0, min_iou: float = 0.5, frame_step: int = 1):
        _import_trackeval_or_die()
        self.max_distance = float(max_distance)
        self.min_iou = float(min_iou)
        self.frame_step = int(frame_step)
        self.predictions: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.ground_truth: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._frame_counter = 0
        self.metrics: Dict[str, Any] = {}

    def update(
        self,
        predicted_tracks: Dict[int, np.ndarray],
        ground_truth: Dict[str, np.ndarray],
        frame_idx: Optional[int] = None,
    ) -> None:
        frame = frame_idx if frame_idx is not None else self._frame_counter
        if frame_idx is None:
            self._frame_counter += 1
        else:
            self._frame_counter = max(self._frame_counter, frame_idx + 1)

        for track_id, pos in predicted_tracks.items():
            if pos is None or len(pos) < 2:
                continue
            self.predictions[frame].append(
                {
                    "track_id": int(track_id) if track_id is not None else -1,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "confidence": 1.0,
                }
            )

        world_pts = ground_truth.get("world_pts")
        person_ids = ground_truth.get("person_ids")
        if world_pts is None or person_ids is None or len(world_pts) != len(person_ids):
            return

        for pt, pid in zip(world_pts, person_ids):
            if pt is None or len(pt) < 2:
                continue
            self.ground_truth[frame].append(
                {
                    "person_id": int(pid),
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                }
            )

    def load_predictions(self, pred_file: str) -> None:
        with open(pred_file, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].strip().lower() == "frame":
                    continue
                if len(row) < 4:
                    continue
                frame = int(row[0])
                self.predictions[frame].append(
                    {
                        "track_id": int(row[1]),
                        "x": float(row[2]),
                        "y": float(row[3]),
                        "confidence": float(row[4]) if len(row) > 4 and row[4] != "" else 1.0,
                    }
                )

    def load_ground_truth_from_wildtrack(self, wildtrack_root: str, frame_indices: List[int]) -> None:
        ann_dir = Path(wildtrack_root) / "annotations_positions"
        for frame_idx in frame_indices:
            path = ann_dir / f"frame_{frame_idx:06d}.json"
            if not path.exists():
                print(f"[Warning] Ground truth not found: {path}")
                continue
            with open(path, "r") as f:
                records = json.load(f)
            for record in records:
                pid = record.get("personID")
                if pid is None:
                    continue
                self.ground_truth[frame_idx].append(
                    {
                        "person_id": int(pid),
                        "x": float(record.get("worldX", 0.0)),
                        "y": float(record.get("worldY", 0.0)),
                    }
                )

    def load_ground_truth_from_file(self, gt_file: str) -> None:
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                frame = int(parts[0])
                self.ground_truth[frame].append(
                    {
                        "person_id": int(parts[1]),
                        "x": float(parts[2]),
                        "y": float(parts[3]),
                    }
                )

    @staticmethod
    def _find_metric_bundle(node: Any) -> Optional[Dict[str, Any]]:
        if isinstance(node, dict):
            keys = set(node.keys())
            if {"CLEAR", "Identity", "HOTA"}.issubset(keys):
                return node
            for value in node.values():
                found = MOTEvaluator._find_metric_bundle(value)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _count_tracks(by_frame: Dict[int, List[Dict[str, Any]]], key: str) -> int:
        ids = set()
        for rows in by_frame.values():
            for row in rows:
                ids.add(int(row[key]))
        return len(ids)

    def _run_trackeval(self) -> Dict[str, float]:
        trackeval = _import_trackeval_or_die()

        all_frames = sorted(set(self.predictions.keys()) | set(self.ground_truth.keys()))
        if all_frames:
            frame_start = int(all_frames[0])
            frame_end = int(all_frames[-1]) + 1
        else:
            frame_start = 0
            frame_end = 1

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config["PRINT_ONLY_COMBINED"] = False
        eval_config["PRINT_CONFIG"] = True
        eval_config["DISPLAY_LESS_PROGRESS"] = False
        eval_config["PRINT_RESULTS"] = True
        eval_config["OUTPUT_SUMMARY"] = False
        eval_config["OUTPUT_DETAILED"] = False
        eval_config["PLOT_CURVES"] = False

        dataset_config = WildTrackBEVDataset.get_default_dataset_config()
        dataset_config.update(
            {
                "PRINT_CONFIG": True,
                "TRACKERS_TO_EVAL": ["bev_tracker"],
                "TRACKER_DISPLAY_NAMES": ["ModTrack"],
                "CLASSES_TO_EVAL": ["pedestrian"],
                "FRAME_START": frame_start,
                "FRAME_END": frame_end,
                "MAX_DISTANCE": float(self.max_distance),
            }
        )
        # With sim=clip(1-dist/max_distance,0,1), near-zero threshold preserves the
        # intended distance gate at max_distance instead of adding a stricter cutoff.
        metrics_config = {"METRICS": ["CLEAR", "Identity", "HOTA"], "THRESHOLD": 1e-10}

        print("[TrackEval] Official TrackEval backend enabled")
        print("[TrackEval] Evaluation Config:")
        print(f"  Eval config: {eval_config}")
        print(f"  Dataset config: {dataset_config}")
        print(f"  Metrics config: {metrics_config}")
        print(f"  Dataset: wildtrack_bev")
        print(f"  Sequence: wildtrack_bev")
        print(f"  Frame range: [{frame_start}, {frame_end})")
        print(f"  Max distance: {self.max_distance:.3f} m")
        print("  Similarity: clip(1 - dist/max_distance, 0, 1), dist>max_distance => 0")
        print("  Metrics: CLEAR, Identity, HOTA")

        dataset = WildTrackBEVDataset(
            dataset_config,
            gt_by_frame=self.ground_truth,
            pred_by_frame=self.predictions,
        )

        evaluator = trackeval.Evaluator(eval_config)
        metrics_list = [
            trackeval.metrics.CLEAR(metrics_config),
            trackeval.metrics.Identity(metrics_config),
            trackeval.metrics.HOTA(metrics_config),
        ]
        results, _messages = evaluator.evaluate([dataset], metrics_list)

        metric_bundle = self._find_metric_bundle(results)
        if metric_bundle is None:
            raise RuntimeError("[TrackEval] Could not locate CLEAR/Identity/HOTA results in evaluator output.")

        clear = metric_bundle.get("CLEAR", {})
        identity = metric_bundle.get("Identity", {})
        hota = metric_bundle.get("HOTA", {})

        mota = _to_percent(_as_float(clear.get("MOTA"), 0.0))
        idf1 = _to_percent(_as_float(identity.get("IDF1"), 0.0))
        hota_v = _to_percent(_as_float(hota.get("HOTA"), 0.0))
        deta = _to_percent(_as_float(hota.get("DetA"), 0.0))
        assa = _to_percent(_as_float(hota.get("AssA"), 0.0))
        loca = _to_percent(_as_float(hota.get("LocA"), 0.0))
        # TrackEval CLEAR exposes both count fields (MT, ML, PT) and ratio fields
        # (MTR, MLR, PTR). Report MT/ML as percentages for consistency with the
        # rest of this repo and py-motmetrics output.
        mt = _to_percent(_as_float(clear.get("MTR"), 0.0))
        ml = _to_percent(_as_float(clear.get("MLR"), 0.0))
        mt_count = int(round(_as_float(clear.get("MT"), 0.0)))
        ml_count = int(round(_as_float(clear.get("ML"), 0.0)))
        pt_count = int(round(_as_float(clear.get("PT"), 0.0)))

        motp_raw = _as_float(clear.get("MOTP"), 0.0)
        motp_similarity = motp_raw / 100.0 if motp_raw > 1.0 else motp_raw
        motp_similarity = float(np.clip(motp_similarity, 0.0, 1.0))
        motp_m = float(np.clip((1.0 - motp_similarity) * self.max_distance, 0.0, self.max_distance))

        tp = int(round(_as_float(clear.get("CLR_TP"), 0.0)))
        fp = int(round(_as_float(clear.get("CLR_FP"), 0.0)))
        fn = int(round(_as_float(clear.get("CLR_FN"), 0.0)))
        ids = int(round(_as_float(clear.get("IDSW"), 0.0)))
        frag = int(round(_as_float(clear.get("Frag"), 0.0)))

        idtp = int(round(_as_float(identity.get("IDTP"), 0.0)))
        idfp = int(round(_as_float(identity.get("IDFP"), 0.0)))
        idfn = int(round(_as_float(identity.get("IDFN"), 0.0)))

        pred_count = tp + fp
        gt_count = tp + fn

        results_out = {
            "MOTA": float(mota),
            "MOTP": float(motp_m),
            "MOTP_pct": float(motp_similarity * 100.0),
            "MOTP_trackeval_raw": float(motp_raw),
            "IDF1": float(idf1),
            "HOTA": float(hota_v),
            "DetA": float(deta),
            "AssA": float(assa),
            "LocA": float(loca),
            "Precision": float(tp / pred_count) if pred_count > 0 else 0.0,
            "Recall": float(tp / gt_count) if gt_count > 0 else 0.0,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "IDS": int(ids),
            "IDSW": int(ids),
            "Frag": int(frag),
            "IDTP": int(idtp),
            "IDFP": int(idfp),
            "IDFN": int(idfn),
            "MT": float(mt),
            "ML": float(ml),
            "MT_count": int(mt_count),
            "ML_count": int(ml_count),
            "PT_count": int(pt_count),
            "num_GT_tracks": int(self._count_tracks(self.ground_truth, "person_id")),
            "num_pred_tracks": int(self._count_tracks(self.predictions, "track_id")),
            "num_frames": int(max(0, frame_end - frame_start)),
            "sequence_name": "wildtrack_bev",
            "dataset_name": "wildtrack_bev",
            "max_distance": float(self.max_distance),
            "backend": "TrackEval",
        }
        self.metrics = results_out
        return results_out

    def evaluate(self) -> Dict[str, float]:
        return self._run_trackeval()

    def compute_metrics(self) -> Dict[str, float]:
        return self.evaluate()

    def print_results(self, results: Dict[str, float]) -> None:
        print("\n" + "=" * 70)
        print("TRACK EVAL (OFFICIAL) RESULTS")
        print("=" * 70)
        print(f"  Backend:       {results.get('backend', 'TrackEval')}")
        print(f"  Dataset:       {results.get('dataset_name', 'wildtrack_bev')}")
        print(f"  Sequence:      {results.get('sequence_name', 'wildtrack_bev')}")
        print(f"  Max distance:  {results.get('max_distance', self.max_distance):.3f} m")

        print("\nPrimary Metrics:")
        print(f"  MOTA: {results.get('MOTA', 0.0):6.2f}%")
        print(
            f"  MOTP: {results.get('MOTP', 0.0):6.3f} m "
            f"(sim={results.get('MOTP_pct', 0.0):.1f}%, raw={results.get('MOTP_trackeval_raw', 0.0):.4f})"
        )
        print(f"  IDF1: {results.get('IDF1', 0.0):6.2f}%")
        print(f"  HOTA: {results.get('HOTA', 0.0):6.2f}%")
        print(f"  DetA: {results.get('DetA', 0.0):6.2f}%")
        print(f"  AssA: {results.get('AssA', 0.0):6.2f}%")
        print(f"  LocA: {results.get('LocA', 0.0):6.2f}%")

        print("\nIdentity/Error Counters:")
        print(f"  IDSW: {int(results.get('IDSW', 0))}")
        print(f"  Frag: {int(results.get('Frag', 0))}")
        print(f"  IDTP: {int(results.get('IDTP', 0))}")
        print(f"  IDFP: {int(results.get('IDFP', 0))}")
        print(f"  IDFN: {int(results.get('IDFN', 0))}")
        print(f"  TP/FP/FN: {int(results.get('TP', 0))}/{int(results.get('FP', 0))}/{int(results.get('FN', 0))}")
        print("=" * 70)

    def save_results(self, results: Dict[str, float], output_file: str) -> None:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[TrackEval] Saved results to: {output_file}")

