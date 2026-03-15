#!/usr/bin/env python3
"""
py-motmetrics-backed MOT evaluation for BEV tracking.

This module keeps a drop-in compatible `MOTEvaluator` interface for
`evaluate_modtrack.py`, while delegating CLEAR/ID metrics to py-motmetrics.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import motmetrics as mm  # type: ignore
except Exception as exc:
    msg = (
        "[py-motmetrics] Failed to import 'motmetrics'.\n"
        "Install it in this environment, for example:\n"
        "  pip install motmetrics pandas scipy\n"
        f"Import error: {exc}"
    )
    raise RuntimeError(msg) from exc


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert scalar-like values to finite float values with fallback default."""
    if value is None:
        return default
    try:
        v = float(value)
    except Exception:
        return default
    if np.isnan(v) or np.isinf(v):
        return default
    return v


def _to_percent(value: float) -> float:
    """Normalize scores to percentage-style units when values are in [0, 1]."""
    if np.isnan(value) or np.isinf(value):
        return 0.0
    if -1.0 <= value <= 1.0:
        return value * 100.0
    return value


class MOTEvaluator:
    """py-motmetrics-backed MOT evaluator with a legacy-compatible interface."""

    def __init__(self, max_distance: float = 1.0, min_iou: float = 0.5, frame_step: int = 1):
        self.max_distance = float(max_distance)
        self.min_iou = float(min_iou)
        self.frame_step = int(frame_step)
        self.predictions: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.ground_truth: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._frame_counter = 0
        self.metrics: Dict[str, Any] = {}

        # Prefer SciPy LAP solver to avoid optional third-party solver deps.
        try:
            mm.lap.default_solver = "scipy"
        except (AttributeError, ImportError):
            # Older/newer motmetrics builds may not expose solver configuration.
            # Keep default behavior in that case.
            pass

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
    def _count_tracks(by_frame: Dict[int, List[Dict[str, Any]]], key: str) -> int:
        ids = set()
        for rows in by_frame.values():
            for row in rows:
                ids.add(int(row[key]))
        return len(ids)

    def _distance_matrix(self, preds: List[Dict[str, Any]], gts: List[Dict[str, Any]]) -> np.ndarray:
        if not gts or not preds:
            return np.zeros((len(gts), len(preds)), dtype=np.float64)
        out = np.zeros((len(gts), len(preds)), dtype=np.float64)
        for gi, gt in enumerate(gts):
            gx = float(gt["x"])
            gy = float(gt["y"])
            for pi, pred in enumerate(preds):
                dx = gx - float(pred["x"])
                dy = gy - float(pred["y"])
                out[gi, pi] = float(np.hypot(dx, dy))
        return out

    def _run_motmetrics(self) -> Dict[str, Any]:
        all_frames = sorted(set(self.predictions.keys()) | set(self.ground_truth.keys()))
        acc = mm.MOTAccumulator(auto_id=False)

        for frame in all_frames:
            gt_rows = self.ground_truth.get(frame, [])
            pred_rows = self.predictions.get(frame, [])
            gt_ids = [int(r["person_id"]) for r in gt_rows]
            pred_ids = [int(r["track_id"]) for r in pred_rows]

            dists = self._distance_matrix(pred_rows, gt_rows)
            if dists.size > 0:
                dists = dists.astype(np.float64, copy=False)
                dists[dists > self.max_distance] = np.nan
            # With auto_id disabled, py-motmetrics requires an explicit frame id.
            acc.update(gt_ids, pred_ids, dists, frameid=int(frame))

        mh = mm.metrics.create()
        available = set(mh.metrics.keys())
        requested = [
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "precision",
            "recall",
            "num_detections",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
            "mostly_tracked",
            "mostly_lost",
            "idtp",
            "idfp",
            "idfn",
        ]
        metrics = [k for k in requested if k in available]
        summary = mh.compute(acc, metrics=metrics, name="bev_tracker")
        row = summary.loc["bev_tracker"]

        def _row_val(name: str, default: Optional[float] = 0.0) -> Optional[float]:
            if name not in row.index:
                return default
            raw = row[name]
            if raw is None:
                return default
            try:
                value = float(raw)
            except Exception:
                return default
            if np.isnan(value) or np.isinf(value):
                return default
            return value

        tp = int(round(_as_float(_row_val("num_detections", 0.0), 0.0)))
        fp = int(round(_as_float(_row_val("num_false_positives", 0.0), 0.0)))
        fn = int(round(_as_float(_row_val("num_misses", 0.0), 0.0)))
        idsw = int(round(_as_float(_row_val("num_switches", 0.0), 0.0)))
        frag = int(round(_as_float(_row_val("num_fragmentations", 0.0), 0.0)))

        num_gt_tracks = int(self._count_tracks(self.ground_truth, "person_id"))
        num_pred_tracks = int(self._count_tracks(self.predictions, "track_id"))
        num_frames = int(round(_as_float(_row_val("num_frames", float(len(all_frames))), float(len(all_frames)))))

        mota = _to_percent(_as_float(_row_val("mota", 0.0), 0.0))
        idf1 = _to_percent(_as_float(_row_val("idf1", 0.0), 0.0))
        precision = _as_float(_row_val("precision", 0.0), 0.0)
        recall = _as_float(_row_val("recall", 0.0), 0.0)

        motp_m = _as_float(_row_val("motp", 0.0), 0.0) if tp > 0 else 0.0
        motp_pct = (1.0 - (motp_m / max(self.max_distance, 1e-12))) * 100.0 if tp > 0 else 0.0
        motp_pct = float(np.clip(motp_pct, 0.0, 100.0))

        mt_count = int(round(_as_float(_row_val("mostly_tracked", 0.0), 0.0)))
        ml_count = int(round(_as_float(_row_val("mostly_lost", 0.0), 0.0)))
        mt_pct = (mt_count / num_gt_tracks * 100.0) if num_gt_tracks > 0 else 0.0
        ml_pct = (ml_count / num_gt_tracks * 100.0) if num_gt_tracks > 0 else 0.0

        idtp_v = _row_val("idtp", None)
        idfp_v = _row_val("idfp", None)
        idfn_v = _row_val("idfn", None)

        results = {
            "MOTA": float(mota),
            "MOTP": float(motp_m),
            "MOTP_pct": float(motp_pct),
            "IDF1": float(idf1),
            "HOTA": None,
            "DetA": None,
            "AssA": None,
            "LocA": None,
            "Precision": float(precision),
            "Recall": float(recall),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "IDS": int(idsw),
            "IDSW": int(idsw),
            "Frag": int(frag),
            "IDTP": int(round(_as_float(idtp_v, 0.0))) if idtp_v is not None else None,
            "IDFP": int(round(_as_float(idfp_v, 0.0))) if idfp_v is not None else None,
            "IDFN": int(round(_as_float(idfn_v, 0.0))) if idfn_v is not None else None,
            "MT": float(mt_pct),
            "ML": float(ml_pct),
            "MT_count": int(mt_count),
            "ML_count": int(ml_count),
            "num_GT_tracks": int(num_gt_tracks),
            "num_pred_tracks": int(num_pred_tracks),
            "num_frames": int(num_frames),
            "sequence_name": "wildtrack_bev",
            "dataset_name": "wildtrack_bev",
            "max_distance": float(self.max_distance),
            "backend": "py-motmetrics",
        }
        self.metrics = results
        return results

    def evaluate(self) -> Dict[str, Any]:
        return self._run_motmetrics()

    def compute_metrics(self) -> Dict[str, Any]:
        return self.evaluate()

    @staticmethod
    def _fmt_float(value: Any, fmt: str) -> str:
        if value is None:
            return "N/A"
        try:
            v = float(value)
        except Exception:
            return "N/A"
        if np.isnan(v) or np.isinf(v):
            return "N/A"
        return format(v, fmt)

    def print_results(self, results: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print("PY-MOTMETRICS RESULTS")
        print("=" * 70)
        print(f"  Backend:       {results.get('backend', 'py-motmetrics')}")
        print(f"  Dataset:       {results.get('dataset_name', 'wildtrack_bev')}")
        print(f"  Sequence:      {results.get('sequence_name', 'wildtrack_bev')}")
        print(f"  Max distance:  {self._fmt_float(results.get('max_distance'), '.3f')} m")

        print("\nPrimary Metrics:")
        print(f"  MOTA: {self._fmt_float(results.get('MOTA'), '6.2f')}%")
        print(
            f"  MOTP: {self._fmt_float(results.get('MOTP'), '6.3f')} m "
            f"(pct={self._fmt_float(results.get('MOTP_pct'), '.1f')}%)"
        )
        print(f"  IDF1: {self._fmt_float(results.get('IDF1'), '6.2f')}%")

        print("\nIdentity/Error Counters:")
        print(f"  IDSW: {int(results.get('IDSW', 0))}")
        print(f"  Frag: {int(results.get('Frag', 0))}")
        print(
            f"  TP/FP/FN: {int(results.get('TP', 0))}/"
            f"{int(results.get('FP', 0))}/{int(results.get('FN', 0))}"
        )
        print("=" * 70)

    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[py-motmetrics] Saved results to: {output_file}")
