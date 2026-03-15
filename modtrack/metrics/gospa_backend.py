#!/usr/bin/env python3
"""
GOSPA evaluation backend for BEV tracking.

This module keeps a drop-in compatible `MOTEvaluator` interface for
`evaluate_modtrack.py`.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def _try_import_gospapy():
    def _load_gospapy_module():
        try:
            import importlib
            module = importlib.import_module("gospapy")
        except Exception:
            return None

        if hasattr(module, "calculate_gospa"):
            return module

        try:
            import importlib
            gospa_submodule = importlib.import_module("gospapy.gospa")
            calculate_gospa = getattr(gospa_submodule, "calculate_gospa", None)
            if calculate_gospa is not None:
                setattr(module, "calculate_gospa", calculate_gospa)
                return module
        except Exception:
            return None
        return None

    module = _load_gospapy_module()
    if module is not None:
        return module
    return None


def _calculate_gospa_local(
    targets: List[np.ndarray],
    tracks: List[np.ndarray],
    c: float,
    p: float,
    alpha: float,
):
    """Local GOSPA implementation with assignment decomposition."""
    m = len(targets)
    n = len(tracks)
    c = float(c)
    p = float(p)
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if c <= 0.0:
        raise ValueError("c must be > 0")
    if p <= 0.0:
        raise ValueError("p must be > 0")

    cutoff_cost = (c ** p) / alpha
    if m == 0 and n == 0:
        return 0.0, [], 0.0, 0.0, 0.0
    if m == 0:
        false = cutoff_cost * n
        return float(false ** (1.0 / p)), [], 0.0, 0.0, float(false)
    if n == 0:
        missed = cutoff_cost * m
        return float(missed ** (1.0 / p)), [], 0.0, float(missed), 0.0

    tgt = np.asarray(targets, dtype=np.float64)
    trk = np.asarray(tracks, dtype=np.float64)
    dmat = np.linalg.norm(tgt[:, None, :] - trk[None, :, :], axis=2)
    dmat = np.minimum(dmat, c)
    real_cost = dmat ** p

    # Augmented LAP to allow unassigned targets/tracks.
    dim = m + n
    big = 1e12
    cost = np.full((dim, dim), big, dtype=np.float64)
    cost[:m, :n] = real_cost
    for i in range(m):
        cost[i, n + i] = cutoff_cost
    for j in range(n):
        cost[m + j, j] = cutoff_cost
    cost[m:, n:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = []
    loc = 0.0
    missed = 0.0
    false = 0.0

    for r, cidx in zip(row_ind.tolist(), col_ind.tolist()):
        if r < m and cidx < n:
            assignment.append((int(r), int(cidx)))
            loc += float(real_cost[r, cidx])
        elif r < m and cidx >= n and (cidx - n) == r:
            missed += float(cutoff_cost)
        elif r >= m and cidx < n and (r - m) == cidx:
            false += float(cutoff_cost)

    total = loc + missed + false
    gospa = float(total ** (1.0 / p))
    return gospa, assignment, float(loc), float(missed), float(false)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        v = float(value)
    except Exception:
        return default
    if np.isnan(v) or np.isinf(v):
        return default
    return v


class MOTEvaluator:
    """GOSPA evaluator with a legacy-compatible interface."""

    def __init__(
        self,
        max_distance: float = 1.0,
        min_iou: float = 0.5,
        frame_step: int = 1,
        p: float = 2.0,
        alpha: float = 2.0,
    ):
        self._gospapy = _try_import_gospapy()
        self.max_distance = float(max_distance)
        self.min_iou = float(min_iou)
        self.frame_step = int(frame_step)
        self.p = float(p)
        self.alpha = float(alpha)
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
    def _count_tracks(by_frame: Dict[int, List[Dict[str, Any]]], key: str) -> int:
        ids = set()
        for rows in by_frame.values():
            for row in rows:
                ids.add(int(row[key]))
        return len(ids)

    def _run_gospa(self) -> Dict[str, Any]:
        gospapy = self._gospapy

        all_frames = sorted(set(self.predictions.keys()) | set(self.ground_truth.keys()))
        gospa_values: List[float] = []
        gospa_loc_values: List[float] = []
        gospa_missed_values: List[float] = []
        gospa_false_values: List[float] = []

        for frame in all_frames:
            gt_rows = self.ground_truth.get(frame, [])
            pred_rows = self.predictions.get(frame, [])

            targets = [np.asarray([float(r["x"]), float(r["y"])], dtype=np.float64) for r in gt_rows]
            tracks = [np.asarray([float(r["x"]), float(r["y"])], dtype=np.float64) for r in pred_rows]

            if gospapy is not None:
                gospa, _assignment, gospa_loc, gospa_missed, gospa_false = gospapy.calculate_gospa(
                    targets=targets,
                    tracks=tracks,
                    c=self.max_distance,
                    p=self.p,
                    alpha=self.alpha,
                )
            else:
                gospa, _assignment, gospa_loc, gospa_missed, gospa_false = _calculate_gospa_local(
                    targets=targets,
                    tracks=tracks,
                    c=self.max_distance,
                    p=self.p,
                    alpha=self.alpha,
                )

            gospa_values.append(_as_float(gospa, 0.0))
            gospa_loc_values.append(_as_float(gospa_loc, 0.0))
            gospa_missed_values.append(_as_float(gospa_missed, 0.0))
            gospa_false_values.append(_as_float(gospa_false, 0.0))

        gospa_arr = np.asarray(gospa_values, dtype=np.float64)
        loc_arr = np.asarray(gospa_loc_values, dtype=np.float64)
        missed_arr = np.asarray(gospa_missed_values, dtype=np.float64)
        false_arr = np.asarray(gospa_false_values, dtype=np.float64)

        results = {
            "GOSPA": float(np.mean(gospa_arr)) if gospa_arr.size else 0.0,
            "GOSPA_loc": float(np.mean(loc_arr)) if loc_arr.size else 0.0,
            "GOSPA_missed": float(np.mean(missed_arr)) if missed_arr.size else 0.0,
            "GOSPA_false": float(np.mean(false_arr)) if false_arr.size else 0.0,
            "GOSPA_sum": float(np.sum(gospa_arr)) if gospa_arr.size else 0.0,
            "GOSPA_loc_sum": float(np.sum(loc_arr)) if loc_arr.size else 0.0,
            "GOSPA_missed_sum": float(np.sum(missed_arr)) if missed_arr.size else 0.0,
            "GOSPA_false_sum": float(np.sum(false_arr)) if false_arr.size else 0.0,
            "num_frames": int(len(all_frames)),
            "num_GT_tracks": int(self._count_tracks(self.ground_truth, "person_id")),
            "num_pred_tracks": int(self._count_tracks(self.predictions, "track_id")),
            "max_distance": float(self.max_distance),
            "p": float(self.p),
            "alpha": float(self.alpha),
            "backend": "gospapy" if gospapy is not None else "local",
        }
        self.metrics = results
        return results

    def evaluate(self) -> Dict[str, Any]:
        return self._run_gospa()

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
        print("GOSPA RESULTS")
        print("=" * 70)
        print(f"  Backend:       {results.get('backend', 'local')}")
        print(f"  Max distance:  {self._fmt_float(results.get('max_distance'), '.3f')} m")
        print(f"  p / alpha:     {self._fmt_float(results.get('p'), '.2f')} / {self._fmt_float(results.get('alpha'), '.2f')}")
        print(f"  Frames:        {int(results.get('num_frames', 0))}")

        print("\nPrimary Metrics:")
        print(f"  GOSPA:         {self._fmt_float(results.get('GOSPA'), '8.4f')}")
        print(f"  Localization:  {self._fmt_float(results.get('GOSPA_loc'), '8.4f')} (mean powered component)")
        print(f"  Missed:        {self._fmt_float(results.get('GOSPA_missed'), '8.4f')} (mean powered component)")
        print(f"  False:         {self._fmt_float(results.get('GOSPA_false'), '8.4f')} (mean powered component)")
        print("=" * 70)

    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[GOSPA] Saved results to: {output_file}")

