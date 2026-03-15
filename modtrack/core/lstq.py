"""LSTQ evaluation helpers for RadarScenes point-level tracking."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np


LSTQ_CLASS_MOVING = 1
LSTQ_CLASS_STATIC = 0
LSTQ_NUM_CLASSES = 2


class LSTQEvaluator:
    """LSTQ (Segmentation and Tracking Quality) evaluator."""

    def __init__(self, min_tube_points: int = 1):
        self.min_tube_points = min_tube_points
        self.confusion = np.zeros((LSTQ_NUM_CLASSES, LSTQ_NUM_CLASSES), dtype=np.int64)
        self.gt_tube_sizes: Dict[int, int] = defaultdict(int)
        self.pred_tube_sizes: Dict[int, int] = defaultdict(int)
        self.tube_overlaps: Dict[Tuple[int, int], int] = defaultdict(int)

    def add_frame(
        self,
        pred_sem: np.ndarray,
        pred_inst: np.ndarray,
        gt_sem: np.ndarray,
        gt_inst: np.ndarray,
    ) -> None:
        assert len(pred_sem) == len(gt_sem) == len(pred_inst) == len(gt_inst)
        n = len(pred_sem)
        if n == 0:
            return

        for i in range(n):
            ps = int(pred_sem[i])
            gs = int(gt_sem[i])
            if 0 <= ps < LSTQ_NUM_CLASSES and 0 <= gs < LSTQ_NUM_CLASSES:
                self.confusion[ps, gs] += 1

        for i in range(n):
            gt_s = int(gt_sem[i])
            gt_id = int(gt_inst[i])
            pred_id = int(pred_inst[i])

            if gt_s == LSTQ_CLASS_MOVING and gt_id > 0:
                self.gt_tube_sizes[gt_id] += 1
                if int(pred_sem[i]) == LSTQ_CLASS_MOVING and pred_id > 0:
                    self.pred_tube_sizes[pred_id] += 1
                    self.tube_overlaps[(pred_id, gt_id)] += 1
            elif int(pred_sem[i]) == LSTQ_CLASS_MOVING and pred_id > 0:
                self.pred_tube_sizes[pred_id] += 1

    def compute_s_cls(self) -> float:
        conf = self.confusion.astype(np.float64)
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp

        union = tp + fp + fn
        iou_per_class = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        n_present = np.sum(union > 0)
        if n_present == 0:
            return 0.0
        return float(np.sum(iou_per_class) / n_present)

    def compute_s_assoc(self) -> float:
        total_aq = 0.0
        n_tubes = 0

        for gt_id, gt_size in self.gt_tube_sizes.items():
            if gt_size < self.min_tube_points:
                continue
            n_tubes += 1

            inner_sum = 0.0
            for (pred_id, g_id), tpa in self.tube_overlaps.items():
                if g_id != gt_id or tpa == 0:
                    continue
                pred_size = self.pred_tube_sizes.get(pred_id, 0)
                if pred_size == 0:
                    continue
                iou = tpa / (pred_size + gt_size - tpa)
                inner_sum += tpa * iou

            total_aq += inner_sum / gt_size

        if n_tubes == 0:
            return 0.0
        return total_aq / n_tubes

    def compute(self) -> Dict[str, float]:
        s_cls = self.compute_s_cls()
        s_assoc = self.compute_s_assoc()
        lstq = float(np.sqrt(s_cls * s_assoc))

        conf = self.confusion.astype(np.float64)
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        union = tp + fp + fn
        iou_per_class = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)

        return {
            "LSTQ": round(lstq * 100, 2),
            "S_cls": round(s_cls * 100, 2),
            "S_assoc": round(s_assoc * 100, 2),
            "IoU_static": round(float(iou_per_class[0]) * 100, 2),
            "IoU_moving": round(float(iou_per_class[1]) * 100, 2),
            "n_gt_tubes": len([g for g, s in self.gt_tube_sizes.items() if s >= self.min_tube_points]),
            "n_pred_tubes": len(self.pred_tube_sizes),
        }
