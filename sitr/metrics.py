"""
Segmentation metrics — report eq. (33) — plus the scalar statistics Stage 2
needs (per-frame instrument IoU used as the reliability target r*_t, eq. 13).

Two per-frame IoU conventions are implemented because the EndoVis 2018 page
("the IoU for each class which is present in a frame") is ambiguous about
whether a class counts as present via the ground truth only, or via ground
truth OR prediction.  Both are reported; `union` is the stricter one (it
penalises hallucinated classes) and is the default target for r*.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .labels import IGNORE_INDEX


# --------------------------------------------------------------------------- #
# dataset-level confusion matrix
# --------------------------------------------------------------------------- #
class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.n = int(num_classes)
        self.mat = np.zeros((self.n, self.n), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        valid = (gt != IGNORE_INDEX)
        p = pred[valid].astype(np.int64)
        g = gt[valid].astype(np.int64)
        if p.size == 0:
            return
        idx = g * self.n + p
        self.mat += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def iou_per_class(self) -> np.ndarray:
        inter = np.diag(self.mat).astype(np.float64)
        gt = self.mat.sum(1).astype(np.float64)
        pr = self.mat.sum(0).astype(np.float64)
        union = gt + pr - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter / np.maximum(union, 1e-12), np.nan)
        return iou

    def dice_per_class(self) -> np.ndarray:
        inter = np.diag(self.mat).astype(np.float64)
        gt = self.mat.sum(1).astype(np.float64)
        pr = self.mat.sum(0).astype(np.float64)
        denom = gt + pr
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1e-12), np.nan)
        return d

    def summary(self, names: Optional[Sequence[str]] = None) -> Dict:
        iou = self.iou_per_class()
        dice = self.dice_per_class()
        out = {
            "mIoU": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0,
            "mDice": float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0,
        }
        if names is not None:
            out["per_class_iou"] = {
                str(n): (None if np.isnan(v) else float(v))
                for n, v in zip(names, iou)
            }
        return out


# --------------------------------------------------------------------------- #
# per-frame metrics
# --------------------------------------------------------------------------- #
def _class_counts(pred: np.ndarray, gt: Optional[np.ndarray], num_classes: int):
    """Returns (intersection, pred_area, gt_area) per class, over valid pixels."""
    if gt is None:
        p = pred.reshape(-1).astype(np.int64)
        pr = np.bincount(p, minlength=num_classes)[:num_classes]
        return None, pr, None
    valid = (gt != IGNORE_INDEX)
    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)
    pr = np.bincount(p, minlength=num_classes)[:num_classes]
    gc = np.bincount(g, minlength=num_classes)[:num_classes]
    hit = p[p == g]
    inter = np.bincount(hit, minlength=num_classes)[:num_classes]
    return inter, pr, gc


def frame_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int,
              class_ids: Optional[Iterable[int]] = None,
              present: str = "union",
              empty_value: float = 1.0) -> float:
    """Eq. (33), evaluated on one frame and averaged over the classes present.

    class_ids   restrict the average to a subset (e.g. instrument classes only)
    present     "union" -> class counts if it appears in gt OR pred
                "gt"    -> class counts only if it appears in gt
    empty_value value returned when no class in the subset appears anywhere
                (both masks empty: a correct "no instrument" prediction)
    """
    inter, pr, gc = _class_counts(pred, gt, num_classes)
    ids = list(range(num_classes)) if class_ids is None else list(class_ids)
    ids = [c for c in ids if 0 <= c < num_classes]
    if not ids:
        return float(empty_value)

    ious: List[float] = []
    for c in ids:
        g, p = int(gc[c]), int(pr[c])
        appears = (g > 0 or p > 0) if present == "union" else (g > 0)
        if not appears:
            continue
        union = g + p - int(inter[c])
        ious.append(float(inter[c]) / float(union) if union > 0 else 1.0)
    if not ious:
        return float(empty_value)
    return float(np.mean(ious))


def instrument_iou(a: np.ndarray, b: np.ndarray, num_classes: int,
                   instrument_ids: Sequence[int],
                   empty_value: float = 1.0) -> float:
    """IoU restricted to instrument classes.

    Used for two things:
      * r*_t = IoU(S_t, Y_t)                       (eq. 13, the training target)
      * T_t  = IoU(S_t, W_{t'->t}(S_{t'}))         (eq. 7, temporal consistency)

    Note on the empty/empty case: eq. (10) drives the reliability score toward 0
    when the predicted foreground is empty, while an IoU of an empty prediction
    against an empty ground truth is conventionally 1.  Those two conventions
    disagree.  `empty_value` exposes the choice instead of hiding it; the Stage 2
    trainer defaults to 1.0 (honest IoU) and reports how many frames hit it.
    """
    return frame_iou(a, b, num_classes, class_ids=instrument_ids,
                     present="union", empty_value=empty_value)


def foreground_area(mask: np.ndarray, instrument_ids: Sequence[int]) -> int:
    """|Omega^fg_t| — the predicted instrument foreground area (eq. 6)."""
    if len(instrument_ids) == 0:
        return 0
    return int(np.isin(mask, np.asarray(instrument_ids, dtype=mask.dtype)).sum())


# --------------------------------------------------------------------------- #
# lightweight statistics (no scipy / sklearn dependency)
# --------------------------------------------------------------------------- #
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # average ranks within ties
    sa = a[order]
    i = 0
    while i < sa.size:
        j = i
        while j + 1 < sa.size and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(_rankdata(x), _rankdata(y))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC of `scores` for the binary `labels` (1 = positive), rank based."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
