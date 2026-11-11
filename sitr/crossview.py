"""
Cross-view indicators: do the LEFT and RIGHT predicted segmentations agree?  (Task D)

The three disparity-based stereo indicators of the gate (`stereo_disagreement`, `occluded_fraction`,
`disparity_confidence`) are photometric quantities from SGBM matching. They measure how well the
two images can be matched, and matching fails on specular instruments (Task 2: left-right consistency 45.6% on
instrument pixels against 65.2% on tissue), so they do not measure segmentation error.

These six compare the two PREDICTED label maps directly. Rectification alone gives ROW correspondence; disparity is
only needed for COLUMN correspondence. So every statistic below is either geometry-free (a set or a count over the
whole frame) or uses rows only (a per-row profile), and none needs a disparity map or any photometric matching. They
run on class labels, not intensities, so specular highlights do not enter.

    name               definition                                                              geometry
    xv_class_jaccard   Jaccard of the two SETS of instrument classes predicted                  none
    xv_area_ratio_max  max over classes of |a_L - a_R| / (a_L + a_R), a = pixel count           none
    xv_ncomp_diff      mean over classes of |n_L - n_R| / (n_L + n_R), n = 8-connected parts     none
    xv_rowprofile_l1   per class sum_y |m_L(c,y) - m_R(c,y)| / sum_y (m_L + m_R), m = class-c
                       pixel count in row y; mean over classes present in either view           rectification
    xv_rowspan_iou     per class, IoU of the row-support intervals [y_min, y_max]; mean          rectification
    xv_runcount_corr   Pearson correlation, over rows, of the per-row count of class runs        rectification

"Classes" are the instrument class ids passed in (the same set the gate's other indicators use); background and the
anatomy classes are not part of these statistics.

Hard requirements implemented here
  * The three row-based indicators are computed on the RECTIFIED grid. Both label maps are rectified with
    cv2.INTER_NEAREST only (class ids are never interpolated), and the stored per-sequence `VerticalCorrection` is
    applied to the right view as a nearest-neighbour vertical warp, the label analogue of what
    `sitr.stereo.compute_evidence` does to the right image.
  * Field-of-view guard. The two views do not share a horizontal field of view: a left-view pixel at column x < d has
    no counterpart (its match would sit at x - d < 0) and a right-view pixel at x > W - d has none either. A strip as wide
    as the 99th-percentile disparity (`DISPARITY_P99_FULL_PX` = 186 px at the full 1280-wide resolution, measured in
    Task 2, scaled to the grid used) is dropped from the LEFT edge of the left view and the RIGHT edge of the right
    view, and each view is intersected with its own valid-pixel mask (`rect.valid_left` / `rect.valid_right`). The
    surviving fraction is returned with `with_info=True`. Without the guard every indicator is biased.
  * Empty cases. A class absent from both views is skipped. If NO instrument class is predicted in either view the views
    agree on there being nothing, and the stated constants `EMPTY_BOTH` are returned (perfect agreement for every
    indicator). If one view predicts classes and the other predicts none, the formulas apply as written (Jaccard 0, area
    ratio 1, ...). `xv_runcount_corr` is a Pearson correlation: where either vector has zero variance it is undefined,
    and the stated rule is 1.0 if both are constant and equal, otherwise 0.0. No NaN is ever returned.

Ground truth appears nowhere in this module.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

NAMES: Tuple[str, ...] = ("xv_class_jaccard", "xv_area_ratio_max", "xv_ncomp_diff",
                          "xv_rowprofile_l1", "xv_rowspan_iou", "xv_runcount_corr")

#: 99th-percentile disparity at full 1280-px resolution (Task 2, results/task2/STEREO_REPORT.md).
DISPARITY_P99_FULL_PX = 186.0

#: returned when no instrument class is predicted in either view: the views agree on "nothing".
EMPTY_BOTH: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 1.0)

OUTSIDE = 255        # label given to pixels with no source, no counterpart or outside the guard


# --------------------------------------------------------------------------- #
# rectification of a label map that lives on a (H, W) grid = uniform downscale of the original image
# --------------------------------------------------------------------------- #
def _grid_maps(rect, hw: Tuple[int, int]):
    """Maps that rectify a (H, W) label grid, built once per (rectifier, grid) and cached on the rectifier.

    The rectifier's own maps are defined on the full-resolution rectified image and give, for each rectified pixel,
    the source coordinate in the ORIGINAL image. Here the rectified grid is the same uniform downscale as the label
    grid, so the full-resolution maps are sampled at the centres of the small cells and their values rescaled to
    label-grid coordinates. Pixel centres follow cv2.resize's convention."""
    cache = getattr(rect, "_xv_maps", None)
    if cache is None:
        cache = {}
        rect._xv_maps = cache
    key = tuple(hw)
    if key not in cache:
        H, W = key
        w0, h0 = rect.rig.size
        sx, sy = w0 / W, h0 / H
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        xf = ((u + 0.5) * sx - 0.5).astype(np.float32)     # rectified full-res coordinates of each small cell
        yf = ((v + 0.5) * sy - 0.5).astype(np.float32)

        def small(m):
            mx = cv2.remap(m[0], xf, yf, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            my = cv2.remap(m[1], xf, yf, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return (((mx + 0.5) / sx - 0.5).astype(np.float32), ((my + 0.5) / sy - 0.5).astype(np.float32))

        vl = cv2.remap(rect.valid_left.astype(np.uint8), xf, yf, cv2.INTER_NEAREST, borderValue=0) > 0
        vr = cv2.remap(rect.valid_right.astype(np.uint8), xf, yf, cv2.INTER_NEAREST, borderValue=0) > 0
        cache[key] = dict(L=small(rect.mapL), R=small(rect.mapR), valid_left=vl, valid_right=vr,
                          xf=xf, yf=yf, u=u, v=v, sx=sx, sy=sy)
    return cache[key]


def _rectify(label: np.ndarray, maps) -> np.ndarray:
    return cv2.remap(label, maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=OUTSIDE)


def _vertical_warp(label_rect: np.ndarray, corr, g) -> np.ndarray:
    """Nearest-neighbour label analogue of VerticalCorrection.apply: R'(x, y) = R(x, y - dy(x, y)), where dy is defined in
    full-resolution rectified-right pixels (here divided by the vertical downscale of the grid)."""
    if corr is None or not getattr(corr, "applied", False):
        return label_rect
    dy = corr.offset(g["xf"], g["yf"]) / g["sy"]
    return cv2.remap(label_rect, g["u"], (g["v"] - dy).astype(np.float32), cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=OUTSIDE)


# --------------------------------------------------------------------------- #
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    ca, cb = a.std() < 1e-12, b.std() < 1e-12
    if ca or cb:
        return 1.0 if (ca and cb and abs(a.mean() - b.mean()) < 1e-12) else 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def _run_counts(lab: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    """Per row, the number of maximal horizontal runs of one instrument class."""
    inst = np.isin(lab, np.asarray(list(ids), dtype=lab.dtype))
    prev = np.concatenate([np.full((lab.shape[0], 1), OUTSIDE, lab.dtype), lab[:, :-1]], 1)
    return (inst & (lab != prev)).sum(1)


def _n_components(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(n - 1)


def _span(mask: np.ndarray):
    rows = np.flatnonzero(mask.any(1))
    return (int(rows[0]), int(rows[-1])) if rows.size else None


def indicators_from_rectified(L: np.ndarray, R: np.ndarray, instrument_ids: Sequence[int]) -> Tuple[float, ...]:
    """The six indicators from two label maps that are already rectified and guarded (OUTSIDE = ignored).
    Exposed separately so the maths can be tested without any geometry."""
    ids = [int(c) for c in instrument_ids]
    mL = {c: (L == c) for c in ids}
    mR = {c: (R == c) for c in ids}
    aL = {c: int(mL[c].sum()) for c in ids}
    aR = {c: int(mR[c].sum()) for c in ids}
    present = [c for c in ids if aL[c] + aR[c] > 0]
    if not present:
        return EMPTY_BOTH

    setL = {c for c in present if aL[c] > 0}
    setR = {c for c in present if aR[c] > 0}
    jacc = len(setL & setR) / len(setL | setR)
    area_ratio = max(abs(aL[c] - aR[c]) / (aL[c] + aR[c]) for c in present)

    ncomp, prof, span = [], [], []
    for c in present:
        nL, nR = _n_components(mL[c]), _n_components(mR[c])
        ncomp.append(abs(nL - nR) / (nL + nR))
        pl, pr = mL[c].sum(1).astype(np.float64), mR[c].sum(1).astype(np.float64)
        prof.append(float(np.abs(pl - pr).sum() / (pl + pr).sum()))
        sl, sr = _span(mL[c]), _span(mR[c])
        if sl is None or sr is None:
            span.append(0.0)
        else:
            inter = max(0, min(sl[1], sr[1]) - max(sl[0], sr[0]) + 1)
            union = max(sl[1], sr[1]) - min(sl[0], sr[0]) + 1
            span.append(inter / union)
    corr = _pearson(_run_counts(L, ids), _run_counts(R, ids))
    return (float(jacc), float(area_ratio), float(np.mean(ncomp)), float(np.mean(prof)),
            float(np.mean(span)), float(corr))


def crossview_indicators(left: np.ndarray, right: np.ndarray, rect, corr=None,
                         instrument_ids: Sequence[int] = (1,),
                         guard_full_px: float = DISPARITY_P99_FULL_PX,
                         with_info: bool = False):
    """Six cross-view indicators from the left and right PREDICTED label maps.

    left, right : uint8 class maps on the same (H, W) grid, each a uniform downscale of its original image
                  (the segmenter's output grid, or the cache's 2x coarser one)
    rect        : the sequence's `sitr.stereo.Rectifier` (only mapL, mapR, valid_left, valid_right and rig.size are used)
    corr        : the sequence's `VerticalCorrection`, or None
    guard_full_px: field-of-view strip width at FULL resolution; scaled to this grid. 0 disables the guard (tests only).

    Returns the six floats in `NAMES` order; with_info=True also returns a dict with the guard width in grid pixels and
    the surviving fraction of each view."""
    if left.shape != right.shape:
        raise ValueError(f"left {left.shape} and right {right.shape} label maps must share a grid")
    H, W = left.shape
    g = _grid_maps(rect, (H, W))
    Lr = _rectify(left, g["L"])
    Rr = _vertical_warp(_rectify(right, g["R"]), corr, g)
    guard = int(math.ceil(guard_full_px * W / rect.rig.size[0])) if guard_full_px > 0 else 0
    col = g["u"]
    keepL = g["valid_left"] & (col >= guard)
    keepR = g["valid_right"] & (col < W - guard)
    Lr = np.where(keepL, Lr, OUTSIDE).astype(left.dtype)
    Rr = np.where(keepR, Rr, OUTSIDE).astype(left.dtype)
    vals = indicators_from_rectified(Lr, Rr, instrument_ids)
    if not with_info:
        return vals
    return vals, dict(guard_px=guard, surviving_left=float(keepL.mean()), surviving_right=float(keepR.mean()))


def as_dict(vals: Sequence[float]) -> Dict[str, float]:
    return dict(zip(NAMES, (float(v) for v in vals)))
