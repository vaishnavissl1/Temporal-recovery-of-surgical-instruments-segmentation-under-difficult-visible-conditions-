"""
Temporal alignment operators W_{s->t} (report Sec. IV-E, eq. 16).

Three operators are available:

  identity       No spatial transform.  Honest baseline for 1 Hz frames.
  farneback      Dense optical flow (ablation, Sec. V-J.4).
  region         Region-level correspondence via eq. (24): each instrument
                 region in the source mask is matched to the best region in
                 the target mask's base prediction by a combined cost of mask
                 IoU, appearance cosine and normalised centroid distance, then
                 translated by the centroid offset.  Unmatched source regions
                 are DISCARDED (not carried forward at their old position) to
                 prevent false positives from stale instrument locations.

All operators warp a *label map* and so always use nearest-neighbour
interpolation: interpolating class indices is meaningless.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# helpers shared with sitr/identity.py
# --------------------------------------------------------------------------- #
def _extract_regions(mask: np.ndarray,
                     instrument_ids: Optional[Sequence[int]] = None,
                     ) -> List[Dict]:
    """Extract connected instrument regions from a label map.

    Returns a list of dicts, each with keys:
        label    : int — the class id
        pixels   : np.ndarray[bool] — [H,W] boolean mask of this region
        centroid : (float, float) — (row, col) centroid
        area     : int — number of pixels
    """
    if instrument_ids is None:
        # treat every non-zero class as an instrument
        instrument_ids = [int(v) for v in np.unique(mask) if v != 0]

    regions: List[Dict] = []
    for cid in instrument_ids:
        class_mask = (mask == cid).astype(np.uint8)
        if class_mask.sum() == 0:
            continue
        n_cc, labels = cv2.connectedComponents(class_mask, connectivity=8)
        for cc in range(1, n_cc):
            pixels = (labels == cc)
            area = int(pixels.sum())
            if area == 0:
                continue
            ys, xs = np.where(pixels)
            centroid = (float(ys.mean()), float(xs.mean()))
            regions.append(dict(label=int(cid), pixels=pixels,
                                centroid=centroid, area=area))
    return regions


def _region_iou(a_pixels: np.ndarray, b_pixels: np.ndarray) -> float:
    inter = int(np.logical_and(a_pixels, b_pixels).sum())
    union = int(np.logical_or(a_pixels, b_pixels).sum())
    return inter / max(union, 1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.clip(np.dot(a.ravel(), b.ravel()) / (na * nb), -1.0, 1.0))


# --------------------------------------------------------------------------- #
# alignment operators
# --------------------------------------------------------------------------- #
class IdentityAlign:
    """W_{s->t} = identity.  Honest default for 1 Hz released frames."""

    name = "identity"

    def __call__(self, mask_s: np.ndarray, gray_s: Optional[np.ndarray] = None,
                 gray_t: Optional[np.ndarray] = None) -> np.ndarray:
        return mask_s


class FarnebackAlign:
    """Dense optical flow alignment (ablation, Sec. V-J.4).

    Flow is computed from frame t back to frame s, so that every pixel of the
    target frame reads its source location in frame s directly -- this is the
    correct direction for backward warping and avoids forward-scatter holes.
    """

    name = "farneback"

    def __init__(self, pyr_scale: float = 0.5, levels: int = 4, winsize: int = 21,
                 iterations: int = 3, poly_n: int = 7, poly_sigma: float = 1.5):
        self.kw = dict(pyr_scale=pyr_scale, levels=levels, winsize=winsize,
                       iterations=iterations, poly_n=poly_n,
                       poly_sigma=poly_sigma, flags=0)
        self._grid_cache: Optional[tuple] = None

    def _grid(self, h: int, w: int):
        if self._grid_cache is None or self._grid_cache[0] != (h, w):
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            self._grid_cache = ((h, w), xs, ys)
        return self._grid_cache[1], self._grid_cache[2]

    def __call__(self, mask_s: np.ndarray, gray_s: Optional[np.ndarray] = None,
                 gray_t: Optional[np.ndarray] = None) -> np.ndarray:
        if gray_s is None or gray_t is None:
            return mask_s
        if gray_s.shape != gray_t.shape:
            gray_s = cv2.resize(gray_s, (gray_t.shape[1], gray_t.shape[0]),
                                interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(gray_t, gray_s, None, **self.kw)

        h, w = mask_s.shape[:2]
        if (h, w) != flow.shape[:2]:
            sy, sx = h / flow.shape[0], w / flow.shape[1]
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= sx
            flow[..., 1] *= sy

        xs, ys = self._grid(h, w)
        map_x = xs + flow[..., 0]
        map_y = ys + flow[..., 1]
        return cv2.remap(mask_s, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class RegionAligner:
    """Region-level correspondence alignment — report eq. (24) applied to warping.

    For each connected instrument region in the source mask, find its best match
    in the target mask's base prediction by a combined cost of:
        lambda_m * (1 - IoU)  +  lambda_a * (1 - cos(E_i, E_j))  +  lambda_d * d(c_i, c_j)

    Matched regions are translated by their centroid offset.  Unmatched source
    regions are DISCARDED — carrying a stale region at its old position creates
    false positives when the instrument has moved.

    When no target mask is available (mask_t not provided), falls back to identity.
    """

    name = "region"

    def __init__(self, lambda_m: float = 1 / 3, lambda_a: float = 1 / 3,
                 lambda_d: float = 1 / 3,
                 instrument_ids: Optional[Sequence[int]] = None):
        self.lambda_m = float(lambda_m)
        self.lambda_a = float(lambda_a)
        self.lambda_d = float(lambda_d)
        self.instrument_ids = list(instrument_ids) if instrument_ids else None
        # target mask and embeddings are set externally before each call
        self._mask_t: Optional[np.ndarray] = None
        self._features_s: Optional[np.ndarray] = None
        self._features_t: Optional[np.ndarray] = None

    def set_target(self, mask_t: np.ndarray,
                   features_s: Optional[np.ndarray] = None,
                   features_t: Optional[np.ndarray] = None) -> None:
        """Set the current frame's mask and features for region matching.

        Must be called before __call__ when region-level matching is desired.
        If not called, the aligner falls back to identity.
        """
        self._mask_t = mask_t
        self._features_s = features_s
        self._features_t = features_t

    def clear_target(self) -> None:
        self._mask_t = None
        self._features_s = None
        self._features_t = None

    def __call__(self, mask_s: np.ndarray, gray_s: Optional[np.ndarray] = None,
                 gray_t: Optional[np.ndarray] = None) -> np.ndarray:
        if self._mask_t is None:
            return mask_s  # fallback: no target info

        h, w = mask_s.shape[:2]
        inst_ids = self.instrument_ids
        if inst_ids is None:
            all_ids = set(np.unique(mask_s).tolist()) | set(np.unique(self._mask_t).tolist())
            inst_ids = [v for v in all_ids if v != 0]
        if not inst_ids:
            return mask_s

        src_regions = _extract_regions(mask_s, inst_ids)
        tgt_regions = _extract_regions(self._mask_t, inst_ids)

        if not src_regions or not tgt_regions:
            # no regions to match — if source has regions but target doesn't,
            # discard (target has no instruments to match against)
            if src_regions and not tgt_regions:
                return np.zeros_like(mask_s)
            return mask_s

        diag = float(np.sqrt(h ** 2 + w ** 2))

        # build cost matrix
        n_src, n_tgt = len(src_regions), len(tgt_regions)
        cost = np.full((n_src, n_tgt), 1e6, dtype=np.float64)
        for i, sr in enumerate(src_regions):
            emb_s_i = self._region_embedding(sr, self._features_s)
            for j, tr in enumerate(tgt_regions):
                iou = _region_iou(sr["pixels"], tr["pixels"])
                emb_t_j = self._region_embedding(tr, self._features_t)
                cos_sim = _cosine(emb_s_i, emb_t_j) if (
                    emb_s_i is not None and emb_t_j is not None) else 0.0
                dy = sr["centroid"][0] - tr["centroid"][0]
                dx = sr["centroid"][1] - tr["centroid"][1]
                dist = float(np.sqrt(dy ** 2 + dx ** 2)) / max(diag, 1e-6)
                cost[i, j] = (self.lambda_m * (1.0 - iou)
                               + self.lambda_a * (1.0 - cos_sim)
                               + self.lambda_d * dist)

        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        # build warped mask — start with background
        warped = np.zeros_like(mask_s)
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] > 2.0:
                # cost too high — discard this source region
                continue
            sr = src_regions[ri]
            tr = tgt_regions[ci]
            # centroid offset
            dy = tr["centroid"][0] - sr["centroid"][0]
            dx = tr["centroid"][1] - sr["centroid"][1]
            dy_int, dx_int = int(round(dy)), int(round(dx))
            # translate source region pixels
            ys, xs = np.where(sr["pixels"])
            new_ys = ys + dy_int
            new_xs = xs + dx_int
            valid = (new_ys >= 0) & (new_ys < h) & (new_xs >= 0) & (new_xs < w)
            warped[new_ys[valid], new_xs[valid]] = mask_s[ys[valid], xs[valid]]

        return warped

    def _region_embedding(self, region: Dict,
                          features: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Mean-pool features over a region's pixels."""
        if features is None:
            return None
        # features: [D, H, W]
        pixels = region["pixels"]  # [H, W] bool
        if pixels.sum() == 0:
            return None
        return features[:, pixels].mean(axis=1)  # [D]


def build_align(name: str, **kwargs):
    name = (name or "identity").lower()
    if name == "identity":
        return IdentityAlign()
    if name in ("flow", "farneback", "optical_flow"):
        return FarnebackAlign(**kwargs)
    if name == "region":
        return RegionAligner(**kwargs)
    raise ValueError(f"unknown alignment operator {name!r} "
                     "(identity|farneback|region)")
