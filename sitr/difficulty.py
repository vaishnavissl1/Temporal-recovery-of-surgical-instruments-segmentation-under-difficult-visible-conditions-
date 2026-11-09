"""Objective per-frame DIFFICULTY INDEX (novelty item 6).

Built from OBSERVABLE cues only: stereo evidence and image statistics. No ground
truth, no manual labels, no model output, and nothing is fitted to segmentation
error. That is what makes validating it against (1 - IoU) a real test and not a
circular one.

Cues (each has a fixed, a-priori direction: higher = harder):
  stereo_lr_fail  fraction of the in-view frame failing the left-right check
  occluded        fraction of the in-view frame hidden from the right camera
  specular        fraction of the frame that is saturated glare
  blur            LOW sharpness: -z of log(Laplacian variance), z taken WITHIN the
                  sequence (the raw value differs ~10x between sequences, so a
                  global threshold would measure the sequence, not the frame)
  haze            mean dark channel (smoke / blood film lifts it; He et al. 2009)
  low_contrast    low grey-level standard deviation

Combination: each cue is converted to its empirical-CDF rank against a REFERENCE
set of frames (the training frames) and the ranks are averaged with EQUAL weights.
No weight is learned, so nothing is tuned.

Offline-analysis caveat (stated, not hidden): `blur` standardises within the
whole sequence, so it peeks at the sequence's other frames. That is fine for the
index as a measurement tool; a deployed gate must not use it causally.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np

IMAGE_CUE_NAMES = ("lap_var", "brightness", "contrast", "dark_channel",
                   "under_frac", "over_frac")
CUE_NAMES = ("stereo_lr_fail", "occluded", "specular", "blur", "haze", "low_contrast")


def image_cues(left_bgr: np.ndarray) -> np.ndarray:
    """Six raw per-frame image statistics (float32, order = IMAGE_CUE_NAMES).
    Laplacian variance is taken at FULL resolution (it is a high-frequency
    measure); the rest at 640x512, which does not change a frame-level mean."""
    g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    small = cv2.resize(left_bgr, (640, 512), interpolation=cv2.INTER_AREA)
    gs = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dark = cv2.erode(small.min(axis=2), np.ones((15, 15), np.uint8))
    return np.array([lap, float(gs.mean()), float(gs.std()), float(dark.mean()),
                     float((gs < 20).mean()), float((gs > 250).mean())], np.float32)


def stereo_cues(lr_ok: np.ndarray, occluded: np.ndarray, specular: np.ndarray,
                in_view: np.ndarray) -> Dict[str, float]:
    """Whole-frame stereo cues over the in-view region (model-free)."""
    n = max(int(in_view.sum()), 1)
    return dict(stereo_lr_fail=float((in_view & ~lr_ok).sum() / n),
                occluded=float((in_view & occluded).sum() / n),
                specular=float((in_view & specular).sum() / n))


def within_sequence_blur(lap_var: np.ndarray, seq_ids: Sequence) -> np.ndarray:
    """-z of log(Laplacian variance) within each sequence (robust: median/MAD).
    Higher = blurrier relative to the sequence's own typical sharpness."""
    lv = np.log(np.maximum(np.asarray(lap_var, float), 1e-6))
    ids = np.asarray(seq_ids)
    out = np.zeros_like(lv)
    for s in np.unique(ids):
        m = ids == s
        med = np.median(lv[m])
        mad = 1.4826 * np.median(np.abs(lv[m] - med)) + 1e-6
        out[m] = -(lv[m] - med) / mad
    return out


def cue_table(stereo: Dict[str, np.ndarray], image: np.ndarray, seq_ids: Sequence
              ) -> Dict[str, np.ndarray]:
    """Assemble the six cues for a set of frames.
    stereo: arrays for stereo_lr_fail / occluded / specular.
    image : [N, 6] raw image cues (order IMAGE_CUE_NAMES)."""
    im = {n: image[:, i] for i, n in enumerate(IMAGE_CUE_NAMES)}
    return dict(stereo_lr_fail=np.asarray(stereo["stereo_lr_fail"], float),
                occluded=np.asarray(stereo["occluded"], float),
                specular=np.asarray(stereo["specular"], float),
                blur=within_sequence_blur(im["lap_var"], seq_ids),
                haze=im["dark_channel"].astype(float),
                low_contrast=-im["contrast"].astype(float))     # lower std = harder


class DifficultyIndex:
    """Equal-weight mean of per-cue empirical-CDF ranks against a reference set.

    `fit` only stores sorted reference values per cue (no labels involved).
    `cue_ranks` / `score` map any frames onto [0, 1]; higher = harder."""

    def __init__(self, cues: Sequence[str] = CUE_NAMES):
        self.cues = tuple(cues)
        self.ref: Dict[str, np.ndarray] = {}

    def fit(self, table: Dict[str, np.ndarray]) -> "DifficultyIndex":
        for c in self.cues:
            v = np.asarray(table[c], float)
            self.ref[c] = np.sort(v[np.isfinite(v)])
        return self

    def cue_ranks(self, table: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.ref:
            raise RuntimeError("fit() first")
        out = {}
        for c in self.cues:
            ref, v = self.ref[c], np.asarray(table[c], float)
            lo = np.searchsorted(ref, v, side="left")
            hi = np.searchsorted(ref, v, side="right")
            out[c] = (lo + hi) / (2.0 * len(ref))          # mid-rank ECDF, ties handled
        return out

    def score(self, table: Dict[str, np.ndarray]) -> np.ndarray:
        r = self.cue_ranks(table)
        return np.mean(np.stack([r[c] for c in self.cues], 0), axis=0)

    @staticmethod
    def quartile_edges(values: np.ndarray) -> np.ndarray:
        return np.quantile(np.asarray(values, float), [0.25, 0.5, 0.75])

    @staticmethod
    def quartile_of(values: np.ndarray, edges: Optional[np.ndarray] = None) -> np.ndarray:
        """0 = easiest quartile .. 3 = hardest. Edges default to the values' own."""
        v = np.asarray(values, float)
        e = DifficultyIndex.quartile_edges(v) if edges is None else edges
        return np.searchsorted(e, v, side="right")
