#!/usr/bin/env python3
"""
Tests for the difficulty index (sitr/difficulty.py). No dataset, no GPU.

    python -m tests.test_difficulty
"""
from __future__ import annotations

import inspect
import os
import sys
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr import difficulty as D  # noqa: E402

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            FAIL.append((name, e))
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
    return deco


def table(n, rng, shift=0.0):
    return {c: rng.normal(shift, 1.0, n) for c in D.CUE_NAMES}


def main() -> int:
    print("\nDifficulty index")

    @check("no label / ground-truth / IoU parameter anywhere in the public API")
    def _():
        bad = ("iou", "gt", "ground", "label", "target", "mask", "r_star", "error", "y")
        funcs = [D.image_cues, D.stereo_cues, D.within_sequence_blur, D.cue_table,
                 D.DifficultyIndex.fit, D.DifficultyIndex.score,
                 D.DifficultyIndex.cue_ranks, D.DifficultyIndex.quartile_of]
        for fn in funcs:
            for p in inspect.signature(fn).parameters:
                assert p.lower() not in bad and not any(
                    b in p.lower() for b in ("iou", "ground", "label", "target", "r_star")), (fn.__name__, p)

    @check("harder cues give a higher score (monotone, every cue)")
    def _():
        rng = np.random.default_rng(0)
        ref = table(2000, rng)
        idx = D.DifficultyIndex().fit(ref)
        base = {c: np.zeros(1) for c in D.CUE_NAMES}
        s0 = idx.score(base)[0]
        for c in D.CUE_NAMES:
            hard = {k: v.copy() for k, v in base.items()}
            hard[c] = np.array([3.0])
            assert idx.score(hard)[0] > s0, c

    @check("score is in [0,1], and a typical reference frame scores near 0.5")
    def _():
        rng = np.random.default_rng(1)
        ref = table(3000, rng)
        idx = D.DifficultyIndex().fit(ref)
        s = idx.score(table(500, rng))
        assert s.min() >= 0 and s.max() <= 1
        assert abs(s.mean() - 0.5) < 0.05, s.mean()

    @check("fit stores only sorted reference values (no weights are learned)")
    def _():
        rng = np.random.default_rng(2)
        idx = D.DifficultyIndex().fit(table(100, rng))
        assert set(idx.ref) == set(D.CUE_NAMES)
        for v in idx.ref.values():
            assert np.all(np.diff(v) >= 0)

    @check("equal weights: swapping which cue is hard leaves the score unchanged")
    def _():
        rng = np.random.default_rng(3)
        same = rng.normal(0, 1, 2000)                  # identical reference for every cue
        idx = D.DifficultyIndex().fit({c: same for c in D.CUE_NAMES})
        a = {c: np.zeros(1) for c in D.CUE_NAMES}; a["specular"] = np.array([2.0])
        b = {c: np.zeros(1) for c in D.CUE_NAMES}; b["haze"] = np.array([2.0])
        assert abs(idx.score(a)[0] - idx.score(b)[0]) < 1e-9

    @check("blur is standardised WITHIN each sequence (10x level gaps vanish)")
    def _():
        rng = np.random.default_rng(4)
        lap = np.r_[np.exp(rng.normal(np.log(400), 0.3, 100)), np.exp(rng.normal(np.log(40), 0.3, 100))]
        seq = np.r_[np.zeros(100), np.ones(100)]
        z = D.within_sequence_blur(lap, seq)
        assert abs(np.median(z[:100])) < 0.1 and abs(np.median(z[100:])) < 0.1
        # inside one sequence a blurrier frame is HIGHER
        sharp, blurry = np.array([500.0, 100.0]), None
        zz = D.within_sequence_blur(np.r_[np.full(50, 400.0) * np.exp(rng.normal(0, .1, 50)), 100.0],
                                    np.zeros(51))
        assert zz[-1] > 1.0

    @check("quartiles: 0 easiest .. 3 hardest, about equal sizes")
    def _():
        v = np.random.default_rng(5).random(1000)
        q = D.DifficultyIndex.quartile_of(v)
        assert set(np.unique(q)) == {0, 1, 2, 3}
        assert all(abs((q == k).mean() - 0.25) < 0.02 for k in range(4))
        assert v[q == 3].min() > v[q == 0].max()

    @check("image_cues: smoke-like haze raises the dark channel, blur lowers Laplacian variance")
    def _():
        rng = np.random.default_rng(6)
        img = cv2.GaussianBlur(rng.integers(0, 255, (1024, 1280, 3), dtype=np.uint8), (0, 0), 1.0)
        c0 = D.image_cues(img)
        hazy = (0.4 * img + 0.6 * 200).astype(np.uint8)
        c1 = D.image_cues(hazy)
        blur = cv2.GaussianBlur(img, (0, 0), 6.0)
        c2 = D.image_cues(blur)
        i = D.IMAGE_CUE_NAMES.index
        assert c1[i("dark_channel")] > c0[i("dark_channel")] + 20
        assert c1[i("contrast")] < c0[i("contrast")]
        assert c2[i("lap_var")] < 0.05 * c0[i("lap_var")]

    @check("stereo_cues: fractions are over the in-view region only")
    def _():
        inv = np.zeros((10, 10), bool); inv[:, :5] = True
        ok = np.zeros((10, 10), bool); ok[:, :2] = True      # 2 of 5 columns pass
        occ = np.zeros((10, 10), bool); occ[:, 4] = True      # 1 of 5 occluded
        spec = np.ones((10, 10), bool)                        # everywhere, but only in-view counts
        c = D.stereo_cues(ok, occ, spec, inv)
        assert abs(c["stereo_lr_fail"] - 0.6) < 1e-9 and abs(c["occluded"] - 0.2) < 1e-9
        assert abs(c["specular"] - 1.0) < 1e-9

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("Difficulty-index invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
