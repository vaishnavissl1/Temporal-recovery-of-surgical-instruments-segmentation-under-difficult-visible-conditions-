#!/usr/bin/env python3
"""
Tests that our evaluation arithmetic equals the OFFICIAL EndoVis 2018 evaluator
(test_data/utils.py). Hand-computed values, no dataset, no GPU.

    python -m tests.test_eval
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.metrics import (frame_iou, official_class_ious,  # noqa: E402
                          official_frame_iou)

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


def grid(rows):
    return np.array(rows, dtype=np.uint8)


def main() -> int:
    print("\nOfficial per-frame score (utils.py:141-193)")

    @check("IoU = TP/(TP+FP+FN), hand computed")
    def _():
        gt = grid([[1, 1, 1, 1], [0, 0, 0, 0]])
        pr = grid([[1, 1, 0, 0], [1, 0, 0, 0]])
        # class 1: TP=2, FN=2, FP=1 -> 2/5
        v = official_class_ious(pr, gt, 3)
        assert abs(v[1] - 2 / 5) < 1e-12 and list(v) == [1]

    @check("background (class 0) is never scored (utils.py:172)")
    def _():
        gt = grid([[0, 0], [0, 1]])
        pr = grid([[0, 0], [0, 1]])
        assert 0 not in official_class_ious(pr, gt, 3)
        # background perfectly right but instrument totally wrong must score 0
        pr2 = grid([[0, 0], [0, 0]])
        assert official_frame_iou(pr2, gt, 3) == 0.0

    @check("a class absent from the GT is NOT scored even if predicted (utils.py:143)")
    def _():
        gt = grid([[1, 1], [0, 0]])
        pr = grid([[1, 1], [2, 2]])          # hallucinates class 2
        v = official_class_ious(pr, gt, 3)
        assert list(v) == [1] and v[1] == 1.0
        assert official_frame_iou(pr, gt, 3) == 1.0

    @check("frame with no non-background class in the GT returns None (skipped)")
    def _():
        gt = grid([[0, 0], [0, 0]])
        pr = grid([[1, 1], [0, 0]])
        assert official_frame_iou(pr, gt, 3) is None

    @check("frame score is the plain mean over the classes present in the GT")
    def _():
        gt = grid([[1, 1, 2, 2]])
        pr = grid([[1, 0, 2, 2]])
        v = official_class_ious(pr, gt, 3)
        assert abs(v[1] - 0.5) < 1e-12 and abs(v[2] - 1.0) < 1e-12
        assert abs(official_frame_iou(pr, gt, 3) - 0.75) < 1e-12

    @check("the repo's legacy frame_iou(class_ids=None) differs: it counts background")
    def _():
        # documents WHY we need the official function: the legacy default
        # averages class 0 too, which inflates the score on mostly-background frames
        gt = np.zeros((10, 10), np.uint8); gt[0, :2] = 1
        pr = np.zeros((10, 10), np.uint8)
        legacy = frame_iou(pr, gt, 3, present="gt")
        assert official_frame_iou(pr, gt, 3) == 0.0
        assert legacy > 0.4, legacy

    print("\nRisk-coverage / AURC")
    from sitr.metrics import aurc, aurc_oracle, risk_coverage

    @check("AURC: perfect ranking = oracle floor, reversed ranking is worst, random ~ mean loss")
    def _():
        loss = np.array([0.0, 0.1, 0.2, 0.9, 1.0])
        perfect = -loss                         # trust the low-loss frames first
        assert abs(aurc(perfect, loss) - aurc_oracle(loss)) < 1e-12
        # hand computed: risks at k=1..5 = 0, .05, .1, .3, .44 -> mean 0.178
        assert abs(aurc_oracle(loss) - np.mean([0, .05, .1, .3, .44])) < 1e-12
        assert aurc(loss, loss) > aurc(perfect, loss)
        rng = np.random.default_rng(0)
        big = rng.random(5000)
        assert abs(aurc(rng.random(5000), big) - big.mean()) < 0.02

    @check("risk-coverage: coverage runs 1/n..1 and the last risk is the overall mean")
    def _():
        loss = np.array([.3, .1, .8, .4])
        cov, risk = risk_coverage(np.array([4., 3., 1., 2.]), loss)
        assert np.allclose(cov, [.25, .5, .75, 1.0])
        assert abs(risk[-1] - loss.mean()) < 1e-12
        assert abs(risk[0] - 0.3) < 1e-12     # highest score first -> loss .3

    @check("AURC ignores non-finite entries instead of propagating NaN")
    def _():
        assert np.isfinite(aurc(np.array([1., np.nan, 2.]), np.array([.1, .2, .3])))

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("Our evaluation arithmetic matches the official evaluator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
