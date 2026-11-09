#!/usr/bin/env python3
"""
Tests for the pure functions of tools/measure_recovery.py (the Task 3 measurement).
The end-to-end run needs a trained release; the arithmetic that turns per-frame
IoUs into the headline numbers must be right regardless.

    python -m tests.test_measure
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.measure_recovery import (Keeper, interpret, per_class,  # noqa: E402
                                    sign_test_p, summarise)

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


def rec(before, after, routed=True):
    return dict(routed=routed, ib=before, ia=after)


def main() -> int:
    print("\nMeasurement arithmetic")

    @check("sign test: 8 up / 2 down -> two-sided p = 2*(1+10+45)/1024")
    def _():
        assert abs(sign_test_p(8, 2) - 2 * (1 + 10 + 45) / 1024) < 1e-12
        assert sign_test_p(5, 5) == 1.0
        assert np.isnan(sign_test_p(0, 0))

    @check("summary: % improved / worse / unchanged, means, harmful rate by hand")
    def _():
        recs = [rec(0.5, 0.7), rec(0.5, 0.4), rec(0.5, 0.505), rec(0.8, 0.8), rec(0.6, 0.9)]
        s = summarise(recs, "ib", "ia", margin=0.01, label="x", n_boot=50)
        assert s["n"] == 5
        assert abs(s["mean_delta"] - np.mean([0.2, -0.1, 0.005, 0.0, 0.3])) < 1e-12
        assert s["pct_improved"] == 40.0 and s["pct_worse"] == 20.0 and s["pct_unchanged"] == 40.0
        assert s["pct_strictly_better"] == 60.0 and s["pct_strictly_worse"] == 20.0
        assert abs(s["pct_exactly_zero"] - 20.0) < 1e-9
        assert abs(s["mean_gain_when_improved"] - 0.25) < 1e-12
        assert abs(s["mean_damage_when_worse"] - 0.1) < 1e-12

    @check("only ROUTED frames are summarised; unrouted ones never enter")
    def _():
        recs = [rec(0.5, 0.9, routed=False), rec(0.5, 0.4, routed=True)]
        s = summarise(recs, "ib", "ia", 0.01, "x", n_boot=20)
        assert s["n"] == 1 and s["mean_delta"] < 0

    @check("frames whose official IoU is undefined (None) are skipped, not counted as 0")
    def _():
        recs = [rec(None, None), rec(0.5, 0.6)]
        s = summarise(recs, "ib", "ia", 0.01, "x", n_boot=20)
        assert s["n"] == 1

    @check("verdict: damage, noise and improvement are told apart")
    def _():
        bad = summarise([rec(0.6, 0.5)] * 30, "ib", "ia", 0.01, "x", n_boot=100)
        assert interpret(bad).startswith("DAMAGES")
        rng = np.random.default_rng(0)
        d = rng.normal(0, 0.05, 200)
        noise = summarise([rec(0.5, 0.5 + x) for x in d], "ib", "ia", 0.01, "x", n_boot=300)
        assert "NOISE" in interpret(noise) or "HARMFUL" in interpret(noise) or "mixed" in interpret(noise)
        good = summarise([rec(0.5, 0.7)] * 30, "ib", "ia", 0.01, "x", n_boot=100)
        assert interpret(good).startswith("IMPROVES")
        assert interpret(dict(n=0)).startswith("no routed")

    @check("Keeper holds exactly the n largest and the n smallest deltas")
    def _():
        deltas = [0.1, -0.5, 0.3, 0.0, -0.2, 0.9, -0.7]
        big, small = Keeper(2, True), Keeper(2, False)
        for i, d in enumerate(deltas):
            big.offer(d, lambda d=d: d); small.offer(d, lambda d=d: d)
        assert sorted(big.items()) == [0.3, 0.9]
        assert sorted(small.items()) == [-0.7, -0.5]

    @check("per-class: improvement is computed per class over frames where the class is in the GT")
    def _():
        recs = [dict(routed=True, cls_before={1: 0.5, 2: 0.4}, cls_after={1: 0.7, 2: 0.3}),
                dict(routed=True, cls_before={1: 0.6}, cls_after={1: 0.6}),
                dict(routed=False, cls_before={1: 0.1}, cls_after={1: 0.9})]
        pc = per_class(recs, 0.01)
        assert pc[1]["n_frames"] == 2 and abs(pc[1]["mean_delta"] - 0.1) < 1e-12
        assert pc[2]["n_frames"] == 1 and abs(pc[2]["mean_delta"] + 0.1) < 1e-12

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("Measurement arithmetic is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
