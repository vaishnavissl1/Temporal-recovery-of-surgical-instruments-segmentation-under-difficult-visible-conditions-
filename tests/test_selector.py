#!/usr/bin/env python3
"""
Tests for the benefit selector (sitr/selector.py). No dataset, no GPU.

    python -m tests.test_selector

Includes the ground-truth leak test for the selector: ground-truth-derived deltas
may enter only `fit`; predict / choose must produce identical output however the
deltas change, and must not accept them.
"""
from __future__ import annotations

import inspect
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.selector import (CANDIDATES, BenefitSelector, oracle_choice,  # noqa: E402
                           policy_value)

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


def synth(n=600, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (n, 4)).astype(np.float32)
    G = {a: rng.normal(0, 1, (n, 3)).astype(np.float32) for a in CANDIDATES}
    # A1 helps when F0 > 0.5, A2 helps when F1 > 0.5, A3 always slightly harms
    d = {"A1": np.where(F[:, 0] > .5, .2, -.05) + rng.normal(0, .01, n),
         "A2": np.where(F[:, 1] > .5, .15, -.08) + rng.normal(0, .01, n),
         "A3": np.full(n, -.01) + rng.normal(0, .005, n)}
    return F, G, d


def main() -> int:
    print("\nBenefit selector")

    @check("learns which action helps when, and picks it; abstains when none will")
    def _():
        F, G, d = synth()
        sel = BenefitSelector(margin=0.02).fit(F, G, d)
        Ft, Gt, dt = synth(400, seed=1)
        ch = sel.choose(sel.predict(Ft, Gt))
        a1 = Ft[:, 0] > .5
        a2 = (Ft[:, 1] > .5) & ~a1
        neither = (Ft[:, 0] <= .5) & (Ft[:, 1] <= .5)
        assert (ch[a1] == "A1").mean() > 0.85
        assert (ch[neither] == "A0").mean() > 0.85          # abstains
        assert (ch[a2] == "A2").mean() > 0.6
        assert not (ch == "A3").any()                        # never picks the action that only harms

    @check("selective policy beats always-apply on harmful corrections (the point of the method)")
    def _():
        F, G, d = synth()
        sel = BenefitSelector(margin=0.02).fit(F, G, d)
        Ft, Gt, dt = synth(500, seed=2)
        ch = sel.choose(sel.predict(Ft, Gt))
        v_sel = policy_value(ch, dt)
        v_a2 = policy_value(np.full(len(Ft), "A2", dtype=object), dt)
        assert v_sel.mean() > v_a2.mean()
        assert (v_sel < -1e-9).mean() < (v_a2 < -1e-9).mean()

    @check("oracle is a ceiling: never below any policy, and never negative")
    def _():
        F, G, d = synth(300, seed=3)
        sel = BenefitSelector().fit(F, G, d)
        ch = sel.choose(sel.predict(F, G))
        o = policy_value(oracle_choice(d), d)
        assert o.mean() >= policy_value(ch, d).mean() - 1e-12
        assert (o >= -1e-12).all()

    @check("margin controls abstention: a huge margin always abstains, margin -1 never does")
    def _():
        F, G, d = synth()
        sel = BenefitSelector().fit(F, G, d)
        p = sel.predict(F, G)
        sel.margin = 10.0
        assert (sel.choose(p) == "A0").all()
        sel.margin = -10.0
        assert (sel.choose(p) != "A0").all()

    @check("an unavailable action (NaN features) is never chosen")
    def _():
        F, G, d = synth()
        sel = BenefitSelector(margin=0.0).fit(F, G, d)
        G2 = {a: v.copy() for a, v in G.items()}
        G2["A1"][:, 0] = np.nan                              # A1 unavailable everywhere
        ch = sel.choose(sel.predict(F, G2))
        assert not (ch == "A1").any()

    print("\nGround truth never reaches the selector's inference path")

    @check("predict/choose take NO delta/ground-truth argument")
    def _():
        for fn in (BenefitSelector.predict, BenefitSelector.choose):
            params = set(inspect.signature(fn).parameters) - {"self"}
            assert not any(k in p.lower() for p in params for k in ("delta", "gt", "ground", "label", "target")), params

    @check("changing the training deltas AFTER fitting cannot change predictions")
    def _():
        F, G, d = synth()
        sel = BenefitSelector().fit(F, G, d)
        before = sel.predict(F, G)
        for a in d:
            d[a][:] = 123.0                                   # tamper with the arrays that were passed to fit
        after = sel.predict(F, G)
        for a in CANDIDATES:
            assert np.array_equal(before[a], after[a], equal_nan=True)

    @check("two fits on the same data are identical (deterministic)")
    def _():
        F, G, d = synth()
        p1 = BenefitSelector().fit(F, G, d).predict(F, G)
        p2 = BenefitSelector().fit(F, G, d).predict(F, G)
        for a in CANDIDATES:
            assert np.allclose(p1[a], p2[a])

    @check("too little training data falls back to the mean instead of overfitting")
    def _():
        F, G, d = synth(30)
        sel = BenefitSelector().fit(F, G, d)
        assert all(m is None for m in sel.models.values())
        assert np.isfinite(sel.predict(F, G)["A1"]).all()

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("Selector invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
