#!/usr/bin/env python3
"""
Tests for the candidate actions (sitr/actions.py). No dataset, no GPU.

    python -m tests.test_actions

The most important one proves ground truth cannot reach the inference path: a
state whose `ground_truth` raises on ANY access is run through every action and
every feature function.
"""
from __future__ import annotations

import os
import sys
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.actions import (ACTION_FEATURE_NAMES, ActionContext, ActionParams,  # noqa: E402
                          action_features, apply_all, delta_scores, a0_keep,
                          a1_stereo, a2_temporal, a3_refine)

PASS, FAIL = [], []
NAMES = ["bg", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma",
         "covered-kidney", "thread", "clamps", "suturing-needle", "suction-instrument",
         "small-intestine", "ultrasound-probe"]
INSTR = [1, 2, 3, 6, 7, 8, 9, 11]


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


class PoisonedGT:
    """A ground-truth stand-in that fails the test on any use."""
    def __getattr__(self, k):
        raise AssertionError(f"ground truth touched: .{k}")

    def __array__(self, *a, **k):
        raise AssertionError("ground truth converted to an array")


class State(SimpleNamespace):
    @property
    def ground_truth(self):
        raise AssertionError("FrameState.ground_truth was READ on the inference path")


def make_state(H=40, W=50, mask=None, seed=0, memory=None):
    rng = np.random.default_rng(seed)
    if mask is None:
        mask = np.zeros((H, W), np.uint8)
    H, W = mask.shape
    probs = np.full((12, H, W), 0.01, np.float32)
    probs[mask.astype(int), np.arange(H)[:, None], np.arange(W)[None, :]] = 0.9
    return State(mask=mask, probs=probs, memory=memory or [], reliability=0.4, t=3)


def ev(H, W, ok=True):
    return dict(lr_ok=np.full((H, W), ok), occluded=np.zeros((H, W), bool),
                confidence=np.full((H, W), 255, np.uint8), in_view=np.ones((H, W), bool))


def main() -> int:
    print("\nActions")

    @check("A0 returns an identical COPY (editing it cannot corrupt the base mask)")
    def _():
        s = make_state(mask=np.ones((40, 50), np.uint8))
        out = a0_keep(ActionContext(state=s))
        assert np.array_equal(out, s.mask) and out is not s.mask
        out[0, 0] = 9
        assert s.mask[0, 0] == 1

    @check("A1 changes a pixel only when left is unsure AND right is confident AND stereo is usable")
    def _():
        H, W = 40, 50
        mask = np.zeros((H, W), np.uint8)
        s = make_state(mask=mask)
        s.probs[:, :20, :] = 1 / 12.0                          # left unsure in the top half
        right = np.full((H, W), 1, np.uint8)                    # right says shaft everywhere
        rconf = np.full((H, W), 250, np.uint8)
        ctx = ActionContext(state=s, evidence=ev(H, W), right_label=right, right_conf=rconf)
        out = a1_stereo(ctx)
        assert (out[:20] == 1).all() and (out[20:] == 0).all()
        # right not confident -> no change
        ctx2 = ActionContext(state=s, evidence=ev(H, W), right_label=right, right_conf=np.full((H, W), 100, np.uint8))
        assert np.array_equal(a1_stereo(ctx2), mask)
        # stereo says disagreement (LR failed) -> no change
        ctx3 = ActionContext(state=s, evidence=ev(H, W, ok=False), right_label=right, right_conf=rconf)
        assert np.array_equal(a1_stereo(ctx3), mask)
        # right label invalid (255) -> no change
        ctx4 = ActionContext(state=s, evidence=ev(H, W), right_label=np.full((H, W), 255, np.uint8), right_conf=rconf)
        assert np.array_equal(a1_stereo(ctx4), mask)

    @check("A1 is unavailable (None) without stereo information")
    def _():
        assert a1_stereo(ActionContext(state=make_state())) is None

    @check("A2 is unavailable on an empty memory and delegates to the recovery otherwise")
    def _():
        s = make_state()
        rec = lambda st: SimpleNamespace(mask=np.full_like(st.mask, 4))
        assert a2_temporal(ActionContext(state=s, recovery=rec)) is None
        s2 = make_state(memory=[SimpleNamespace(reliability=0.9)])
        out = a2_temporal(ActionContext(state=s2, recovery=rec))
        assert (out == 4).all()

    @check("A3 re-labels a small island inside an organ and leaves big regions alone")
    def _():
        H, W = 60, 80
        mask = np.full((H, W), 5, np.uint8)                     # covered-kidney everywhere
        mask[30:33, 40:43] = 4                                   # 9 px island of kidney-parenchyma
        mask[5:25, 5:25] = 10                                    # big intestine block (400 px) stays
        s = make_state(mask=mask)
        out = a3_refine(ActionContext(state=s, target_names=NAMES, instrument_ids=INSTR), ActionParams())
        assert (out[30:33, 40:43] == 5).all()
        assert (out[5:25, 5:25] == 10).all()

    @check("A3 bridges a 1-px gap in a thin class ONLY where the model gave it probability")
    def _():
        H, W = 40, 60
        mask = np.zeros((H, W), np.uint8)
        mask[20, 5:30] = 1; mask[20, 31:55] = 1                  # shaft with a 1-px gap at x=30
        s = make_state(mask=mask)
        s.probs[1, 20, 30] = 0.4                                  # the model half-believed it
        s.probs[0, 20, 30] = 0.5
        out = a3_refine(ActionContext(state=s, target_names=NAMES, instrument_ids=INSTR), ActionParams())
        assert out[20, 30] == 1
        s2 = make_state(mask=mask)                                # no belief -> must NOT bridge
        s2.probs[1, 20, 30] = 0.01
        out2 = a3_refine(ActionContext(state=s2, target_names=NAMES, instrument_ids=INSTR), ActionParams())
        assert out2[20, 30] == 0

    @check("A3 removes a thin-class speck but keeps a real thread")
    def _():
        H, W = 40, 60
        mask = np.zeros((H, W), np.uint8)
        mask[10, 10] = 6                                          # 1 px speck of thread
        mask[30, 5:50] = 6                                        # a real 45 px thread
        s = make_state(mask=mask)
        out = a3_refine(ActionContext(state=s, target_names=NAMES, instrument_ids=INSTR), ActionParams())
        assert out[10, 10] != 6 and (out[30, 5:50] == 6).all()

    print("\nBaselines")

    @check("FUSE = temporal then stereo on top; with no memory it is stereo alone; with neither it is A0")
    def _():
        from sitr.actions import a_fuse, a_fuse_refine
        H, W = 40, 50
        mask = np.zeros((H, W), np.uint8)
        s = make_state(mask=mask, memory=[SimpleNamespace(reliability=0.9)])
        s.probs[:, :10, :] = 1 / 12.0                           # left unsure in the top 10 rows
        rec = lambda st: SimpleNamespace(mask=np.full_like(st.mask, 4))
        right = np.full((H, W), 1, np.uint8); rc = np.full((H, W), 250, np.uint8)
        ctx = ActionContext(state=s, evidence=ev(H, W), right_label=right, right_conf=rc,
                            target_names=NAMES, instrument_ids=INSTR, recovery=rec)
        f = a_fuse(ctx)
        assert (f[:10] == 1).all() and (f[10:] == 4).all()        # stereo overrides the unsure rows, temporal elsewhere
        s2 = make_state(mask=mask); s2.probs[:, :10, :] = 1 / 12.0
        ctx2 = ActionContext(state=s2, evidence=ev(H, W), right_label=right, right_conf=rc,
                             target_names=NAMES, instrument_ids=INSTR, recovery=rec)
        f2 = a_fuse(ctx2)
        assert (f2[:10] == 1).all() and (f2[10:] == 0).all()      # no memory -> stereo only
        ctx3 = ActionContext(state=make_state(mask=mask), target_names=NAMES, instrument_ids=INSTR)
        assert np.array_equal(a_fuse(ctx3), mask)                  # neither -> unchanged
        assert a_fuse_refine(ctx).shape == mask.shape

    @check("building a context for FUSE does not mutate the original state's mask")
    def _():
        from sitr.actions import a_fuse
        mask = np.zeros((40, 50), np.uint8)
        s = make_state(mask=mask, memory=[SimpleNamespace(reliability=0.9)])
        rec = lambda st: SimpleNamespace(mask=np.full_like(st.mask, 4))
        a_fuse(ActionContext(state=s, target_names=NAMES, instrument_ids=INSTR, recovery=rec))
        assert (s.mask == 0).all()

    print("\nGround truth never reaches the inference path")

    @check("every action and feature function runs with a state whose ground_truth RAISES on access")
    def _():
        H, W = 40, 50
        mask = np.zeros((H, W), np.uint8); mask[10:20, 10:30] = 1
        s = make_state(mask=mask, memory=[SimpleNamespace(reliability=0.8)])
        s.probs[:, :5, :] = 1 / 12.0
        rec = lambda st: SimpleNamespace(mask=st.mask.copy())
        ctx = ActionContext(state=s, evidence=ev(H, W), right_label=np.ones((H, W), np.uint8),
                            right_conf=np.full((H, W), 255, np.uint8), target_names=NAMES,
                            instrument_ids=INSTR, recovery=rec)
        masks = apply_all(ctx)
        for k, m in masks.items():
            assert m is not None, k
            f = action_features(ctx, m)
            assert f.shape == (len(ACTION_FEATURE_NAMES),) and np.isfinite(f).all()
        assert not hasattr(ctx, "ground_truth")

    @check("the ActionContext dataclass has no ground-truth field at all")
    def _():
        assert "ground_truth" not in ActionContext.__dataclass_fields__
        assert "gt" not in ActionContext.__dataclass_fields__

    print("\nLabels (training-data construction only)")

    @check("delta_scores: an action that fixes the mask scores positive, one that breaks it negative")
    def _():
        gt = np.zeros((20, 20), np.uint8); gt[5:15, 5:15] = 1
        bad = np.zeros_like(gt)
        good = gt.copy()
        d = delta_scores(bad, good, gt, 12, INSTR)
        assert d["instr"] > 0.5 and d["official"] > 0.5
        d2 = delta_scores(good, bad, gt, 12, INSTR)
        assert d2["instr"] < -0.5

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("Action invariants hold; ground truth cannot reach inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
