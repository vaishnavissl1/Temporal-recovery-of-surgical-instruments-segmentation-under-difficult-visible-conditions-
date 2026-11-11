#!/usr/bin/env python3
"""
TASK D.2 -- does the cross-view evidence improve the reliability gate?

    python -m tools.compare_crossview --stage oof  --oof work_v2/cache_oof_xv --stereo-oof work_v2/cache_oof_stereo \
        --folds work_v2/folds.json --out results/taskD
    python -m tools.compare_crossview --stage test --oof work_v2/cache_oof_xv --stereo-oof work_v2/cache_oof_stereo \
        --test work_v2/cache_test_xv --stereo-test work_v2/cache_test_stereo --folds work_v2/folds.json --out results/taskD

Exactly four indicator sets are compared, each fit with the existing folds in work_v2/folds.json (a gate for a held-out
fold is trained on the other folds' out-of-fold predictions and run under self-predicted history):

    1. paper4              the four original confidence/temporal indicators
    2. paper4_stereo3      + the three disparity indicators      (known-negative control)
    3. paper4_crossview6   + the six cross-view indicators       (the new evidence)
    4. crossview6          the six alone

DECISION RULE, fixed in this file before any number was seen. Cross-view counts as working if, out of fold,
`paper4_crossview6` beats `paper4` by AT LEAST +0.02 AUROC with a sequence-clustered 95% interval that EXCLUDES ZERO
(AUROC difference of the full out-of-fold sample >= 0.02 and lower interval end > 0; 2000 bootstrap draws that resample
whole sequences, seed 0; 15 training sequences).
  PASS -> continue; the cross-view gate feeds Task C row 7.
  FAIL -> stop and write up the negative result. Do not search for a variant that passes and do not drop indicators until
          something clears the bar. Record which of the six carried weight and what the interval was.

Fixed configuration (the one tools/compare_stereo.py used for the committed stereo negative, so the control reproduces):
gradient-boosted gate, memory K=5, tau_train=0.5, bad frame = INSTRUMENT IoU < 0.5 (the gate's own target and its standing
label). The same scores are also scored against the Task C risk (official frame IoU) and reported as secondary columns; the
decision rests on the primary only.

The test set is scored ONCE, in --stage test, which refuses to run unless the out-of-fold decision file already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr import reliability as REL  # noqa: E402
from sitr.metrics import aurc, aurc_oracle, pearson, roc_auc  # noqa: E402
from sitr.reliability import indicator_names  # noqa: E402
from tools.compare_stereo import boot_diff, cv_run, load_dir, test_run  # noqa: E402

SETS = ["paper4", "paper4_stereo3", "paper4_crossview6", "crossview6"]
THETA, K, TAU_TRAIN, FAMILY = 0.5, 5, 0.5, "gbm"
MIN_GAIN, N_BOOT = 0.02, 2000
PAIRS = [("paper4_crossview6", "paper4"), ("paper4_stereo3", "paper4"), ("crossview6", "paper4"),
         ("paper4_crossview6", "paper4_stereo3")]


def merge_stereo(xv_caches, stereo_caches):
    """Attach the disparity indicators (sd, oc, dc) to the cross-view caches, verifying the frames are the same ones."""
    for k, c in xv_caches.items():
        s = stereo_caches[k]
        if len(c) != len(s) or not np.array_equal(c.q, s.q) or not np.array_equal(c.masks, s.masks):
            raise SystemExit(f"{k}: the cross-view cache and the stereo cache are not the same frames")
        c.sd, c.oc, c.dc = s.sd, s.oc, s.dc
    return xv_caches


def metrics(r_hat, r_star, theta=THETA):
    bad = (r_star < theta).astype(int)
    return dict(auroc=roc_auc(-r_hat, bad), aurc=aurc(r_hat, 1.0 - r_star), aurc_oracle=aurc_oracle(1.0 - r_star),
                aurc_random=float(np.mean(1.0 - r_star)), pearson=pearson(r_hat, r_star),
                mse=float(np.mean((r_hat - r_star) ** 2)), bad_rate=float(bad.mean()), n=int(len(bad)))


def diffs(preds, r_star, seq, n_boot=N_BOOT):
    out = {}
    for x, y in PAIRS:
        d = boot_diff(preds[x], preds[y], r_star, seq, THETA, n_boot)
        d["auroc_full_sample_diff"] = roc_auc(-preds[x], (r_star < THETA).astype(int)) - roc_auc(-preds[y], (r_star < THETA).astype(int))
        out[f"{x} vs {y}"] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["oof", "test"])
    ap.add_argument("--oof", required=True)
    ap.add_argument("--stereo-oof", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--stereo-test", default=None)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", default="results/taskD")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    folds = json.load(open(a.folds))["folds"]
    kw = {"seed": 0}
    oof = merge_stereo(load_dir(a.oof), load_dir(a.stereo_oof))
    dec_path = os.path.join(a.out, "decision.json")
    t0 = time.time()

    if a.stage == "oof":
        print(f"OUT-OF-FOLD: {len(oof)} sequences, {sum(len(c) for c in oof.values())} frames, gbm, K={K}")
        preds, r_star, seq = {}, None, None
        for s in SETS:
            rh, rs, sq = cv_run(oof, folds, FAMILY, s, K, TAU_TRAIN, THETA, kw)
            preds[s] = rh
            r_star, seq = rs, sq
            m = metrics(rh, rs)
            print(f"  {s:20s} AUROC {m['auroc']:.4f}  AURC {m['aurc']:.4f}", flush=True)
        R = {"family": FAMILY, "K": K, "tau_train": TAU_TRAIN, "theta": THETA,
             "label": "bad frame = instrument IoU < 0.5 (the gate's standing label)", "n_oof_frames": int(len(r_star)),
             "sets": {s: list(indicator_names(s)) for s in SETS}, "cv": {s: metrics(preds[s], r_star) for s in SETS}}
        R["cv_paired"] = diffs(preds, r_star, seq)
        # leave-one-indicator-out on the 10-indicator set (the one the decision rule is about)
        full = R["cv"]["paper4_crossview6"]["auroc"]
        names = indicator_names("paper4_crossview6")
        R["loo_paper4_crossview6"] = {}
        for n in names:
            key = f"loo_{n}"
            REL.INDICATOR_SETS[key] = tuple(x for x in names if x != n)
            rh, rs, _ = cv_run(oof, folds, FAMILY, key, K, TAU_TRAIN, THETA, kw)
            m = metrics(rh, rs)
            R["loo_paper4_crossview6"][n] = dict(auroc_without=m["auroc"], auroc_drop=full - m["auroc"], aurc_without=m["aurc"])
            print(f"  without {n:20s} AUROC {m['auroc']:.4f}  (drop {full - m['auroc']:+.4f})", flush=True)
            del REL.INDICATOR_SETS[key]
        # leave-one-indicator-out on the paper4 baseline's own indicators, for scale
        # decision
        d = R["cv_paired"]["paper4_crossview6 vs paper4"]
        gain, lo, hi = d["auroc_full_sample_diff"], d["auc"]["ci95"][0], d["auc"]["ci95"][1]
        passed = bool(gain >= MIN_GAIN and lo > 0.0)
        R["decision"] = dict(rule=f"paper4_crossview6 beats paper4 by >= {MIN_GAIN} AUROC with a sequence-clustered 95% interval excluding zero",
                             auroc_gain_full_sample=gain, bootstrap_mean=d["auc"]["mean"], ci95=[lo, hi],
                             passed=passed, outcome="PASS: continue and feed Task C row 7" if passed else
                             "FAIL: stop; write up the negative result; do not search for a variant")
        print(f"\nDECISION: gain {gain:+.4f}  interval [{lo:+.4f}, {hi:+.4f}]  ->  {R['decision']['outcome']}")
        np.savez_compressed(os.path.join(a.out, "gate_scores_oof.npz"), seq=seq, r_star=r_star, **{s: preds[s] for s in SETS})
        json.dump(R, open(dec_path, "w"), indent=1)
        json.dump({"stage": "oof", "seconds": time.time() - t0, **R}, open(os.path.join(a.out, "gate_comparison_oof.json"), "w"), indent=1)
        print(f"wrote {dec_path} ({time.time()-t0:.0f}s)")
        return 0

    # ---------------------------------------------------------------- test, once
    if not os.path.exists(dec_path):
        raise SystemExit("the out-of-fold decision file does not exist; run --stage oof first. The test set is scored once, after it.")
    if os.path.exists(os.path.join(a.out, "gate_comparison_test.json")):
        raise SystemExit("the test set has already been scored for this comparison (gate_comparison_test.json exists); refusing to score it again")
    dec = json.load(open(dec_path))["decision"]
    test = merge_stereo({k: v for k, v in load_dir(a.test).items() if k.startswith("test/")},
                        {k: v for k, v in load_dir(a.stereo_test).items() if k.startswith("test/")})
    print(f"TEST: {len(test)} sequences; the out-of-fold decision was: {dec['outcome']}")
    tp = {s: test_run(oof, test, FAMILY, s, K, TAU_TRAIN, THETA, kw) for s in SETS}
    r_star, seq = tp[SETS[0]][1], tp[SETS[0]][2]
    preds = {s: tp[s][0] for s in SETS}
    R = {"n_test_frames": int(len(r_star)), "oof_decision": dec, "test": {s: metrics(preds[s], r_star) for s in SETS},
         "test_paired": diffs(preds, r_star, seq),
         "note": "only 4 test sequences: the sequence bootstrap interval is coarse; read it as a warning, not a test"}
    for s in SETS:
        print(f"  {s:20s} AUROC {R['test'][s]['auroc']:.4f}  AURC {R['test'][s]['aurc']:.4f}")
    np.savez_compressed(os.path.join(a.out, "gate_scores_test.npz"), seq=seq, r_star=r_star, **{s: preds[s] for s in SETS})
    json.dump({"stage": "test", "seconds": time.time() - t0, **R}, open(os.path.join(a.out, "gate_comparison_test.json"), "w"), indent=1)
    print(f"wrote {a.out}/gate_comparison_test.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
