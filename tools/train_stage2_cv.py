#!/usr/bin/env python3
"""
Stage 2 v2 -- reliability estimator trained on out-of-fold errors, selected by
cross-validation, with a risk-controlled routing threshold, and tested on the
official EndoVis 2018 test sequences.

Inputs (both produced by tools/run_v2.py):
  --oof-cache   caches of the 15 training sequences, each predicted by a Stage 1
                model that NEVER saw it (cross-fitting)
  --test-cache  caches of the 4 official test sequences, predicted by the final
                Stage 1 model (trained on all 15)
  --folds       folds.json from the same run (Stage 2 CV reuses Stage 1's folds)

Selection criterion, fixed before any number is seen:
  maximise leave-fold-out AUC for detecting bad frames (r* < theta) under
  self-predicted history; configurations within 0.005 AUC of the best are
  treated as tied and the lowest MSE among them wins.

What is chosen by data here:   meta-model family, memory size K, tau_train
What is a stated requirement:   alpha (risk budget), theta (bad-frame IoU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.reliability import (SequenceCache, StreamConfig,  # noqa: E402
                              indicator_names)
from sitr.stage2v2 import (FAMILIES, collect, conformal_tau,  # noqa: E402
                           cross_validate, diagnostics, make_folds, realised_risk)
from sitr.utils import save_checkpoint, write_json  # noqa: E402

TIE_AUC = 0.005


def load_dir(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            c = SequenceCache.load(os.path.join(d, f))
            out[c.name] = c
    if not out:
        raise FileNotFoundError(f"no caches in {d}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof-cache", required=True)
    ap.add_argument("--test-cache", default=None)
    ap.add_argument("--folds", default=None, help="folds.json; else built here")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--families", nargs="+", default=["gbm", "mlp"],
                    choices=list(FAMILIES))
    ap.add_argument("--memory-sizes", nargs="+", type=int, default=[3, 5, 8])
    ap.add_argument("--tau-trains", nargs="+", type=float, default=[0.5])
    ap.add_argument("--indicators", default="extended",
                    choices=["paper4", "extended"])
    ap.add_argument("--theta", type=float, default=0.5,
                    help="a frame is BAD if its instrument IoU is below this")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="risk budget: at most this fraction of all frames may "
                         "be bad frames that are accepted without repair")
    ap.add_argument("--mlp-epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    oof = load_dir(args.oof_cache)
    for k in oof:
        if not k.startswith("train/"):
            raise SystemExit(f"--oof-cache must contain only training sequences, "
                             f"found {k}")
    if args.folds:
        with open(args.folds) as f:
            folds = json.load(f)["folds"]
    else:
        folds = make_folds([(k, len(c)) for k, c in oof.items()], args.k)
    flat = [k for f in folds for k in f]
    if sorted(flat) != sorted(oof) or len(set(flat)) != len(flat):
        raise SystemExit("folds do not partition the OOF sequences exactly")
    print(f"{len(oof)} OOF sequences in {len(folds)} folds: "
          + "  ".join(f"[{','.join(s.split('/')[-1] for s in f)}]" for f in folds))

    for c in oof.values():
        c.require(args.indicators)

    # ------------------------------------------------------------ selection
    rows = []
    print("\nleave-fold-out CV (self-predicted history)")
    for fam in args.families:
        for K in args.memory_sizes:
            for tt in args.tau_trains:
                cfg = StreamConfig(memory_size=K, tau=tt, indicators=args.indicators)
                kw = {"seed": args.seed}
                if fam == "mlp":
                    kw["epochs"] = args.mlp_epochs
                print(f"  {fam}  K={K}  tau_train={tt}")
                res = cross_validate(oof, folds, fam, cfg, args.theta, kw,
                                     log=lambda s: print(s))
                d = res["diagnostics"]
                rows.append({"family": fam, "memory_size": K, "tau_train": tt,
                             **{k: d[k] for k in ("auc_detect_bad_frames", "mse",
                                                  "pearson_r", "bad_frame_rate")},
                             "_r_hat": res["r_hat"], "_r_star": res["r_star"]})
                print(f"    pooled: AUC={d['auc_detect_bad_frames']:.4f}  "
                      f"MSE={d['mse']:.5f}  r={d['pearson_r']:.4f}")

    finite = [r for r in rows if np.isfinite(r["auc_detect_bad_frames"])]
    if finite:
        best_auc = max(r["auc_detect_bad_frames"] for r in finite)
        tied = [r for r in finite if r["auc_detect_bad_frames"] >= best_auc - TIE_AUC]
    else:
        # AUC needs both bad and good frames. If the OOF data has only one kind,
        # Stage 1 is either near-perfect or broken -- either way the ranking
        # criterion is undefined, so say so loudly and fall back to MSE.
        print("\n!! AUC is undefined for every configuration: the out-of-fold "
              "frames are all bad or all good (r* on one side of theta). "
              "Check the Stage 1 results before trusting Stage 2. Selecting "
              "by MSE only.")
        tied = [r for r in rows if np.isfinite(r["mse"])] or rows
    sel = min(tied, key=lambda r: r["mse"])
    print(f"\nselected: {sel['family']}  K={sel['memory_size']}  "
          f"tau_train={sel['tau_train']}  (CV AUC {sel['auc_detect_bad_frames']:.4f}, "
          f"MSE {sel['mse']:.5f}; {len(tied)} config(s) within {TIE_AUC} AUC)")

    # --------------------------------------------------------- threshold
    ct = conformal_tau(sel["_r_hat"], sel["_r_star"], args.theta, args.alpha)
    alpha_table = []
    for a in sorted({0.02, 0.05, 0.10, 0.15, args.alpha}):
        c = conformal_tau(sel["_r_hat"], sel["_r_star"], args.theta, a)
        alpha_table.append({"alpha": a, "tau": c["tau"], "rho": c["rho"],
                            "feasible": c["feasible"]})
    print(f"\nconformal risk control (alpha={args.alpha}): tau={ct['tau']:.3f}  "
          f"routes {ct['rho']:.1%} of frames"
          + ("" if ct["feasible"] else "   !! alpha unattainable: routing all"))
    print("  risk budget -> threshold -> routing rate:")
    for r in alpha_table:
        print(f"    alpha={r['alpha']:.2f}  tau={r['tau']:.3f}  rho={r['rho']:.3f}")

    # ------------------------------------------------------------- final fit
    cfg = StreamConfig(memory_size=sel["memory_size"], tau=sel["tau_train"],
                       indicators=args.indicators)
    kw = {"seed": args.seed}
    if sel["family"] == "mlp":
        kw["epochs"] = args.mlp_epochs
    fam = FAMILIES[sel["family"]](**kw).fit(list(oof.values()), cfg)
    names = list(indicator_names(args.indicators))
    exp = fam.export()
    meta = {"kind": exp["kind"], "feature_names": names,
            "args": {"hidden": exp.get("hidden", 64),
                     "dropout": exp.get("dropout", 0.0),
                     "memory_size": sel["memory_size"],
                     "tau_train": sel["tau_train"], "indicators": args.indicators},
            "tau": ct["tau"], "alpha": args.alpha, "theta": args.theta}
    rel_path = os.path.join(args.out, "reliability_best.pt")
    if exp["kind"] == "sklearn":
        torch.save({"model": None, "meta": meta, "estimator": exp["estimator"]},
                   rel_path)
    else:
        torch.save({"model": exp["state_dict"], "meta": meta}, rel_path)

    # ------------------------------------------------------------ test set
    test_block = None
    if args.test_cache:
        test = load_dir(args.test_cache)
        tst = [c for k, c in test.items() if k.startswith("test/")]
        d = collect(tst, cfg, fam.predict_fn(), 1.0, 7)
        td = diagnostics(d["r_hat"], d["r_star"], args.theta)
        rr = realised_risk(d["r_hat"], d["r_star"], ct["tau"], args.theta)
        test_block = {**td, **rr, "sequences": sorted(test)}
        print(f"\nOFFICIAL TEST SEQUENCES ({len(tst)}): AUC={td['auc_detect_bad_frames']:.4f}  "
              f"r={td['pearson_r']:.4f}  MSE={td['mse']:.5f}")
        held = rr["risk_bad_accepted"] <= args.alpha
        print(f"  routed {rr['rho']:.1%}   bad frames caught {rr['bad_frames_caught']:.1%}"
              f"   good frames routed {rr['good_frames_routed']:.1%}")
        print(f"  bad-and-accepted = {rr['risk_bad_accepted']:.3f} vs budget "
              f"{args.alpha}  -> {'HELD' if held else 'EXCEEDED (report this)'}")

    # ---------------------------------------------------------------- report
    report = {
        "version": 2, "args": {**vars(args), "memory_size": sel["memory_size"],
                               "tau_train": sel["tau_train"]},
        "reported_checkpoint": "reliability_best.pt",
        "selection": {"criterion": f"max CV AUC; ties within {TIE_AUC} -> min MSE",
                      "selected": {k: v for k, v in sel.items() if not k.startswith("_")},
                      "grid": [{k: v for k, v in r.items() if not k.startswith("_")}
                               for r in rows]},
        "indicator_set": args.indicators, "indicator_names": names,
        # key kept for compatibility with the release manifest code
        "val_self_predicted_history": diagnostics(sel["_r_hat"], sel["_r_star"],
                                                  args.theta),
        "threshold": {**{k: v for k, v in ct.items() if k != "grid"},
                      "grid": ct["grid"], "alpha_table": alpha_table},
        "test_self_predicted_history": test_block,
        "routing_rate_rho_test": test_block["rho"] if test_block else None,
        "seconds": round(time.time() - t0, 1),
    }
    write_json(os.path.join(args.out, "stage2_report.json"), report)
    print(f"\nwrote {rel_path}\nwrote {os.path.join(args.out, 'stage2_report.json')}"
          f"  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
