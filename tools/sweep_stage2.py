#!/usr/bin/env python3
"""
Select the Stage 2 hyperparameters FROM THE VALIDATION DATA, not by assertion.

The report specifies both of these as validation-set hyperparameters, not
constants (Sec. IV-C: "The threshold tau is a validation set hyperparameter
rather than a fixed constant"; Sec. IV-D: "The memory size K is a validation
set hyperparameter"). Leaving them at a default therefore does not implement
the method as written -- it implements a guess at it.

This sweeps them jointly on the held-out validation sequences:

    K         memory size, eq. (15)
    tau_train quality level at which a past frame counts as reliable enough to
              build history from (distinct from the deployed decision threshold
              tau, which each run then sweeps separately)

Selection criterion, stated up front so it cannot be chosen after the fact:
the gate's job is to flag frames the base model actually got wrong, so the
primary criterion is AUC for detecting r* < theta on held-out sequences under
self-predicted history, with validation MSE as the tie-break.

Because Stage 1 is frozen and its outputs are cached, each configuration costs
under a minute -- the whole grid is ~10 minutes and needs no GPU.

    python -m tools.sweep_stage2 --cache work/cache --out work/stage2_sweep \
        --val-seqs 2 5 9 15

The winning configuration is written to <out>/ as a normal Stage 2 result, so
it drops straight into the release folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(args, K: int, tau_train: float, out_dir: str) -> dict:
    cmd = [sys.executable, "-m", "tools.train_stage2",
           "--cache", args.cache,
           "--val-seqs", *map(str, args.val_seqs),
           "--indicators", args.indicators,
           "--memory-size", str(K),
           "--tau-train", f"{tau_train:.3f}",
           "--theta", str(args.theta),
           "--epochs", str(args.epochs),
           "--align", args.align,
           "--out", out_dir, "--no-plots"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"configuration K={K} tau_train={tau_train} failed")
    with open(os.path.join(out_dir, "stage2_report.json")) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="runs/stage2_sweep")
    ap.add_argument("--val-seqs", nargs="+", type=int, default=[2, 5, 9, 15])
    ap.add_argument("--indicators", default="extended",
                    choices=["paper4", "extended"])
    ap.add_argument("--memory-sizes", nargs="+", type=int,
                    default=[2, 3, 5, 8, 12])
    ap.add_argument("--tau-trains", nargs="+", type=float,
                    default=[0.4, 0.5, 0.6])
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--align", default="identity",
                    choices=["identity", "farneback"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "_configs")
    os.makedirs(work, exist_ok=True)

    grid = [(K, t) for K in args.memory_sizes for t in args.tau_trains]
    print(f"sweeping {len(grid)} configurations "
          f"(K in {args.memory_sizes}, tau_train in {args.tau_trains})")
    print("criterion: AUC for detecting r* < theta on held-out sequences "
          "under self-predicted history; MSE breaks ties\n")
    print(f"{'K':>4} {'tau_tr':>7} {'AUC':>7} {'MSE':>9} {'pearson':>8} "
          f"{'tau*':>6} {'rho':>7} {'cold':>6}")
    print("-" * 60)

    rows, t0 = [], time.time()
    for K, tt in grid:
        d = os.path.join(work, f"K{K}_t{tt:.2f}")
        rep = run_one(args, K, tt, d)
        v = rep["val_self_predicted_history"]
        tau = rep["threshold"]["tau"]
        rho = next((r["rho"] for r in rep["threshold"]["grid"]
                    if abs(r["tau"] - tau) < 1e-9), float("nan"))
        row = {"memory_size": K, "tau_train": tt, "dir": d,
               "auc": v.get("auc_detect_bad_frames", float("nan")),
               "mse": v.get("mse", float("nan")),
               "pearson_r": v.get("pearson_r", float("nan")),
               "tau": tau, "rho": rho,
               "cold_start_rate": rep.get("frac_cold_start_frames", float("nan"))}
        rows.append(row)
        print(f"{K:>4} {tt:>7.2f} {row['auc']:>7.4f} {row['mse']:>9.5f} "
              f"{row['pearson_r']:>8.4f} {tau:>6.2f} {rho:>7.3f} "
              f"{row['cold_start_rate']:>6.3f}")

    finite = [r for r in rows if r["auc"] == r["auc"]]          # drop NaN AUC
    if not finite:
        raise SystemExit("every configuration produced a NaN AUC — usually "
                         "means no validation frame fell below theta")
    best = max(finite, key=lambda r: (round(r["auc"], 4), -r["mse"]))

    print("-" * 60)
    print(f"selected: K={best['memory_size']}  tau_train={best['tau_train']:.2f}"
          f"  -> AUC={best['auc']:.4f}  MSE={best['mse']:.5f}  "
          f"deployed tau={best['tau']:.2f} (routes {best['rho']:.1%})")

    # A flat grid is itself a finding: it means the choice does not matter much,
    # which is worth one sentence in the paper rather than a silent pick.
    spread = max(r["auc"] for r in finite) - min(r["auc"] for r in finite)
    if spread < 0.01:
        print(f"note: AUC varies by only {spread:.4f} across the whole grid — "
              "the gate is insensitive to K and tau_train on this data. Say so.")

    for fn in os.listdir(best["dir"]):
        shutil.copy2(os.path.join(best["dir"], fn), os.path.join(args.out, fn))
    with open(os.path.join(args.out, "hyperparameter_sweep.json"), "w") as f:
        json.dump({"criterion": "auc_detect_bad_frames, tie-break -mse",
                   "theta": args.theta, "indicators": args.indicators,
                   "val_seqs": args.val_seqs, "grid": rows,
                   "selected": best, "auc_spread": spread,
                   "seconds": round(time.time() - t0, 1)}, f, indent=2)

    print(f"\nwinning run copied to {args.out}")
    print(f"full grid: {os.path.join(args.out, 'hyperparameter_sweep.json')}")
    print(f"took {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
