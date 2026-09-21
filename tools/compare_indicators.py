#!/usr/bin/env python3
"""
Is the extended indicator set ACTUALLY better than the report's four?

A single training run cannot answer this. The estimator is initialised randomly
and trained on ~1600 frames, so run-to-run variation is easily as large as the
difference being measured. Comparing one run against one run and reporting the
winner is how a null result gets published as a positive one.

This repeats both configurations over several seeds and reports mean +- std, so
the comparison is either significant or it is not.

    python -m tools.compare_indicators --cache work/cache --val-seqs 2 5 9 15 \
        --seeds 5 --out work/indicator_ablation

Interpretation rule, stated before you see the numbers: if the difference in
mean AUC is smaller than the pooled standard deviation, the sets are tied and
the honest write-up says so. A tie is a legitimate finding -- it is exactly the
confidence-only ablation the report proposes in Sec. V-J.4, answered with
evidence instead of assertion.

Two metrics are tested, because they answer different questions:

    AUC (primary)   can the gate RANK frames -- does it flag the ones the base
                    model actually got wrong?  This is what the gate is for.
    MSE (secondary) is r_t a calibrated estimate of the quality VALUE?  This
                    matters only if Stage 3 weights by r_t rather than merely
                    thresholding it.

They can disagree, and a split decision is reported as a split decision rather
than resolved in favour of whichever one looks better.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(args, indicators: str, seed: int, out_dir: str) -> dict:
    cmd = [sys.executable, "-m", "tools.train_stage2",
           "--cache", args.cache,
           "--val-seqs", *map(str, args.val_seqs),
           "--indicators", indicators,
           "--memory-size", str(args.memory_size),
           "--tau-train", str(args.tau_train),
           "--epochs", str(args.epochs),
           "--seed", str(seed),
           "--out", out_dir, "--no-plots"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        raise SystemExit(f"{indicators} seed={seed} failed")
    with open(os.path.join(out_dir, "stage2_report.json")) as f:
        return json.load(f)


def summarise(vals):
    if len(vals) < 2:
        return (vals[0] if vals else float("nan")), 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="runs/indicator_ablation")
    ap.add_argument("--val-seqs", nargs="+", type=int, default=[2, 5, 9, 15])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--memory-size", type=int, default=5)
    ap.add_argument("--tau-train", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = {"paper4": [], "extended": []}
    t0 = time.time()

    print(f"{args.seeds} seeds x 2 indicator sets = {args.seeds*2} runs\n")
    print(f"{'set':<10}{'seed':>5}{'AUC':>9}{'pearson':>9}{'MSE':>10}{'rho':>8}")
    print("-" * 51)
    for ind in ("paper4", "extended"):
        for seed in range(args.seeds):
            d = os.path.join(args.out, "_runs", f"{ind}_s{seed}")
            rep = run_one(args, ind, seed, d)
            v = rep["val_self_predicted_history"]
            tau = rep["threshold"]["tau"]
            rho = next((r["rho"] for r in rep["threshold"]["grid"]
                        if abs(r["tau"] - tau) < 1e-9), float("nan"))
            row = {"seed": seed, "auc": v["auc_detect_bad_frames"],
                   "pearson": v["pearson_r"], "mse": v["mse"],
                   "tau": tau, "rho": rho}
            results[ind].append(row)
            print(f"{ind:<10}{seed:>5}{row['auc']:>9.4f}{row['pearson']:>9.4f}"
                  f"{row['mse']:>10.5f}{rho:>8.3f}")

    print("-" * 51)
    summary = {}
    for ind in ("paper4", "extended"):
        a_m, a_s = summarise([r["auc"] for r in results[ind]])
        p_m, p_s = summarise([r["pearson"] for r in results[ind]])
        m_m, m_s = summarise([r["mse"] for r in results[ind]])
        summary[ind] = {"auc_mean": a_m, "auc_std": a_s,
                        "pearson_mean": p_m, "pearson_std": p_s,
                        "mse_mean": m_m, "mse_std": m_s, "runs": results[ind]}
        print(f"{ind:<10}  AUC {a_m:.4f} +- {a_s:.4f}   "
              f"r {p_m:.4f} +- {p_s:.4f}   MSE {m_m:.5f} +- {m_s:.5f}")

    def compare(key, lower_is_better=False):
        d = summary["extended"][f"{key}_mean"] - summary["paper4"][f"{key}_mean"]
        pooled = max((summary["extended"][f"{key}_std"] ** 2
                      + summary["paper4"][f"{key}_std"] ** 2) ** 0.5, 1e-9)
        if abs(d) < pooled:
            win = None
        elif lower_is_better:
            win = "extended" if d < 0 else "paper4"
        else:
            win = "extended" if d > 0 else "paper4"
        return d, pooled, win

    d_auc, pooled, win_auc = compare("auc")
    d_mse, pooled_mse, win_mse = compare("mse", lower_is_better=True)

    print("\n" + "=" * 51)
    print(f"AUC  (primary)  : {d_auc:+.4f}   pooled std {pooled:.4f}   "
          f"-> {win_auc or 'tied'}")
    print(f"MSE  (secondary): {d_mse:+.5f}   pooled std {pooled_mse:.5f}   "
          f"-> {win_mse or 'tied'}")

    if win_auc is None:
        verdict = ("TIED ON AUC — the difference is smaller than run-to-run "
                   "variation. On the criterion that matters for the gate "
                   "(does it flag the frames the base model got wrong), the "
                   "report's four indicators are sufficient on this dataset "
                   "and the extra five add no separable signal.")
    else:
        verdict = (f"{win_auc.upper()} WINS ON AUC — the difference exceeds "
                   f"run-to-run variation ({abs(d_auc):.4f} > {pooled:.4f}). "
                   "Report the mean +- std, not a single run.")

    # AUC ranks frames; MSE measures how well r_t is calibrated as a quality
    # estimate. Stage 3 consumes r_t as a number, not only as a ranking, so a
    # clean MSE separation is worth reporting even when AUC ties.
    if win_mse is not None and win_mse != win_auc:
        verdict += (f" Note the split decision: {win_mse} is better on MSE by "
                    f"more than run-to-run variation, so it predicts the "
                    "quality VALUE more accurately while ranking frames no "
                    "better. If Stage 3 uses r_t only to gate, this does not "
                    f"matter; if it weights by r_t, prefer {win_mse}.")
    print(verdict)

    summary["difference_auc"] = d_auc
    summary["pooled_std"] = pooled
    summary["difference_mse"] = d_mse
    summary["pooled_std_mse"] = pooled_mse
    summary["winner_auc"] = win_auc
    summary["winner_mse"] = win_mse
    summary["verdict"] = verdict
    summary["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "indicator_ablation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'indicator_ablation.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
