#!/usr/bin/env python3
"""
Task A: compare the temporal action A2 (and everything that contains it) across the three alignment
operators, side by side, with sequence-clustered intervals.

    python -m tools.compare_aligners --release work_v2/release --out results/taskA \
        identity=work_v2/actions farneback=work_v2/actions_farneback region=work_v2/actions_region

Per aligner and per group (oof = 15 training sequences with their fold model; test = the 4 official
test sequences) it reports, for the policy "apply the action on every FLAGGED frame, keep the rest":
  * mean change in instrument IoU on flagged frames and over all frames
  * % improved / % worse / % unchanged  (a change within +-margin_eval counts as unchanged; stated choice,
    the same 0.01 used by tools/train_selector.py and tools/measure_recovery.py), plus the strict-zero share
  * the ORACLE ceiling: per flagged frame the best of {keep, A1, A2, A3, FUSE}, with both denominators
Flagged = reliability r < tau and not degenerate (exactly the selector's routing rule).

Stop rule, fixed before any number was seen. The decision uses OUT-OF-FOLD data only. For each non-identity
aligner the paired difference to identity in (a) mean A2 change on flagged frames and (b) oracle mean change
on flagged frames is estimated with a cluster bootstrap over the 15 training sequences (resample whole
sequences, 2000 draws, seed 0). If every one of those intervals contains 0, the identity aligner was not the
cause. The test sequences (4) are reported once, after that decision, with the same intervals.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.train_selector import load_group, stack  # noqa: E402

CAND = ("A1", "A2", "A3", "FUSE")


def per_frame(D, tau, margin):
    routed = (D["r"] < tau) & ~D["degenerate"].astype(bool)
    dm = np.stack([np.where(D[f"avail_{k}"], D[f"d_{k}"], -np.inf) for k in CAND], 1)
    oracle = np.maximum(dm.max(1), 0.0)             # keep (0) is always available
    a2 = np.where(D["avail_A2"], D["d_A2"], 0.0)    # A2 unavailable (empty memory) -> frame left as is
    fuse = np.where(D["avail_FUSE"], D["d_FUSE"], 0.0)
    return routed, {"A2": np.where(routed, a2, 0.0), "FUSE": np.where(routed, fuse, 0.0),
                    "oracle": np.where(routed, oracle, 0.0)}, D["avail_A2"] & routed


def summarise(routed, v, avail, margin):
    n, nr = len(routed), int(routed.sum())
    c = avail if avail is not None else routed
    return dict(n_frames=n, n_flagged=nr, flagged_pct=100 * nr / n,
                mean_on_flagged=float(v[routed].mean()), mean_all_frames=float(v.mean()),
                n_applied=int(c.sum()),
                pct_improved_of_flagged=100 * float((v[routed] > margin).mean()),
                pct_worse_of_flagged=100 * float((v[routed] < -margin).mean()),
                pct_unchanged_of_flagged=100 * float((np.abs(v[routed]) <= margin).mean()),
                pct_exactly_zero_of_flagged=100 * float((v[routed] == 0).mean()))


def cluster_ci(seq, routed, a, b, n_boot=2000, seed=0):
    """95% interval of mean_on_flagged(a) - mean_on_flagged(b), resampling whole sequences."""
    rng = np.random.default_rng(seed)
    names = sorted(set(seq))
    s_a = {s: (a[(seq == s) & routed].sum(), b[(seq == s) & routed].sum(), int(((seq == s) & routed).sum())) for s in names}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), len(names), replace=True)
        tot = np.array([s_a[names[i]] for i in pick], dtype=float).sum(0)
        out.append((tot[0] - tot[1]) / max(tot[2], 1))
    out = np.array(out)
    full = (a[routed].sum() - b[routed].sum()) / max(int(routed.sum()), 1)
    return float(full), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-eval", type=float, default=0.01)
    ap.add_argument("dirs", nargs="+", help="name=action_dir, first one is the reference (identity)")
    a = ap.parse_args()
    tau = float(json.load(open(os.path.join(a.release, "manifest.json")))["tau"])
    spec = [d.split("=", 1) for d in a.dirs]
    os.makedirs(a.out, exist_ok=True)
    res = {"tau": tau, "margin_eval": a.margin_eval, "reference": spec[0][0], "groups": {}}
    for group in ("oof", "test"):
        data = {}
        for name, d in spec:
            S = load_group(d, group)
            data[name] = stack(S, sorted(S))
        ref = data[spec[0][0]]
        for name, D in data.items():       # the aligner must not change anything but the temporal candidate
            assert np.array_equal(D["seq"], ref["seq"]) and np.array_equal(D["t"], ref["t"]), (group, name, "frame order")
            assert np.allclose(D["r"], ref["r"]), (group, name, "reliability differs")
        g = {}
        pf = {}
        for name, D in data.items():
            routed, vals, avail = per_frame(D, tau, a.margin_eval)
            pf[name] = (routed, vals)
            g[name] = {k: summarise(routed, v, avail if k == "A2" else None, a.margin_eval) for k, v in vals.items()}
            g[name]["A2_availability_pct_of_flagged"] = 100 * float(avail.sum() / routed.sum())
        seq = ref["seq"]
        diffs = {}
        r0, v0 = pf[spec[0][0]]
        for name in list(data)[1:]:
            r1, v1 = pf[name]
            assert np.array_equal(r0, r1)
            diffs[name] = {k: dict(zip(("diff", "ci_lo", "ci_hi"), cluster_ci(seq, r0, v1[k], v0[k]))) for k in ("A2", "FUSE", "oracle")}
        res["groups"][group] = {"policies": g, "paired_difference_vs_reference": diffs, "n_sequences": len(set(seq))}

    # decision, out of fold only
    dec = {}
    for name, dd in res["groups"]["oof"]["paired_difference_vs_reference"].items():
        dec[name] = {k: (dd[k]["ci_lo"] <= 0.0 <= dd[k]["ci_hi"]) for k in ("A2", "oracle")}
    res["decision_oof"] = {"interval_contains_zero": dec,
                           "identity_was_not_the_cause": bool(all(all(v.values()) for v in dec.values()))}
    json.dump(res, open(os.path.join(a.out, "aligner_comparison.json"), "w"), indent=1)

    for group in ("oof", "test"):
        G = res["groups"][group]
        print(f"\n=== {group.upper()}  ({G['n_sequences']} sequences) -- policy: apply on FLAGGED frames, keep the rest")
        print(f"{'aligner':<10}{'action':<8}{'flagged':>9}{'dIoU flagged':>14}{'dIoU all':>10}{'improved%':>11}{'worse%':>8}{'unchanged%':>12}{'exact0%':>9}")
        for name, P in G["policies"].items():
            for k in ("A2", "FUSE", "oracle"):
                s = P[k]
                print(f"{name:<10}{k:<8}{s['n_flagged']:>9}{s['mean_on_flagged']:>+14.4f}{s['mean_all_frames']:>+10.4f}"
                      f"{s['pct_improved_of_flagged']:>11.1f}{s['pct_worse_of_flagged']:>8.1f}{s['pct_unchanged_of_flagged']:>12.1f}{s['pct_exactly_zero_of_flagged']:>9.1f}")
        for name, dd in G["paired_difference_vs_reference"].items():
            for k, v in dd.items():
                print(f"  {name} - {res['reference']} [{k}] on flagged: {v['diff']:+.4f}  95% (sequence-clustered) [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")
    print("\nDecision (out of fold):", json.dumps(res["decision_oof"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
