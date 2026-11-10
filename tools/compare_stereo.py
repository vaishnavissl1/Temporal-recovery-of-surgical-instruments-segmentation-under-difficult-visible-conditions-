#!/usr/bin/env python3
"""
TASK 4, part C -- do the stereo indicators beat the temporal ones?

    python -m tools.compare_stereo --oof work_v2/cache_oof_stereo --folds work_v2/folds.json \
        [--test work_v2/cache_test_stereo] --out results/task4

Everything is cross-fitted: the gate for a held-out fold is trained on the other
folds' OUT-OF-FOLD predictions, and run under SELF-PREDICTED history (the
inference regime). The official test sequences, when given, are scored by a gate
fit on all training sequences.

Indicator sets compared (same family, same K, same tau_train for every set, so
only the indicators differ):
    paper4              the original four (confidence + temporal)
    extended            the 9 (the baseline in the brief)
    extended_stereo     9 + 3 stereo                       <- "9+3"
    nontemporal         confidence/shape only: NO temporal indicator at all
    nontemporal_stereo  nontemporal + 3 stereo   <- the fair "stereo replaces temporal?" test
    stereo3             the three stereo indicators alone

Why a bootstrap and not "several seeds": gradient-boosted trees without
subsampling are deterministic, so repeating the run with other seeds would show
ZERO variation and a false sense of precision. The real uncertainty is which
sequences happened to be sampled, so every difference is given with a paired
cluster bootstrap over SEQUENCES (frames in a sequence are correlated).

Decision rule, fixed before any number is seen: a set beats another only if the
95% bootstrap interval of the AUC difference excludes 0; otherwise they are TIED.
A measured tie or loss is reported as the result.
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
from sitr.metrics import aurc, aurc_oracle, pearson, roc_auc, spearman  # noqa: E402
from sitr.reliability import SequenceCache, StreamConfig, indicator_names  # noqa: E402
from sitr.stage2v2 import FAMILIES, collect, cross_validate  # noqa: E402

SETS = ["paper4", "extended", "extended_stereo", "nontemporal", "nontemporal_stereo", "stereo3"]


def load_dir(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            c = SequenceCache.load(os.path.join(d, f))
            out[c.name] = c
    return out


def metrics(r_hat, r_star, theta):
    bad = (r_star < theta).astype(int)
    return dict(auc=roc_auc(-r_hat, bad), aurc=aurc(r_hat, 1.0 - r_star),
                aurc_bad01=aurc(r_hat, bad.astype(float)),
                aurc_oracle=aurc_oracle(1.0 - r_star), aurc_random=float(np.mean(1.0 - r_star)),
                pearson=pearson(r_hat, r_star), spearman=spearman(r_hat, r_star),
                mse=float(np.mean((r_hat - r_star) ** 2)), bad_rate=float(bad.mean()), n=int(len(bad)))


def cv_run(oof, folds, family, setname, K, tau_train, theta, kw):
    cfg = StreamConfig(memory_size=K, tau=tau_train, indicators=setname)
    res = cross_validate(oof, folds, family, cfg, theta, kw, log=lambda s: None)
    seq = []
    for fold in folds:
        for k in fold:
            seq += [k] * len(oof[k])
    return res["r_hat"], res["r_star"], np.array(seq)


def test_run(oof, test, family, setname, K, tau_train, theta, kw):
    cfg = StreamConfig(memory_size=K, tau=tau_train, indicators=setname)
    fam = FAMILIES[family](**kw).fit(list(oof.values()), cfg)
    names = sorted(test)
    d = collect([test[k] for k in names], cfg, fam.predict_fn(), 1.0, 7)
    seq = np.array([k for k in names for _ in range(len(test[k]))])
    return d["r_hat"], d["r_star"], seq


def boot_diff(A, B, r_star, seq, theta, n_boot=2000, seed=0):
    """Paired cluster bootstrap over sequences of metric(A) - metric(B)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seq)
    idx = {s: np.where(seq == s)[0] for s in uniq}
    bad = (r_star < theta).astype(int)
    out = {"auc": [], "aurc": []}
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        if bad[ii].min() == bad[ii].max():
            continue
        out["auc"].append(roc_auc(-A[ii], bad[ii]) - roc_auc(-B[ii], bad[ii]))
        out["aurc"].append(aurc(A[ii], 1 - r_star[ii]) - aurc(B[ii], 1 - r_star[ii]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: dict(mean=float(np.mean(v)), ci95=ci(v)) for k, v in out.items()}


def verdict(d, name_a, name_b):
    lo, hi = d["auc"]["ci95"]
    if lo > 0:
        return f"{name_a} BEATS {name_b} (AUC difference interval excludes 0)"
    if hi < 0:
        return f"{name_a} LOSES to {name_b} (AUC difference interval excludes 0)"
    return f"{name_a} and {name_b} are TIED (AUC difference interval includes 0)"


def indicator_table(oof, K, tau_train, theta):
    cfg = StreamConfig(memory_size=K, tau=tau_train, indicators="extended_stereo")
    caches = [oof[k] for k in sorted(oof)]
    d = collect(caches, cfg, None, 0.0, 0)
    names = indicator_names("extended_stereo")
    seq = np.array([c.name for c in caches for _ in range(len(c))])
    y, bad = d["r_star"], (d["r_star"] < theta).astype(int)
    rows = []
    for j, n in enumerate(names):
        x = d["feats"][:, j]
        if x.std() < 1e-9:
            rows.append(dict(indicator=n, pearson=float("nan"))); continue
        # within-sequence: remove each sequence's own mean from BOTH sides
        xs, ys = x.copy(), y.copy()
        for s in np.unique(seq):
            m = seq == s
            xs[m] -= x[m].mean(); ys[m] -= y[m].mean()
        sm = [(x[seq == s].mean(), y[seq == s].mean()) for s in np.unique(seq)]
        a = roc_auc(x, 1 - bad)
        rows.append(dict(indicator=n, pearson=pearson(x, y), spearman=spearman(x, y),
                         auc_good_vs_bad=max(a, 1 - a), sign=("+" if pearson(x, y) >= 0 else "-"),
                         within_seq_pearson=pearson(xs, ys),
                         between_seq_pearson=pearson(np.array([s[0] for s in sm]), np.array([s[1] for s in sm]))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--family", default="gbm", choices=list(FAMILIES))
    ap.add_argument("--memory-size", type=int, default=5)
    ap.add_argument("--tau-train", type=float, default=0.5)
    ap.add_argument("--theta", type=float, default=0.5, help="a frame is BAD below this instrument IoU")
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--mlp-epochs", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/task4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    oof = load_dir(a.oof)
    folds = json.load(open(a.folds))["folds"]
    kw = {"seed": 0}
    if a.family == "mlp":
        kw["epochs"] = a.mlp_epochs
    R = {"family": a.family, "K": a.memory_size, "tau_train": a.tau_train, "theta": a.theta,
         "n_oof_frames": int(sum(len(c) for c in oof.values()))}

    # ------------------------------------------------------------- CV on OOF
    print(f"cross-fitted CV on {len(oof)} OOF sequences, family={a.family}, theta={a.theta}")
    preds = {}
    for s in a.sets:
        preds[s] = cv_run(oof, folds, a.family, s, a.memory_size, a.tau_train, a.theta, kw)
    base = preds[a.sets[0]]
    r_star, seq = base[1], base[2]
    R["cv"] = {s: metrics(preds[s][0], preds[s][1], a.theta) for s in a.sets}
    print("\nHELD-OUT (OOF) SEQUENCES")
    print(f"{'set':20s} {'AUC':>7s} {'AURC':>8s} {'Pearson':>8s} {'MSE':>8s}   (AURC floor {R['cv'][a.sets[0]]['aurc_oracle']:.4f}, random {R['cv'][a.sets[0]]['aurc_random']:.4f})")
    for s in a.sets:
        m = R["cv"][s]
        print(f"{s:20s} {m['auc']:7.4f} {m['aurc']:8.4f} {m['pearson']:8.4f} {m['mse']:8.5f}")

    comps = [("extended_stereo", "extended"), ("nontemporal_stereo", "nontemporal"),
             ("nontemporal_stereo", "extended"), ("stereo3", "nontemporal"), ("extended", "nontemporal")]
    R["cv_paired"] = {}
    print("\nPaired cluster bootstrap over sequences (difference = first minus second; AURC: negative is better)")
    for x, y in comps:
        if x in preds and y in preds:
            d = boot_diff(preds[x][0], preds[y][0], r_star, seq, a.theta, a.n_boot)
            d["verdict"] = verdict(d, x, y)
            R["cv_paired"][f"{x} vs {y}"] = d
            print(f"  {x} vs {y}: dAUC {d['auc']['mean']:+.4f} [{d['auc']['ci95'][0]:+.4f}, {d['auc']['ci95'][1]:+.4f}]  "
                  f"dAURC {d['aurc']['mean']:+.4f} [{d['aurc']['ci95'][0]:+.4f}, {d['aurc']['ci95'][1]:+.4f}]  -> {d['verdict']}")

    # --------------------------------------------------- indicator diagnostics
    print("\nPer-indicator (teacher-forced history; static indicators do not depend on it)")
    R["indicators"] = indicator_table(oof, a.memory_size, a.tau_train, a.theta)
    print(f"{'indicator':22s} {'r':>7s} {'AUC':>6s} {'within-seq r':>13s} {'between-seq r':>14s}")
    for r in R["indicators"]:
        if "auc_good_vs_bad" in r:
            print(f"{r['indicator']:22s} {r['pearson']:+7.3f} {r['auc_good_vs_bad']:6.3f} {r['within_seq_pearson']:+13.3f} {r['between_seq_pearson']:+14.3f}")

    # leave-one-indicator-out on the 12-indicator set
    print("\nLeave-one-indicator-out (12-indicator gate): drop in CV AUC when an indicator is removed")
    full = R["cv"].get("extended_stereo", {}).get("auc")
    R["loo"] = {}
    if full is not None:
        names = indicator_names("extended_stereo")
        for n in names:
            key = f"loo_{n}"
            REL.INDICATOR_SETS[key] = tuple(x for x in names if x != n)
            rh, rs, _ = cv_run(oof, folds, a.family, key, a.memory_size, a.tau_train, a.theta, kw)
            m = metrics(rh, rs, a.theta)
            R["loo"][n] = dict(auc_without=m["auc"], auc_drop=full - m["auc"], aurc_without=m["aurc"])
            print(f"  without {n:22s} AUC {m['auc']:.4f}  (drop {full - m['auc']:+.4f})")
            del REL.INDICATOR_SETS[key]

    # ------------------------------------------------------------ official test
    if a.test:
        test = load_dir(a.test)
        tkeys = [k for k in test if k.startswith("test/")]
        test = {k: test[k] for k in tkeys}
        print(f"\nOFFICIAL TEST SEQUENCES ({len(test)}), gate fit on all training OOF data")
        tp = {s: test_run(oof, test, a.family, s, a.memory_size, a.tau_train, a.theta, kw) for s in a.sets}
        tr_star, tseq = tp[a.sets[0]][1], tp[a.sets[0]][2]
        R["test"] = {s: metrics(tp[s][0], tp[s][1], a.theta) for s in a.sets}
        print(f"{'set':20s} {'AUC':>7s} {'AURC':>8s} {'Pearson':>8s} {'MSE':>8s}")
        for s in a.sets:
            m = R["test"][s]
            print(f"{s:20s} {m['auc']:7.4f} {m['aurc']:8.4f} {m['pearson']:8.4f} {m['mse']:8.5f}")
        R["test_paired"] = {}
        print("  (only 4 test sequences: the sequence-bootstrap interval is coarse, read it as a warning, not a test)")
        for x, y in comps:
            if x in tp and y in tp:
                d = boot_diff(tp[x][0], tp[y][0], tr_star, tseq, a.theta, a.n_boot)
                d["verdict"] = verdict(d, x, y)
                R["test_paired"][f"{x} vs {y}"] = d
                print(f"  {x} vs {y}: dAUC {d['auc']['mean']:+.4f} [{d['auc']['ci95'][0]:+.4f}, {d['auc']['ci95'][1]:+.4f}]  "
                      f"dAURC {d['aurc']['mean']:+.4f}  -> {d['verdict']}")
    R["seconds"] = time.time() - t0
    with open(os.path.join(a.out, f"gate_comparison_{a.family}.json"), "w") as f:
        json.dump(R, f, indent=1)
    print(f"\nwrote {a.out}/gate_comparison_{a.family}.json  ({R['seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
