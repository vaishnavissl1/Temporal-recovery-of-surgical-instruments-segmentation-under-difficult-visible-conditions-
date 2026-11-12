#!/usr/bin/env python3
"""
Owner's request, items 1 and 3 (CPU only).

    python -m tools.gate_ablation_seeds --work work_v2 --scores work_v2/failure_detection --out results/taskC [--seeds 5]

Item 1. Max-softmax (the mean top-class probability over predicted-foreground pixels) IS ALREADY one of the production gate's nine indicators (`q_confidence`, sitr/reliability.py),
so nothing is added. What is reported is the gate WITH it (the nine) and WITHOUT it (the other eight, set `extended_no_q`), the same MLP, K=3, tau_train 0.5, 300 epochs, the same folds,
cross-fitted out of fold and then fitted once on all 15 training sequences and scored once on the test sequences, against the max-softmax baseline on the same frames. AUROC and AURC on both
splits and both failure labels (official 12-class frame IoU below 0.5, and instrument IoU below 0.5). The without-q variant is an ablation, not a candidate: nothing is selected from it.

Item 3. The with-q gate refitted out of fold with `--seeds` random seeds (seed controls the MLP initialisation, the batch sampling and the scheduled-sampling draws); mean, standard deviation
and range of AUROC and AURC out of fold. The test set is scored for seed 0 only (item 1), never for the other seeds. The segmenter is not re-seeded.

Seed 0 of the with-q refit is the production recipe (tools/train_stage2_cv.py defaults), so its out-of-fold figures should reproduce the production gate's; the difference is recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.metrics import aurc, aurc_oracle, roc_auc  # noqa: E402
from tools.compare_stereo import cv_run, load_dir, test_run  # noqa: E402
from tools.failure_detect_table import THETA, align, boot, load_scores  # noqa: E402

K, TAU_TRAIN, EPOCHS = 3, 0.5, 300


def frames(work, scores_dir):
    folds = json.load(open(os.path.join(work, "folds.json")))["folds"]
    oof_c = load_dir(os.path.join(work, "cache_oof_stereo"))
    test_c = {k: v for k, v in load_dir(os.path.join(work, "cache_test_stereo")).items() if k.startswith("test/")}
    out = {}
    for split, group, caches in (("train", "oof", oof_c), ("test", "test", test_c)):
        sc = load_scores(scores_dir, split)
        keys, rows = align(split, sc, caches, os.path.join(work, "actions"))
        cat = lambda f: np.concatenate([f(rows[k]) for k in keys])  # noqa: E731
        risk = cat(lambda r: r["s"]["risk"])
        r_star = cat(lambda r: r["r_star"])
        seq = np.concatenate([[k] * len(rows[k]["s"]["risk"]) for k in keys])
        out[group] = dict(keys=keys, risk=risk, r_star=r_star, seq=seq, ms=cat(lambda r: r["s"]["ms_fg"]).astype(np.float64), prod=cat(lambda r: r["prod"]),
                          bad=(risk > 1 - THETA).astype(int), bad_i=(r_star < THETA).astype(int))
    return folds, oof_c, test_c, out


def metrics(score, G):
    ok = np.isfinite(score) & np.isfinite(G["risk"])
    s, rk, ri, b, bi = score[ok], G["risk"][ok], 1.0 - G["r_star"][ok], G["bad"][ok], G["bad_i"][ok]
    return dict(auroc_official=roc_auc(-s, b), aurc_official=aurc(s, rk), auroc_instrument=roc_auc(-s, bi), aurc_instrument=aurc(s, ri), n=int(ok.sum()))


def ordered(rh, sq, keys):
    return np.concatenate([rh[sq == k] for k in keys])


def fit_oof(oof_c, folds, setname, seed, G):
    t = time.time()
    rh, _, sq = cv_run(oof_c, folds, "mlp", setname, K, TAU_TRAIN, THETA, {"seed": seed, "epochs": EPOCHS})
    print(f"  oof {setname} seed {seed}: {time.time() - t:.0f}s", flush=True)
    return ordered(rh, sq, G["oof"]["keys"])


def fit_test(oof_c, test_c, setname, seed, G):
    t = time.time()
    th, _, tsq = test_run(oof_c, test_c, "mlp", setname, K, TAU_TRAIN, THETA, {"seed": seed, "epochs": EPOCHS})
    print(f"  test {setname} seed {seed}: {time.time() - t:.0f}s", flush=True)
    return ordered(th, tsq, G["test"]["keys"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work_v2")
    ap.add_argument("--scores", default="work_v2/failure_detection")
    ap.add_argument("--out", default="results/taskC")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    folds, oof_c, test_c, G = frames(a.work, a.scores)
    R = dict(recipe=dict(family="mlp", memory_size=K, tau_train=TAU_TRAIN, epochs=EPOCHS, folds="work_v2/folds.json (3 folds of 5 sequences)", theta=THETA),
             note="Max-softmax is already one of the nine production indicators (q_confidence); nothing was added. Seed 0 with q is the production recipe.",
             n_frames={g: int(len(G[g]["risk"])) for g in G}, item1={}, item3={})

    # ---- item 1: with / without max-softmax, seed 0, both splits, scored once
    scores = {"oof": {}, "test": {}}
    for label, setname in (("gate with max-softmax (9 indicators), refit seed 0", "extended"), ("gate without max-softmax (8 indicators), refit seed 0", "extended_no_q")):
        scores["oof"][label] = fit_oof(oof_c, folds, setname, 0, G)
        scores["test"][label] = fit_test(oof_c, test_c, setname, 0, G)
    for g in ("oof", "test"):
        det = {"max-softmax baseline": G[g]["ms"], "production gate (released scores)": G[g]["prod"], **scores[g]}
        ref = G[g]["ms"]
        rows = {}
        for name, sc in det.items():
            m = metrics(sc, G[g])
            b = boot(sc, G[g]["risk"], G[g]["bad"], G[g]["bad_i"], G[g]["seq"], ref=None if name.startswith("max-softmax") else ref)
            m.update(auroc_official_ci95=b["auroc_ci95"], aurc_official_ci95=b["aurc_ci95"], n_clusters=b["n_clusters"])
            if "diff_auroc_vs_production_gate" in b:
                m["auroc_official_minus_max_softmax"] = b["diff_auroc_vs_production_gate"]
                m["aurc_official_minus_max_softmax"] = b["diff_aurc_vs_production_gate"]
            rows[name] = m
        ok = np.isfinite(G[g]["risk"])
        R["item1"][g] = dict(n_frames=int(ok.sum()), bad_rate_official=float(G[g]["bad"][ok].mean()), bad_rate_instrument=float(G[g]["bad_i"][ok].mean()),
                             aurc_floor_official=aurc_oracle(G[g]["risk"][ok]), aurc_random_official=float(G[g]["risk"][ok].mean()), detectors=rows)
    json.dump(R, open(os.path.join(a.out, "gate_ablation_seeds.json"), "w"), indent=1)
    print("item 1 written", flush=True)

    # ---- item 3: five seeds, out of fold only
    seeds = {}
    for s in range(a.seeds):
        sc = scores["oof"]["gate with max-softmax (9 indicators), refit seed 0"] if s == 0 else fit_oof(oof_c, folds, "extended", s, G)
        seeds[s] = metrics(sc, G["oof"])
        print(f"  seed {s}: {seeds[s]}", flush=True)
    keys = ("auroc_official", "aurc_official", "auroc_instrument", "aurc_instrument")
    summ = {k: dict(values=[seeds[s][k] for s in seeds], mean=float(np.mean([seeds[s][k] for s in seeds])), std=float(np.std([seeds[s][k] for s in seeds], ddof=1)),
                    min=float(min(seeds[s][k] for s in seeds)), max=float(max(seeds[s][k] for s in seeds))) for k in keys}
    ms = metrics(G["oof"]["ms"], G["oof"])
    R["item3"] = dict(seeds=list(seeds), per_seed=seeds, summary=summ, max_softmax_oof=ms,
                      production_oof={k: v for k, v in metrics(G["oof"]["prod"], G["oof"]).items()},
                      statement="the segmenter is not re-seeded; only the gate (initialisation, batch draws, scheduled sampling) is")
    json.dump(R, open(os.path.join(a.out, "gate_ablation_seeds.json"), "w"), indent=1)
    print("item 3 written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
