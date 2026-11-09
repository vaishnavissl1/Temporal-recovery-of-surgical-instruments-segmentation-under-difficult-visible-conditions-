#!/usr/bin/env python3
"""Build cross-fitting folds that keep every rare class in every fold's TRAINING set.

Why: sitr.stage2v2.make_folds balances only by frame count. On EndoVis 2018 that
can put BOTH ultrasound-probe sequences (seq_5, seq_14) in one held-out fold, so
that fold's model never sees a probe and its out-of-fold errors on probe are
caused by missing training data, not by difficulty. Likewise the 12 needle frames
live in only two sequences (seq_1, seq_4).

Method (no hand-picked folds): enumerate EVERY way to split the sequences into k
folds of equal size and keep the split that maximises, lexicographically, the
sorted list of ratios  (training frames containing class c) / (all frames
containing class c)  over every fold and every class that is rare
(fewer than --rare-frac of frames). Ties are broken by the sequence order, so the
result is deterministic. Whole sequences only (frames at 1 Hz still leak).

Class presence comes from the mask scan produced by
tools/task1_scans/scan_masks.py (exact colours per label PNG), with the repairs
override applied, so nothing here is typed in by hand.

    python -m tools.make_folds_balanced --scan results/task1/mask_scan.json \
        --k 3 --out work_v2/folds.json
"""
import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np

PALETTE = {(0, 0, 0): 0, (0, 255, 0): 1, (0, 255, 255): 2, (125, 255, 12): 3,
           (255, 55, 0): 4, (24, 55, 125): 5, (187, 155, 25): 6, (0, 255, 125): 7,
           (255, 255, 125): 8, (123, 15, 175): 9, (124, 155, 5): 10,
           (12, 255, 141): 11}


def presence_from_scan(scan: dict):
    """-> {sequence key: (n_frames, int array[12] of frames containing each class)}."""
    frames = {k: v for k, v in scan.items() if k.startswith("train/")}
    for k, v in scan.items():                      # repairs override originals
        if k.startswith("repair/"):
            seq, fr = k.split("/")[1].rsplit("_", 1)
            frames[f"train/{seq}/{fr}"] = v
    out = defaultdict(lambda: [0, np.zeros(12, dtype=np.int64)])
    for key, v in frames.items():
        seq = "/".join(key.split("/")[:2])
        out[seq][0] += 1
        for c in v["colors"]:
            c = int(c)
            cid = PALETTE.get(((c >> 16) & 255, (c >> 8) & 255, c & 255))
            if cid is not None:
                out[seq][1][cid] += 1
    return {s: (n, p) for s, (n, p) in out.items()}


def build_folds(seqs: dict, k: int, rare_frac: float):
    keys = sorted(seqs, key=lambda s: int(s.split("_")[-1]))
    n_total = sum(seqs[s][0] for s in keys)
    pres = np.stack([seqs[s][1] for s in keys])             # [S, 12]
    totals = pres.sum(0)
    rare = [c for c in range(1, 12) if 0 < totals[c] < rare_frac * n_total]
    size = len(keys) // k
    if size * k != len(keys):
        raise SystemExit(f"{len(keys)} sequences cannot be split into {k} equal folds")

    def partitions(items, k):
        if k == 1:
            yield [items]
            return
        first, rest = items[0], items[1:]       # fix the first item: no permutations
        for combo in itertools.combinations(rest, size - 1):
            grp = (first,) + combo
            rem = [x for x in rest if x not in combo]
            for tail in partitions(rem, k - 1):
                yield [list(grp)] + tail

    best, best_key = None, None
    for part in partitions(list(range(len(keys))), k):
        ratios = []
        for held in part:
            train = pres[[i for i in range(len(keys)) if i not in held]].sum(0)
            ratios += [train[c] / totals[c] for c in rare]
        key = sorted(ratios)
        if best_key is None or key > best_key:
            best, best_key = part, key
    folds = [sorted(keys[i] for i in grp) for grp in best]
    folds.sort(key=lambda f: int(f[0].split("_")[-1]))
    return folds, rare, totals, pres, keys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rare-frac", type=float, default=0.35,
                    help="a class is 'rare' if it appears in fewer than this "
                         "fraction of training frames (a stated choice)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.scan) as f:
        scan = json.load(f)
    seqs = presence_from_scan(scan)
    folds, rare, totals, pres, keys = build_folds(seqs, a.k, a.rare_frac)
    names = ["background", "shaft", "clasper", "wrist", "kidney", "covered-kidney",
             "thread", "clamps", "needle", "suction", "intestine", "probe"]
    print(f"rare classes (< {a.rare_frac:.0%} of frames): {[names[c] for c in rare]}")
    for i, held in enumerate(folds):
        idx = [keys.index(s) for s in held]
        train = pres[[j for j in range(len(keys)) if j not in idx]].sum(0)
        cov = {names[c]: f"{int(train[c])}/{int(totals[c])}" for c in rare}
        print(f"fold {i}: held out {held}\n         training frames per rare class: {cov}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"folds": folds, "k": a.k,
                   "note": "whole sequences only; chosen by exhaustive search to "
                           "maximise the worst-case rare-class training coverage "
                           "(tools/make_folds_balanced.py)"}, f, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
