#!/usr/bin/env python3
"""
TASK H -- does per-PIXEL stereo evidence say WHERE the segmentation is wrong?

    python -m tools.occlusion_localisation --work work_v2 --root <endovis2018_cvdataset> --out results/taskH

The frame-level versions of this signal failed (difficulty index Spearman about 0; occluded_fraction AUC 0.506). A frame-level failure does not
imply the pixel-level one fails, and localisation is a different claim: it would say which part of a mask to distrust. Nothing here is fitted, so
there is no threshold or model to tune; the out-of-fold block is the reported result and the test block is computed once, after, for completeness.

Universe: pixels PREDICTED as an instrument class (8 ids) that lie inside the stereo field of view (in_view). Label: the pixel is WRONG if the ground-
truth class there differs from the predicted class (ground truth is used only to make this label). Everything is on the 256x320 cache grid: the cached
predicted mask (out-of-fold honest: each training sequence is predicted by the fold model that excluded it), the ground truth nearest-neighbour
downsampled, and the 512x640 stereo evidence nearest-neighbour downsampled, exactly as tools/add_stereo_indicators.py does.

Pixel scores, higher = more suspect:
  occluded       the pixel is hidden from the right camera (binary)
  lr_fail        the pixel fails the left-right disparity consistency check (binary)
  low_conf       255 - disparity confidence (0..255)
Reference localiser (not stereo): boundary_prox = 31 - (distance in pixels to the nearest class boundary of the predicted mask, clipped to 31). Errors concentrate
at mask boundaries, so any stereo signal has to be read against it.

Reported, each as a pixel-level AUROC with a 95% interval from a bootstrap that resamples whole SEQUENCES (2000 draws, seed 0):
  overall               all universe pixels, pooled
  interior              pixels at least 4 grid pixels from a boundary (4 is a stated choice)
  near_boundary         the rest
  boundary_stratified   probability that a random wrong pixel outranks a random correct pixel AT THE SAME boundary distance (a pair is compared only within a
                        distance level, levels combined by the number of wrong/correct pairs they contribute). This removes the boundary confound.
The AUROCs come from per-sequence histograms over (boundary distance, score level, label), so the intervals are exact functions of the data, not approximations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.reliability import SequenceCache  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

INSTR = [1, 2, 3, 6, 7, 8, 9, 11]
DMAX = 31
INTERIOR = 4
SCORES = ("occluded", "lr_fail", "low_conf", "boundary_prox")


def boundary_distance(pred):
    """Distance (grid pixels, clipped to DMAX) to the nearest pixel whose 4-neighbour has a different predicted class."""
    b = np.zeros(pred.shape, bool)
    dv, dh = pred[1:, :] != pred[:-1, :], pred[:, 1:] != pred[:, :-1]
    b[1:, :] |= dv; b[:-1, :] |= dv; b[:, 1:] |= dh; b[:, :-1] |= dh
    d = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    return np.minimum(np.round(d), DMAX).astype(np.int64)


def seq_hist(cache, seq, meta, sd_dir, labels):
    """Histogram [score, dist(32), level(256), label(2)] of the universe pixels of one sequence."""
    H = np.zeros((len(SCORES), DMAX + 1, 256, 2), np.int64)
    h, w = cache.masks.shape[1:]
    for t, fn in enumerate(meta["frames"]):
        pred = cache.masks[t]
        z = np.load(os.path.join(sd_dir, os.path.splitext(fn)[0] + ".npz"))
        fit = lambda x, dt: cv2.resize(x.astype(dt), (w, h), interpolation=cv2.INTER_NEAREST)  # noqa: E731
        inview = fit(z["in_view"], np.uint8).astype(bool)
        uni = np.isin(pred, INSTR) & inview
        if not uni.any():
            continue
        gt = cv2.resize(read_mask(labels[t]), (w, h), interpolation=cv2.INTER_NEAREST)
        wrong = (gt != pred)[uni].astype(np.int64)
        d = boundary_distance(pred)[uni]
        occ = fit(z["occluded"], np.uint8).astype(np.int64)[uni] * 255
        lrf = (~fit(z["lr_ok"], np.uint8).astype(bool)).astype(np.int64)[uni] * 255
        lc = 255 - fit(z["confidence"], np.uint8).astype(np.int64)[uni]
        bp = (DMAX - d) * 8                         # 0..248, a monotone map of proximity into the 256 levels
        for i, sc in enumerate((occ, lrf, lc, bp)):
            np.add.at(H[i], (d, sc, wrong), 1)
    return H


def auc_from_counts(pos, neg):
    """AUROC from two histograms over the same ordered levels (ties count one half)."""
    pos, neg = pos.astype(np.float64), neg.astype(np.float64)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    cneg = np.cumsum(neg) - neg                      # negatives strictly below each level
    return float((pos * (cneg + 0.5 * neg)).sum() / (P * N))


def aucs(H):
    """H: [score, dist, level, label]. Returns dict score -> {overall, interior, near_boundary, boundary_stratified}."""
    out = {}
    for i, name in enumerate(SCORES):
        h = H[i]
        pos, neg = h[..., 1], h[..., 0]                # [dist, level]
        o = auc_from_counts(pos.sum(0), neg.sum(0))
        it = auc_from_counts(pos[INTERIOR:].sum(0), neg[INTERIOR:].sum(0))
        nb = auc_from_counts(pos[:INTERIOR].sum(0), neg[:INTERIOR].sum(0))
        num = den = 0.0
        for d in range(DMAX + 1):
            P, N = pos[d].sum(), neg[d].sum()
            if P > 0 and N > 0:
                num += P * N * auc_from_counts(pos[d], neg[d]); den += P * N
        out[name] = dict(overall=o, interior=it, near_boundary=nb, boundary_stratified=(num / den if den else float("nan")))
    return out


def boot(Hs, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(Hs)
    acc = {n: {k: [] for k in ("overall", "interior", "near_boundary", "boundary_stratified")} for n in SCORES}
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        a = aucs(sum(Hs[keys[i]] for i in pick))
        for n in SCORES:
            for k in acc[n]:
                acc[n][k].append(a[n][k])
    return {n: {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in d.items()} for n, d in acc.items()}


def run_group(name, caches_dir, a, split):
    seqs = {s.key: s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=(split,))}
    Hs = {}
    for f in sorted(os.listdir(caches_dir)):
        if not f.endswith(".npz"):
            continue
        c = SequenceCache.load(os.path.join(caches_dir, f))
        if c.name not in seqs:
            continue
        sd_dir = os.path.join(a.work, "stereo", c.name.replace("/", "_"))
        meta = json.load(open(os.path.join(sd_dir, "stereo_meta.json")))
        s = seqs[c.name]
        if not (len(meta["frames"]) == len(c) == len(s.labels)):
            raise SystemExit(f"{c.name}: frame counts disagree (stereo {len(meta['frames'])}, cache {len(c)}, labels {len(s.labels)})")
        Hs[c.name] = seq_hist(c, s, meta, sd_dir, s.labels)
        n = int(Hs[c.name][0].sum())
        wrong = int(Hs[c.name][0][..., 1].sum())
        print(f"  {c.name:<12} {n} universe pixels, {100*wrong/max(n,1):.1f}% wrong", flush=True)
    tot = sum(Hs.values())
    res = aucs(tot)
    ci = boot(Hs)
    G = dict(n_sequences=len(Hs), n_pixels=int(tot[0].sum()), wrong_rate=float(tot[0][..., 1].sum() / tot[0].sum()), scores={})
    for n in SCORES:
        G["scores"][n] = {k: dict(auroc=res[n][k], ci95=ci[n][k]) for k in res[n]}
    G["per_sequence_overall_auroc"] = {k: {n: aucs(h)[n]["overall"] for n in SCORES} for k, h in Hs.items()}
    return G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work_v2")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskH")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {"definition": __doc__, "interior_min_boundary_distance_px": INTERIOR, "grid": "256x320",
         "groups": {"oof": run_group("oof", os.path.join(a.work, "cache_oof"), a, "train")}}
    json.dump(R, open(os.path.join(a.out, "occlusion_localisation.json"), "w"), indent=1)
    print("\nOUT OF FOLD (15 training sequences)")
    G = R["groups"]["oof"]
    for n, d in G["scores"].items():
        print(f"  {n:14s} " + "  ".join(f"{k} {v['auroc']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]" for k, v in d.items()))
    R["groups"]["test"] = run_group("test", os.path.join(a.work, "cache_test"), a, "test")
    json.dump(R, open(os.path.join(a.out, "occlusion_localisation.json"), "w"), indent=1)
    print("\nTEST (4 sequences, computed once)")
    for n, d in R["groups"]["test"]["scores"].items():
        print(f"  {n:14s} " + "  ".join(f"{k} {v['auroc']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]" for k, v in d.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
