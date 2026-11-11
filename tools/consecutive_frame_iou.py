#!/usr/bin/env python3
"""
TASK K -- IoU between the GROUND-TRUTH instrument masks of consecutive frames (and of random pairs, for scale).

    python -m tools.consecutive_frame_iou --root <endovis2018_cvdataset> --out results/taskK

Ground truth only: this is the best any "copy the previous mask" method could do, so it bounds what temporal propagation can
offer at this frame rate. Computed at full 1280x1024 resolution from the label PNGs (repairs applied, as everywhere).

Two definitions, both reported, neither substituted for the other:
  instrument_iou   sitr.metrics.instrument_iou: per-class IoU over the 8 instrument classes, a class counted if it is in either
                   mask, an empty pair scoring 1.0. This is the definition the gate's target and DATASET_FACTS.md section 6 use.
  region_iou       IoU of the binary union of the instrument classes (class identity ignored).

A pair is (frame t-1, frame t) of one sequence, and is used only if the two frame file numbers differ by exactly 1 (pairs across a
removed frame are counted and excluded). Random pairs: for every frame one partner drawn uniformly from the same sequence at
least 20 frames away (seed 0). Intervals are 95% percentile intervals of the mean from a bootstrap that resamples whole
sequences (2000 draws, seed 0).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.metrics import instrument_iou  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

INSTR = [1, 2, 3, 6, 7, 8, 9, 11]
NC = 12
LOW = 0.5                      # "low overlap" threshold, the repo's standing 0.5 (not tuned)


def frame_no(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def region_iou(a, b):
    ia, ib = np.isin(a, INSTR), np.isin(b, INSTR)
    u = int((ia | ib).sum())
    return 1.0 if u == 0 else float((ia & ib).sum()) / u


def pair_scores(a, b):
    return instrument_iou(a, b, NC, INSTR, empty_value=1.0), region_iou(a, b)


def describe(v):
    v = np.asarray(v, float)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return dict(n=int(v.size), mean=float(v.mean()), median=float(q[2]), p05=float(q[0]), p25=float(q[1]), p75=float(q[3]),
                p95=float(q[4]), min=float(v.min()), max=float(v.max()), frac_below_0_5=float((v < LOW).mean()),
                frac_below_0_2=float((v < 0.2).mean()), frac_above_0_8=float((v > 0.8).mean()),
                histogram_10_bins=np.histogram(v, bins=10, range=(0, 1))[0].astype(int).tolist())


def cluster_ci(per_seq, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(per_seq)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        means.append(float(np.concatenate([per_seq[keys[i]] for i in pick]).mean()))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskK")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("train", "test"))
    seqs = sorted(seqs, key=lambda s: (s.split != "train", s.index))
    rng = np.random.default_rng(0)
    R = {"definitions": {"instrument_iou": "per-class IoU over the 8 instrument classes, class counted if in either mask, empty pair = 1.0",
                         "region_iou": "IoU of the union of the instrument classes", "low_overlap_threshold": LOW,
                         "resolution": "1280x1024 (label PNGs)", "interval": "95% percentile interval of the mean, bootstrap over sequences (2000 draws, seed 0)"},
         "groups": {}}
    out = {"train": {"cons": {}, "rand": {}}, "test": {"cons": {}, "rand": {}}}
    skipped = {}
    for s in seqs:
        masks = [read_mask(p) for p in s.labels]
        nums = [frame_no(p) for p in s.frames]
        cons = [pair_scores(masks[t - 1], masks[t]) for t in range(1, len(masks)) if nums[t] - nums[t - 1] == 1]
        skipped[s.key] = int(len(masks) - 1 - len(cons))
        rp = []
        for t in range(len(masks)):
            far = [u for u in range(len(masks)) if abs(u - t) >= 20]
            if far:
                rp.append(pair_scores(masks[t], masks[int(rng.choice(far))]))
        out[s.split]["cons"][s.key] = np.array(cons)
        out[s.split]["rand"][s.key] = np.array(rp)
        print(f"  {s.key:<12} {len(cons)} consecutive pairs (skipped {skipped[s.key]})  instrument IoU mean {np.mean([c[0] for c in cons]):.3f}  "
              f"region {np.mean([c[1] for c in cons]):.3f}   random pairs {np.mean([r[0] for r in rp]):.3f}", flush=True)
    for split in ("train", "test"):
        G = {"sequences": {}, "pooled": {}}
        for kind in ("cons", "rand"):
            d = out[split][kind]
            for j, name in enumerate(("instrument_iou", "region_iou")):
                per = {k: v[:, j] for k, v in d.items()}
                pooled = np.concatenate(list(per.values()))
                G["pooled"][f"{'consecutive' if kind == 'cons' else 'random_pairs_20_apart'}_{name}"] = dict(
                    **describe(pooled), mean_ci95_sequence_bootstrap=cluster_ci(per))
                if kind == "cons":
                    for k, v in per.items():
                        G["sequences"].setdefault(k, {})[name] = describe(v)
        G["frame_gaps_skipped"] = {k: v for k, v in skipped.items() if k.split("/")[0] == split}
        R["groups"][split] = G
    json.dump(R, open(os.path.join(a.out, "consecutive_frame_iou.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "pair_scores.npz"),
                        **{f"{sp}_{k}_{key.replace('/', '_')}": v for sp in out for k in out[sp] for key, v in out[sp][k].items()})
    print("wrote", os.path.join(a.out, "consecutive_frame_iou.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
