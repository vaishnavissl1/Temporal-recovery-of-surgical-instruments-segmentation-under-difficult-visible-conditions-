#!/usr/bin/env python3
"""
TASK I -- what is each mIoU definition choice worth, on the SAME predictions?

    python -m tools.metric_ablation --actions work_v2/actions --root <endovis2018_cvdataset> --out results/taskI

The predictions are the M1 (segmentation only) masks of the 4 official test sequences, 997 frames (work_v2/actions/test_masks, key A0).
The reference row is the exact official protocol (test_data/utils.py): full 1280x1024, class 0 never scored, a class scored only if it is in
the ground truth, mean over classes within a frame, mean over frames within a sequence, then the sequence means averaged WEIGHTED by the
number of scored frames (utils.py:91). Every other row changes exactly ONE choice from that reference, and the last rows combine them.

Choices, each a pair of settings (reference first):
  background        class 0 excluded        | class 0 included in the per-frame mean
  presence rule     class counted if in the ground truth | counted if in the ground truth OR the prediction
  aggregation       per-frame mean over classes, then over frames | per-class mean over frames, then over classes (mcIoU)
  sequence weights  sequence means weighted by scored frames (official) | sequence means averaged unweighted
  resolution        1280x1024 (prediction upsampled nearest) | 512x640 (ground truth downsampled nearest)

Masks are resampled with nearest neighbour only. Counts per class are taken once per frame and resolution and every variant is computed from
those counts, so the variants differ only by the stated choice. suction-instrument has zero test pixels, so under the per-class aggregation
11 of the 12 classes are scored (the 12th has no frame in which it is present in the ground truth).
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
from sitr.metrics import _class_counts  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

NC = 12


def per_frame_counts(pred, gt):
    inter, pr, gc = _class_counts(pred, gt, NC)
    return inter.astype(np.int64), pr.astype(np.int64), gc.astype(np.int64)


def frame_class_ious(counts, include_bg, presence):
    inter, pr, gc = counts
    out = {}
    for c in range(0 if include_bg else 1, NC):
        g, p = int(gc[c]), int(pr[c])
        present = (g > 0) if presence == "gt" else (g > 0 or p > 0)
        if not present:
            continue
        u = g + p - int(inter[c])
        out[c] = float(inter[c]) / u if u > 0 else 1.0
    return out


def score(frames, seq_of, include_bg=False, presence="gt", per_class=False, weighted=True):
    """frames: list of counts tuples; seq_of: sequence key per frame. Returns (mIoU, number of frames scored, classes scored)."""
    fi = [frame_class_ious(c, include_bg, presence) for c in frames]
    if per_class:
        cls = {}
        for d, s in zip(fi, seq_of):
            for c, v in d.items():
                cls.setdefault(c, []).append((s, v))
        if weighted:
            per = {c: float(np.mean([v for _, v in lst])) for c, lst in cls.items()}
        else:                                          # per class: unweighted mean of sequence means
            per = {c: float(np.mean([np.mean([v for s2, v in lst if s2 == s]) for s in sorted({s for s, _ in lst})])) for c, lst in cls.items()}
        return float(np.mean(list(per.values()))), len(frames), len(per), per
    vals = [(s, float(np.mean(list(d.values())))) for d, s in zip(fi, seq_of) if d]
    seqs = sorted({s for s, _ in vals})
    means = {s: float(np.mean([v for s2, v in vals if s2 == s])) for s in seqs}
    if weighted:
        w = {s: sum(1 for s2, _ in vals if s2 == s) for s in seqs}
        m = float(np.average([means[s] for s in seqs], weights=[w[s] for s in seqs]))
    else:
        m = float(np.mean([means[s] for s in seqs]))
    return m, len(vals), None, means


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", default="work_v2/actions")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskI")
    ap.add_argument("--crosscheck-json", default="results/task6/final_results.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",))
    full, small, key = [], [], []
    for s in seqs:
        for t, (fp, lp) in enumerate(zip(s.frames, s.labels)):
            stem = os.path.splitext(os.path.basename(fp))[0]
            pred = np.load(os.path.join(a.actions, "test_masks", f"{s.split}_{s.name}", stem + ".npz"))["A0"]
            gt = read_mask(lp)
            pf = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            gs = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            full.append(per_frame_counts(pf, gt))
            small.append(per_frame_counts(pred, gs))
            key.append(s.key)
        print(f"  {s.key}: {len(s.frames)} frames", flush=True)

    ref, n_ref, _, ref_seq = score(full, key)
    rows = [("reference: official protocol", dict(), full)]
    rows += [
        ("background included in the per-frame mean", dict(include_bg=True), full),
        ("class counted if in ground truth OR prediction", dict(presence="union"), full),
        ("per-class mean over frames, then over classes (mcIoU)", dict(per_class=True), full),
        ("sequence means averaged unweighted", dict(weighted=False), full),
        ("evaluated at 512x640 (ground truth downsampled)", dict(), small),
    ]
    out = []
    for name, kw, fr in rows:
        m, n, ncls, extra = score(fr, key, **kw)
        out.append(dict(name=name, mIoU=m, delta_points_vs_reference=100.0 * (m - ref), frames_scored=n,
                        classes_scored=ncls, settings=kw or "official defaults", resolution="512x640" if fr is small else "1280x1024"))
    combos = [("background included + ground-truth-or-prediction presence", dict(include_bg=True, presence="union"), full),
              ("background included, evaluated at 512x640", dict(include_bg=True), small),
              ("per-class mean (mcIoU) + background included", dict(include_bg=True, per_class=True), full)]
    for name, kw, fr in combos:
        m, n, ncls, extra = score(fr, key, **kw)
        out.append(dict(name=name, mIoU=m, delta_points_vs_reference=100.0 * (m - ref), frames_scored=n, classes_scored=ncls,
                        settings=kw, resolution="512x640" if fr is small else "1280x1024", combined=True))
    R = {"reference_mIoU": ref, "frames": n_ref, "per_sequence_reference": ref_seq, "rows": out,
         "predictions": "M1 test masks (work_v2/actions/test_masks, A0), 997 frames, 4 sequences",
         "unit": "delta_points_vs_reference is in mIoU points (0.01 = 1 point on a 0-100 scale is +1.0)"}
    if os.path.exists(a.crosscheck_json):
        m1 = json.load(open(a.crosscheck_json))["models"]["M1"]["mIoU"]
        R["crosscheck_task6_M1_mIoU"] = m1
        R["crosscheck_abs_difference"] = abs(m1 - ref)
    json.dump(R, open(os.path.join(a.out, "metric_ablation.json"), "w"), indent=1)
    lines = ["| setting | mIoU | change vs reference (points) | frames | classes scored |", "|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['name']} | {r['mIoU']:.4f} | {r['delta_points_vs_reference']:+.2f} | {r['frames_scored']} | {r['classes_scored'] if r['classes_scored'] else 'per frame'} |")
    open(os.path.join(a.out, "metric_ablation.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if "crosscheck_abs_difference" in R:
        print(f"\ncross-check against results/task6 M1 mIoU {R['crosscheck_task6_M1_mIoU']:.6f}: |difference| {R['crosscheck_abs_difference']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
