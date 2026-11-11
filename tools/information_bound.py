#!/usr/bin/env python3
"""
TASK G -- how much mask information does each axis carry? Stereo versus temporal, on ground-truth masks.

    python -m tools.information_bound --work work_v2 --root <endovis2018_cvdataset> --out results/taskG --every 5

LIMITATION, stated first: the RIGHT view has no labels (only the left eye is annotated), so the stereo axis cannot be scored against ground truth the way the
temporal axis can. What can be measured is a BOUND: how much of a ground-truth LEFT mask survives being carried into the right view and back.

Stereo round trip (rectified-left geometry, both disparities from the same SGBM run, so independent in estimation):
  1. rectify the ground-truth left label map (nearest neighbour only);
  2. forward: scatter each left pixel with a valid LEFT disparity dl to the right view at (x - dl, y), nearest pixel, nearer surfaces (larger disparity) written last;
  3. back, TWO-MAP: scatter each right pixel that received a label to the left view at (x + dr, y) using the independently computed RIGHT disparity dr;
  4. back, SAME-MAP (the inflated variant, shown so the inflation is visible): pull the label back through the same dl, p -> p - dl(p) -> p.
  The round trip is scored against the rectified ground truth on the pixels where it is defined (valid dl, landing inside the image, and for the two-map variant a
  label arriving back). No hole filling. Convention x_left = x_right + d (sitr.stereo.disparity_pair).
  A control that uses NO disparity, the rectified ground truth against itself, would be 1.0 by construction and carries no information; the quantity reported is how much
  of the instrument region has a defined round trip at all, and the IoU on that part.

Temporal comparison, same frames, same metric, same grid: the ground-truth mask at t-1 warped to t with each of the three aligners (identity, Farneback, region matching,
sitr.align), scored against the ground-truth mask at t, in the same rectified-left geometry and on the same pixel set.

Metric: instrument IoU (sitr.metrics.instrument_iou: per class over the 8 instrument classes, a class counted if in either mask) and the region IoU of the union of instrument
classes. Everything is computed on a 320x256 grid obtained by taking every 4th pixel of the rectified full-resolution maps (nearest, never interpolated). Frames: every
`--every`-th frame of the training sequences (t >= 1) whose ground truth has instrument pixels inside the stereo-valid set; the same frames for every method. Intervals: 95%
percentile intervals of means from a bootstrap that resamples whole sequences (2000 draws, seed 0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.align import build_align  # noqa: E402
from sitr.dataset import discover_sequences  # noqa: E402
from sitr.stereo import (SGBMParams, VerticalCorrection, disparity_pair, get_rectifier, right_path)  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

INSTR = [1, 2, 3, 6, 7, 8, 9, 11]
NC = 12
S = 4                       # decimation of the rectified full-resolution maps (every 4th pixel)
OFF = 2


def scatter(labels, src_idx, dst_idx, order_key, n):
    """Write labels into a flat map of size n at dst_idx; larger order_key (nearer surface) written last so it wins."""
    out = np.full(n, 255, np.uint8)
    o = np.argsort(order_key, kind="stable")
    out[dst_idx[o]] = labels[o]
    return out


def round_trip(Lrect, dl, dr):
    """Returns (two_map, same_map, valid_two, valid_same), all HxW (uint8 labels with 255 = undefined; bool valid sets)."""
    H, W = Lrect.shape
    ys, xs = np.mgrid[0:H, 0:W]
    v = np.isfinite(dl) & (Lrect != 255)
    xr = np.rint(xs - np.where(v, dl, 0)).astype(np.int64)
    v &= (xr >= 0) & (xr < W)
    yy, xx, xt = ys[v], xs[v], xr[v]
    lab, d = Lrect[v], dl[v]
    flatR = scatter(lab, None, yy * W + xt, d, H * W)                      # right-view labels (255 = no pixel landed)
    MR = flatR.reshape(H, W)
    # same-map: pull back through dl
    same = np.full((H, W), 255, np.uint8)
    got = MR[yy, xt]
    same[yy, xx] = got
    valid_same = np.zeros((H, W), bool)
    valid_same[yy, xx] = got != 255
    # two-map: scatter right labels back with the independent right disparity
    rv = (MR != 255) & np.isfinite(dr)
    ry, rx = ys[rv], xs[rv]
    xl = np.rint(rx + dr[rv]).astype(np.int64)
    ok = (xl >= 0) & (xl < W)
    ry, rx, xl = ry[ok], rx[ok], xl[ok]
    flatL = scatter(MR[ry, rx], None, ry * W + xl, dr[ry, rx], H * W).reshape(H, W)
    valid_two = flatL != 255
    return flatL, same, valid_two, valid_same


def iou_pair(a, b, valid):
    """a, b: label maps on the same grid; valid: bool. Returns (instrument IoU, region IoU) over the valid pixels."""
    a, b = a[valid].astype(np.int64), b[valid].astype(np.int64)
    ious = []
    for c in INSTR:
        ia, ib = a == c, b == c
        u = int((ia | ib).sum())
        if u:
            ious.append(float((ia & ib).sum()) / u)
    ra, rb = np.isin(a, INSTR), np.isin(b, INSTR)
    u = int((ra | rb).sum())
    return (float(np.mean(ious)) if ious else float("nan")), (float((ra & rb).sum()) / u if u else float("nan"))


def describe(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return dict(n=int(v.size), mean=float(v.mean()), median=float(q[2]), p05=float(q[0]), p25=float(q[1]), p75=float(q[3]), p95=float(q[4]),
                frac_below_0_5=float((v < 0.5).mean()), histogram_10_bins=np.histogram(v, bins=10, range=(0, 1))[0].astype(int).tolist())


def cluster_ci(values, seqs, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    v, s = np.asarray(values, float), np.asarray(seqs)
    uniq = np.unique(s)
    idx = {u: np.where(s == u)[0] for u in uniq}
    m = [float(np.nanmean(v[np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])])) for _ in range(n_boot)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work_v2")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskG")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--limit-seqs", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = [s for s in sorted(discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("train",)), key=lambda s: s.index)]
    if a.limit_seqs:
        seqs = seqs[:a.limit_seqs]
    aligners = {n: build_align(n, **({"instrument_ids": INSTR} if n == "region" else {})) for n in ("identity", "farneback", "region")}
    rows, t0 = [], time.time()
    for s in seqs:
        meta = json.load(open(os.path.join(a.work, "stereo", f"{s.split}_{s.name}", "stereo_meta.json")))
        dist, rot = meta["convention"]
        rect = get_rectifier(meta["calibration_file"], meta["alpha"], dist, rot)
        corr = (VerticalCorrection(**{k: meta["correction"][k] for k in ("a", "b", "c", "n_matches", "mad", "max_abs_px", "applied")})
                if meta.get("correction") else None)
        params = SGBMParams(**meta["sgbm"])
        n_used = 0

        def rect_gray(t):
            Lr = rect.rectify(cv2.imread(s.frames[t]), cv2.imread(right_path(s.frames[t])))[0]
            return cv2.resize(cv2.cvtColor(Lr, cv2.COLOR_BGR2GRAY), (320, 256), interpolation=cv2.INTER_AREA)

        for t in range(1, len(s.frames), a.every):
            gt_t = rect.rectify_labels(read_mask(s.labels[t]), 255)
            L, Rimg = cv2.imread(s.frames[t]), cv2.imread(right_path(s.frames[t]))
            Lr, Rr = rect.rectify(L, Rimg)
            if corr is not None:
                Rr = corr.apply(Rr)
            dl, dr = disparity_pair(Lr, Rr, params)
            two, same, v_two, v_same = round_trip(gt_t, dl, dr)
            dec = lambda m: m[OFF::S, OFF::S]  # noqa: E731
            g, tw, sm, vt, vs = dec(gt_t), dec(two), dec(same), dec(v_two), dec(v_same)
            ginst = np.isin(g, INSTR)
            if not (ginst & vt).any():
                continue
            gt_p = dec(rect.rectify_labels(read_mask(s.labels[t - 1]), 255))
            gray_p, gray_t = rect_gray(t - 1), rect_gray(t)
            row = dict(seq=s.key, t=t, instr_pixels_valid_two_map=float((ginst & vt).sum() / ginst.sum()),
                       instr_pixels_valid_same_map=float((ginst & vs).sum() / ginst.sum()),
                       valid_fraction_all_pixels=float(vt.mean()))
            row["stereo_two_map"], row["stereo_two_map_region"] = iou_pair(g, tw, vt)
            row["stereo_same_map_on_two_map_set"], row["stereo_same_map_on_two_map_set_region"] = iou_pair(g, sm, vt)
            row["stereo_same_map_own_set"], row["stereo_same_map_own_set_region"] = iou_pair(g, sm, vs)
            for name, al in aligners.items():
                w = al(gt_p, gray_p, gray_t)
                row[f"temporal_{name}_on_stereo_set"], row[f"temporal_{name}_on_stereo_set_region"] = iou_pair(g, w, vt)
                row[f"temporal_{name}_all_pixels"], row[f"temporal_{name}_all_pixels_region"] = iou_pair(g, w, np.ones_like(vt))
            rows.append(row)
            n_used += 1
        print(f"  {s.key:<12} {n_used} frames used   {(time.time()-t0)/60:.1f} min", flush=True)
    keys = [k for k in rows[0] if k not in ("seq", "t")]
    seq_arr = [r["seq"] for r in rows]
    R = {"limitation": "the right view has no labels, so the stereo axis is bounded by a round trip of the ground-truth LEFT mask, not scored against right-view ground truth",
         "n_frames": len(rows), "n_sequences": len(set(seq_arr)), "grid": "320x256 (every 4th pixel of the rectified 1280x1024 maps, nearest)",
         "frames": "every %d-th frame of the training sequences with instrument ground truth inside the stereo-valid set" % a.every, "metrics": {}}
    for k in keys:
        vals = [r[k] for r in rows]
        R["metrics"][k] = dict(**describe(vals), mean_ci95_sequence_bootstrap=cluster_ci(vals, seq_arr))
    best = max(("identity", "farneback", "region"), key=lambda n: R["metrics"][f"temporal_{n}_on_stereo_set"]["mean"])
    diff = np.array([r["stereo_two_map"] - r[f"temporal_{best}_on_stereo_set"] for r in rows])
    R["stereo_two_map_minus_best_temporal"] = dict(best_temporal_aligner=best, mean=float(np.nanmean(diff)),
                                                   ci95=cluster_ci(diff, seq_arr), frac_frames_stereo_higher=float(np.nanmean(diff > 0)))
    json.dump(R, open(os.path.join(a.out, "information_bound.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(a.out, "per_frame.json"), "w"))
    for k in keys:
        if not k.endswith("_region"):
            m = R["metrics"][k]
            print(f"  {k:42s} mean {m['mean']:.3f} [{m['mean_ci95_sequence_bootstrap'][0]:.3f}, {m['mean_ci95_sequence_bootstrap'][1]:.3f}]  median {m['median']:.3f}")
    print("stereo(two-map) minus best temporal:", R["stereo_two_map_minus_best_temporal"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
