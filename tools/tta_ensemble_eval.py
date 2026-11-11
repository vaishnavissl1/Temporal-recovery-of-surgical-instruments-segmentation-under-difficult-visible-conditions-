#!/usr/bin/env python3
"""
TASK E -- segmenter TTA and the 4-model ensemble, scored with the same metrics as the single model.

    python -m tools.tta_ensemble_eval --masks work_v2/failure_detection/masks --root <endovis2018_cvdataset> --out results/taskE

Three SEPARATE systems on the 997 official test frames; never merged into one row:
  single     the segmentation model on its own (the system of record, M1). Re-scored here from the SAME run that produced the other two, so the comparison is
             like for like; it agrees with the published M1 to within the fp16 difference between two inference paths (reported).
  TTA        the same model, six passes (horizontal flip x scales 0.75, 1.0, 1.25), softmax probabilities averaged before argmax.
  ensemble   a DIFFERENT SYSTEM: four models (the three cross-fitting fold models and the final model), softmax averaged before argmax. Valid on the test sequences only,
             because every training sequence lay inside some member's training set, and it uses four trained models where the single system uses one.
Metrics: official mIoU, mcIoU (per class over frames, then over classes), per-class IoU, boundary F1 on the instrument union (3 px tolerance, as tools/final_eval.py),
clDice on shaft / thread / needle, all at 1280x1024 under the official rule (class 0 never scored; a class counted only if in the ground truth). Differences against
the single model are given per frame with a 95% interval from a bootstrap that resamples whole sequences (4 sequences: the interval is coarse, read it as a warning).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.metrics import official_class_ious  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402
from tools.final_eval import INSTR, THIN, TOL_PX, boundary_f1, cl_dice  # noqa: E402

NC = 12
CLASS_NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma", "covered-kidney", "thread", "clamps",
               "suturing-needle", "suction-instrument", "intestine", "ultrasound-probe"]
SYSTEMS = ("single", "tta", "ensemble")


def job(args):
    label_path, preds = args
    gt = read_mask(label_path)
    out = {}
    for name, m in preds.items():
        full = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        cls = official_class_ious(full, gt, NC)
        row = dict(cls={int(c): float(v) for c, v in cls.items()}, frame=(float(np.mean(list(cls.values()))) if cls else None))
        row["bf1"] = boundary_f1(np.isin(full, INSTR), np.isin(gt, INSTR))
        row["cl"] = {n: cl_dice(full == c, gt == c) for c, n in THIN.items() if (gt == c).any()}
        out[name] = row
    return out


def cluster_ci(v, seqs, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    v, s = np.asarray(v, float), np.asarray(seqs)
    uniq = np.unique(s)
    idx = {u: np.where(s == u)[0] for u in uniq}
    m = [float(np.nanmean(v[np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])])) for _ in range(n_boot)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", default="work_v2/failure_detection/masks")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskE")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--published", default="results/task6/final_results.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = [s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",))]
    seqs.sort(key=lambda s: s.index)
    jobs, keys = [], []
    for s in seqs:
        z = np.load(os.path.join(a.masks, f"{s.split}__{s.name}.npz"))
        assert all(k in z.files for k in SYSTEMS), (s.key, z.files)
        assert z["single"].shape[0] == len(s.labels), (s.key, z["single"].shape, len(s.labels))
        arrs = {k: z[k] for k in SYSTEMS}              # load each array once (indexing an NpzFile re-reads the whole array)
        for t, lp in enumerate(s.labels):
            jobs.append((lp, {k: arrs[k][t] for k in SYSTEMS}))
            keys.append(s.key)
    print(f"{len(jobs)} frames, {len(seqs)} sequences", flush=True)
    with mp.Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=4)
    R = {"n_frames": len(res), "systems": {}, "notes": {
        "single": "the segmentation model alone",
        "tta": "same model, 6 passes (flip x scales 0.75/1.0/1.25), probabilities averaged",
        "ensemble": "a DIFFERENT SYSTEM: 4 trained models (3 cross-fitting fold models + final), probabilities averaged; test sequences only because every training sequence is inside some member's training set"}}
    fr = {}
    for name in SYSTEMS:
        f = np.array([np.nan if r[name]["frame"] is None else r[name]["frame"] for r in res])
        fr[name] = f
        per_class = {}
        for c in range(1, NC):
            v = [r[name]["cls"][c] for r in res if c in r[name]["cls"]]
            per_class[CLASS_NAMES[c]] = dict(iou=(float(np.mean(v)) if v else None), n_frames=len(v))
        scored = [v["iou"] for v in per_class.values() if v["iou"] is not None]
        bf = [r[name]["bf1"] for r in res if r[name]["bf1"] is not None]
        cld = {n: float(np.mean([r[name]["cl"][n] for r in res if r[name]["cl"].get(n) is not None] or [np.nan])) for n in THIN.values()}
        R["systems"][name] = dict(mIoU=float(np.nanmean(f)), mcIoU=float(np.mean(scored)), n_classes_scored=len(scored), per_class=per_class,
                                  boundary_f1_instruments=float(np.mean(bf)), boundary_tolerance_px=TOL_PX, cldice=cld,
                                  per_sequence={k: float(np.nanmean(f[np.array(keys) == k])) for k in sorted(set(keys))},
                                  trained_models=4 if name == "ensemble" else 1, forward_passes={"single": 1, "tta": 6, "ensemble": 4}[name])
    R["differences_vs_single"] = {}
    for name in ("tta", "ensemble"):
        d = fr[name] - fr["single"]
        R["differences_vs_single"][name] = dict(mIoU_points=100 * float(np.nanmean(d)), ci95_points=[100 * x for x in cluster_ci(d, keys)],
                                                frames_improved_by_more_than_0_01=float(np.nanmean(d > 0.01)), frames_worse_by_more_than_0_01=float(np.nanmean(d < -0.01)),
                                                mcIoU_points=100 * (R["systems"][name]["mcIoU"] - R["systems"]["single"]["mcIoU"]))
    if os.path.exists(a.published):
        m1 = json.load(open(a.published))["models"]["M1"]["mIoU"]
        R["published_M1_mIoU"] = m1
        R["single_minus_published_M1"] = R["systems"]["single"]["mIoU"] - m1
    json.dump(R, open(os.path.join(a.out, "tta_ensemble.json"), "w"), indent=1)
    for n in SYSTEMS:
        s = R["systems"][n]
        print(f"{n:9s} mIoU {s['mIoU']:.4f}  mcIoU {s['mcIoU']:.4f}  bF1 {s['boundary_f1_instruments']:.3f}  clDice " + "/".join(f"{v:.3f}" for v in s["cldice"].values()))
    print(json.dumps(R["differences_vs_single"], indent=1))
    if "single_minus_published_M1" in R:
        print(f"single re-scored minus published M1: {R['single_minus_published_M1']:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
