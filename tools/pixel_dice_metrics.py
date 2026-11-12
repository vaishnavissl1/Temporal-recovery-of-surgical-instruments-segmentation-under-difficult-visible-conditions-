#!/usr/bin/env python3
"""
Pixel accuracy, Dice, precision and recall of the proposed model and the two trained baselines on the 997 official test frames.

    python -m tools.pixel_dice_metrics --root <endovis2018_cvdataset> --out results/taskR

Nothing is re-run: it reads the argmax masks already written by tools/baseline_eval.py (work_v2/baselines/masks/<arch>.npy, 512x640,
frames in sequence order) and scores them exactly as the official IoU is scored, at 1280x1024 (nearest-neighbour up from the working grid):

  pixel accuracy   share of labelled pixels (background included) whose predicted class equals the ground truth, pooled over all frames
  Dice             2 TP / (2 TP + FP + FN) per class and frame; mean over the classes present in the frame's ground truth, background excluded
                   (the same class rule as the official mIoU), then over frames (mDice); and per class over its frames, then over classes (mcDice)
  precision, recall  TP / (TP + FP) and TP / (TP + FN) per class and frame under the same class rule; precision counts as 0 when the class
                   is absent from the prediction; means taken as for Dice

IoU stays the headline metric (it is what the challenge scores); these are the other standard views of the same masks.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sitr.dataset import discover_sequences  # noqa: E402
from sitr.metrics import _class_counts  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

NC = 12
NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma", "covered-kidney", "thread",
         "clamps", "suturing-needle", "suction-instrument", "small-intestine", "ultrasound-probe"]
MODELS = [("Ours: CAFormer-S18 + FPN", "caformer_surgenet"), ("SegFormer-B0", "segformer_b0"), ("U-Net (classic, from scratch)", "unet_classic")]


def job(args):
    label_path, mask = args
    gt = read_mask(label_path)
    pred = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = gt != 255
    correct = int((pred[valid] == gt[valid]).sum())
    total = int(valid.sum())
    inter, pr, gc = _class_counts(pred, gt, NC)
    out = {}
    for c in range(1, NC):
        g = int(gc[c])
        if g == 0:
            continue
        tp, p = int(inter[c]), int(pr[c])
        out[c] = dict(dice=2.0 * tp / (g + p), precision=(tp / p if p else 0.0), recall=tp / g)
    return correct, total, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", default=os.path.join(ROOT, "work_v2"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "taskR"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = sorted(discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",)), key=lambda s: s.index)
    labels = [l for s in seqs for l in s.labels]
    keys = [s.key for s in seqs for _ in s.labels]
    R = dict(n_frames=len(labels), protocol=__doc__, models=[])
    for label, arch in MODELS:
        masks = np.load(os.path.join(a.work, "baselines", "masks", arch + ".npy"))
        assert len(masks) == len(labels), (arch, len(masks), len(labels))
        with mp.Pool(a.workers) as pool:
            res = pool.map(job, [(labels[i], masks[i]) for i in range(len(labels))], chunksize=4)
        correct = sum(r[0] for r in res)
        total = sum(r[1] for r in res)
        row = dict(label=label, arch=arch, pixel_accuracy=correct / total, n_pixels=total)
        for m in ("dice", "precision", "recall"):
            frame = np.array([np.mean([v[m] for v in r[2].values()]) if r[2] else np.nan for r in res])
            per_class = {}
            for c in range(1, NC):
                v = [r[2][c][m] for r in res if c in r[2]]
                per_class[NAMES[c]] = float(np.mean(v)) if v else None
            sc = [v for v in per_class.values() if v is not None]
            row["m" + m] = float(np.nanmean(frame))               # mean over frames of the per-frame class mean
            row["mc" + m] = float(np.mean(sc))                    # mean over classes of the per-class mean over frames
            row["per_class_" + m] = per_class
            row["per_sequence_m" + m] = {k: float(np.nanmean(frame[np.array(keys) == k])) for k in sorted(set(keys))}
        R["models"].append(row)
        print(f"{label:32s} accuracy {row['pixel_accuracy']:.4f}  mDice {row['mdice']:.4f}  mcDice {row['mcdice']:.4f}  precision {row['mprecision']:.4f}  recall {row['mrecall']:.4f}", flush=True)
    json.dump(R, open(os.path.join(a.out, "pixel_dice.json"), "w"), indent=1)
    print("wrote", os.path.join(a.out, "pixel_dice.json"))


if __name__ == "__main__":
    main()
