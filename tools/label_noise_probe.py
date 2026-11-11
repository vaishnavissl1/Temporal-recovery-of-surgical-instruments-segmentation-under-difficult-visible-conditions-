#!/usr/bin/env python3
"""
TASK L -- label-noise probe, stage 1: rank the training frames by OUT-OF-FOLD loss and write the top frames for inspection.

    python -m tools.label_noise_probe --work work_v2 --root <endovis2018_cvdataset> --out results/taskL --top 20

Every training sequence is predicted by the cross-fitting model whose fold EXCLUDED it (the same models and folds as the gate's out-of-fold data), so a
high loss means "this frame disagrees with a model that never saw it", which is what a label error and a hard frame both look like. The loss is the mean
per-pixel cross-entropy of the raw (temperature 1) softmax against the ground-truth class map at 512x640 (labels nearest-neighbour downsampled; pixels
with no valid label, if any, are ignored).

Writes results/taskL/oof_loss_ranking.csv (every training frame: sequence, frame, loss, official frame IoU at the working grid, instrument IoU) and, for the
top frames, a three-panel image per frame (photograph | ground truth | model prediction) into work_v2/label_probe/ . The images contain EndoVis frames, so they
stay on this machine: they are not committed and not published (dataset licence; see NEEDS OWNER in RUNLOG.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.metrics import official_frame_iou  # noqa: E402
from sitr.utils import cap_gpu_memory, get_device, set_seed  # noqa: E402
from tools import failure_detect_collect as C  # noqa: E402
from tools.cache_right_view import load_model  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

PAL = np.array([(0, 0, 0), (0, 255, 0), (0, 255, 255), (125, 255, 12), (255, 55, 0), (24, 55, 125), (187, 155, 25),
                (0, 255, 125), (255, 255, 125), (123, 15, 175), (124, 155, 5), (12, 255, 141)], np.uint8)


def overlay(img_bgr, cls):
    col = PAL[np.clip(cls, 0, 11)][..., ::-1]
    out = (0.55 * img_bgr + 0.45 * col).astype(np.uint8)
    out[cls == 0] = img_bgr[cls == 0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work_v2")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskL")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--img-dir", default="work_v2/label_probe")
    a = ap.parse_args()
    set_seed(0)
    dev = get_device("auto")
    cap_gpu_memory(3.3)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.img_dir, exist_ok=True)
    folds = json.load(open(os.path.join(a.work, "folds.json")))["folds"]
    seqs = {s.key: s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("train",))}
    rows = []
    for i, held in enumerate(folds):
        model, _ = load_model(os.path.join(a.work, "crossfit", f"fold{i}", "best.pt"), dev)
        for key in held:
            s = seqs[key]
            for t, (fp, lp) in enumerate(zip(s.frames, s.labels)):
                x = C.load_image(fp, dev)
                with torch.no_grad(), torch.autocast("cuda"):
                    logits, _ = model(x)
                logits = logits.float()
                gt = cv2.resize(read_mask(lp), (C.W, C.H), interpolation=cv2.INTER_NEAREST)
                g = torch.from_numpy(gt.astype(np.int64)).to(dev)[None]
                valid = g < C.NC
                ce = F.cross_entropy(logits, torch.where(valid, g, torch.zeros_like(g)), reduction="none")
                loss = float(ce[valid].mean())
                pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                off = official_frame_iou(pred, gt, C.NC)
                from sitr.metrics import instrument_iou
                rows.append(dict(sequence=key, frame=t, file=os.path.basename(fp), loss=loss, official_iou=(np.nan if off is None else off),
                                 instrument_iou=instrument_iou(pred, gt, C.NC, C.INSTR)))
            print(f"  fold {i} {key}: {len(s.frames)} frames", flush=True)
        del model
        torch.cuda.empty_cache()
    rows.sort(key=lambda r: -r["loss"])
    with open(os.path.join(a.out, "oof_loss_ranking.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    top = rows[:a.top]
    mdl = {}
    for r in top:
        s = seqs[r["sequence"]]
        fi = next(i for i, p in enumerate(s.frames) if os.path.basename(p) == r["file"])
        img = cv2.resize(cv2.imread(s.frames[fi]), (C.W, C.H), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(read_mask(s.labels[fi]), (C.W, C.H), interpolation=cv2.INTER_NEAREST)
        fold = next(i for i, h in enumerate(folds) if r["sequence"] in h)
        mdl.setdefault(fold, load_model(os.path.join(a.work, "crossfit", f"fold{fold}", "best.pt"), dev)[0])
        with torch.no_grad(), torch.autocast("cuda"):
            pred = mdl[fold](C.load_image(s.frames[fi], dev))[0].float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        panel = np.concatenate([img, overlay(img, gt), overlay(img, pred)], 1)
        name = f"{r['sequence'].replace('/', '_')}_{r['file'].replace('.png', '')}_loss{r['loss']:.2f}.png"
        cv2.imwrite(os.path.join(a.img_dir, name), panel)
        r["image"] = name
    json.dump(dict(n_frames=len(rows), top=top, loss_definition="mean per-pixel cross-entropy, raw softmax, out-of-fold model, 512x640"),
              open(os.path.join(a.out, "top_frames.json"), "w"), indent=1)
    for r in top:
        print(f"{r['sequence']:<12} {r['file']:<14} loss {r['loss']:.3f}  official IoU {r['official_iou']:.3f}  instrument IoU {r['instrument_iou']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
