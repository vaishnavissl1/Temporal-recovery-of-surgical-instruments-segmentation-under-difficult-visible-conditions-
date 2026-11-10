#!/usr/bin/env python3
"""
Segment the RIGHT view of every frame with the same model that segmented the left
view, and cache argmax label + confidence (one .npz per frame).

    python -m tools.cache_right_view --work work_v2 --root <endovis2018_cvdataset> \
        --out work_v2/right_view

Honesty: out-of-fold (training) sequences use the model of THEIR fold, which never
saw them, with that fold's out-of-fold calibration temperature (the mean of the
other folds, the rule run_v2 used). Test sequences use the final model. The right
view is never labelled, so no label is involved at any point.

Preprocessing is identical to the left view: BGR->RGB, cv2.resize INTER_AREA to
the working size, /255, ImageNet mean/std. Output is on the RIGHT working grid;
sitr.stereo.right_to_left_grid pulls it onto the left grid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import IMAGENET_MEAN, IMAGENET_STD, discover_sequences  # noqa: E402
from sitr.models import build_model  # noqa: E402
from sitr.stereo import right_path  # noqa: E402
from sitr.utils import get_device, keep_awake, load_checkpoint  # noqa: E402


def load_model(path, device):
    ck = load_checkpoint(path, map_location=device)
    meta = ck["meta"]
    m = build_model(int(meta["num_classes"]), arch=meta["arch"], pretrained=False).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(memory_format=torch.channels_last), meta


@torch.no_grad()
def segment(model, device, paths, h, w, temperature, batch=2):
    for i in range(0, len(paths), batch):
        xs = []
        for p in paths[i:i + batch]:
            img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            xs.append(((img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(xs)).to(device).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(x)
        conf, lab = torch.softmax(logits.float() / temperature, 1).max(1)
        for c, l in zip(conf.cpu().numpy(), lab.cpu().numpy()):
            yield l.astype(np.uint8), np.clip(c * 255.0, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="work_v2 (holds crossfit/, final/, folds.json)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    keep_awake()
    device = get_device(a.device)
    folds = json.load(open(os.path.join(a.work, "folds.json")))["folds"]
    temps = [json.load(open(os.path.join(a.work, "crossfit", f"fold{j}", "calibration.json")))["temperature"]
             for j in range(len(folds))]
    seqs = discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=tuple(a.splits))
    t0, done, total = time.time(), 0, sum(len(s) for s in seqs)
    cache = {}
    for s in sorted(seqs, key=lambda s: (s.split != "train", s.index)):
        if s.split == "train":
            fi = next(i for i, f in enumerate(folds) if s.key in f)
            ck_path = os.path.join(a.work, "crossfit", f"fold{fi}", "best.pt")
            T = float(np.mean([t for j, t in enumerate(temps) if j != fi]))
        else:
            ck_path = os.path.join(a.work, "final", "best.pt")
            T = None
        if ck_path not in cache:
            cache = {ck_path: load_model(ck_path, device)}           # keep one model in memory
        model, meta = cache[ck_path]
        if T is None:
            T = float(meta["temperature"])
        sd = os.path.join(a.out, f"{s.split}_{s.name}")
        os.makedirs(sd, exist_ok=True)
        todo = [(i, p) for i, p in enumerate(s.frames)
                if not os.path.exists(os.path.join(sd, os.path.splitext(os.path.basename(p))[0] + ".npz"))]
        if todo:
            rp = [right_path(p) for _, p in todo]
            for (i, p), (lab, conf) in zip(todo, segment(model, device, rp, a.height, a.width, T)):
                stem = os.path.splitext(os.path.basename(p))[0]
                tmp = os.path.join(sd, stem + ".tmp.npz")
                np.savez_compressed(tmp, label=lab, conf=conf)
                os.replace(tmp, os.path.join(sd, stem + ".npz"))
        done += len(s)
        print(f"  {s.key:<12} T={T:.3f}  {done}/{total} frames  {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"finished -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
