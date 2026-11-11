#!/usr/bin/env python3
"""
Figure 6a input -- pixel-level reliability of the segmenter's softmax confidence, before and after temperature scaling. GPU, about 5 minutes; run when the GPU is free.

    python -m tools.calibration_pixels --root <endovis2018_cvdataset> --out results/paper/calibration_pixels.json

Model: work_v2/final (the proposed segmenter). Stored temperature: T = 2.687824 (work_v2/final/calibration.json, fitted earlier on a held-out split of the training sequences, NOT on test).
Data: every 4th labelled frame of the 4 official test sequences (about 250 frames), scored at the network resolution (512x640; ground truth by nearest neighbour). Nothing is tuned here: this is a report
of the calibration that the stored temperature produces. For each of 15 equal-width confidence bins, over all pixels of all classes: mean confidence, accuracy, pixel count. ECE = sum over bins of
(count/N) |accuracy - confidence|. Both curves come from the same logits (T = 1 and T = stored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.utils import cap_gpu_memory, get_device, set_seed  # noqa: E402
from tools import failure_detect_collect as C  # noqa: E402
from tools.baseline_eval import load  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

BINS = 15


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="work_v2/final/best.pt")
    ap.add_argument("--calibration", default="work_v2/final/calibration.json")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--out", default="results/paper/calibration_pixels.json")
    a = ap.parse_args()
    set_seed(0)
    dev = get_device("auto")
    cap_gpu_memory(3.3)
    T = float(json.load(open(a.calibration))["temperature"])
    model, _ = load(a.model, "caformer_surgenet", dev)
    seqs = sorted(discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",)), key=lambda s: s.index)
    frames = [(p, l) for s in seqs for p, l in list(zip(s.frames, s.labels))[:: a.every]]
    res = {}
    for name, temp in (("before", 1.0), ("after", T)):
        res[name] = dict(temperature=temp, conf_sum=np.zeros(BINS), correct=np.zeros(BINS), count=np.zeros(BINS))
    nll = {"before": 0.0, "after": 0.0}
    npx = 0
    with torch.no_grad():
        for fp, lp in frames:
            x = C.load_image(fp, dev)
            with torch.autocast("cuda"):
                lo, _ = model(x)
            lo = lo.float()[0]
            h, w = lo.shape[1:]
            gt = cv2.resize(read_mask(lp), (w, h), interpolation=cv2.INTER_NEAREST)
            g = torch.from_numpy(gt.astype(np.int64)).to(dev)
            for name, temp in (("before", 1.0), ("after", T)):
                p = torch.softmax(lo / temp, 0)
                conf, pred = p.max(0)
                ok = (pred == g).float()
                b = torch.clamp((conf * BINS).long(), max=BINS - 1)
                r = res[name]
                r["conf_sum"] += torch.bincount(b.flatten(), weights=conf.flatten(), minlength=BINS).cpu().numpy()
                r["correct"] += torch.bincount(b.flatten(), weights=ok.flatten(), minlength=BINS).cpu().numpy()
                r["count"] += torch.bincount(b.flatten(), minlength=BINS).cpu().numpy()
                nll[name] += float(torch.nn.functional.cross_entropy((lo / temp)[None], g[None], reduction="sum"))
            npx += g.numel()
    out = {"description": __doc__, "n_frames": len(frames), "n_pixels": int(npx), "bins": BINS, "temperature_stored": T}
    for name, r in res.items():
        n = r["count"].sum()
        acc = np.where(r["count"] > 0, r["correct"] / np.maximum(r["count"], 1), np.nan)
        conf = np.where(r["count"] > 0, r["conf_sum"] / np.maximum(r["count"], 1), np.nan)
        ece = float(np.nansum(r["count"] / n * np.abs(acc - conf)))
        out[name] = dict(temperature=r["temperature"], ece=ece, nll_per_pixel=nll[name] / npx, mean_confidence=float(r["conf_sum"].sum() / n), pixel_accuracy=float(r["correct"].sum() / n),
                         bins=[dict(conf=None if np.isnan(c) else float(c), acc=None if np.isnan(c2) else float(c2), count=int(k)) for c, c2, k in zip(conf, acc, r["count"])])
        print(f"{name:7s} T={r['temperature']:.3f}  ECE {ece:.4f}  NLL {nll[name]/npx:.4f}  acc {out[name]['pixel_accuracy']:.4f}  mean conf {out[name]['mean_confidence']:.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
