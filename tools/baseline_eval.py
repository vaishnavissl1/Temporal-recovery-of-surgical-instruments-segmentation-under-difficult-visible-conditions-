#!/usr/bin/env python3
"""
TASK P.1 -- evaluate the baselines and the proposed model under one identical protocol.

    python -m tools.baseline_eval --work work_v2 --root <endovis2018_cvdataset> --out results/taskP

Models: the proposed model (CAFormer-S18 + FPN + full-resolution branch, SurgeNetXL-initialised; work_v2/final), and the baselines trained by tools.train_stage1 with the IDENTICAL recipe
(work_v2/baselines/<arch>: 15 training sequences, 512x640 input, 20 epochs, batch 1 x accumulation 8, AdamW lr 3e-4 with 0.1x on the pretrained encoder, weight decay 1e-4, warmup 3% then
cosine to 1%, EMA 0.999, loss CE + Dice + 0.5 Tversky, inverse-sqrt class weights, same augmentations, same seed 42): a classic U-Net (from scratch), DeepLabV3+ with a ResNet-18 encoder
(ImageNet-pretrained, the backbone of STswinCL), SegFormer-B0 (ImageNet-pretrained MiT-B0). Where a model has a pretrained encoder, its pretraining differs by architecture and that is the point of
each baseline; the proposed encoder is SurgeNetXL (self-supervised on surgical video), the others ImageNet-1k or none. This difference is stated, not removed.

Evaluation is identical for all: the model's argmax at 512x640, nearest-neighbour up to 1280x1024, the official rule (class 0 never scored, a class counted only if in the ground truth), on the 997
official test frames. Reported: mIoU, mcIoU, per-class IoU, boundary F1 (3 px), clDice on shaft/thread/needle, parameters, GFLOPs (torch flop counter, 2 x multiply-adds, one 512x640 forward),
and measured ms/frame (batch 1, image on the GPU, fp16 autocast, GPU synchronised, mean of 40 frames after warm-up). Differences from the proposed model are given per frame with a 95% interval from a
bootstrap over the 4 test sequences (coarse; a warning, not a test). Published numbers are not placed here.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.models import build_model  # noqa: E402
from sitr.utils import cap_gpu_memory, get_device, load_checkpoint, set_seed  # noqa: E402
from tools import failure_detect_collect as C  # noqa: E402
from tools.tta_ensemble_eval import CLASS_NAMES, NC, cluster_ci, job  # noqa: E402
from tools.final_eval import THIN  # noqa: E402

MODELS = [("Ours: CAFormer-S18 + FPN", "work_v2/final/best.pt", "caformer_surgenet"),
          ("DeepLabV3+ (ResNet-18)", "work_v2/baselines/deeplabv3p_r18/best.pt", "deeplabv3p_r18"),
          ("SegFormer-B0", "work_v2/baselines/segformer_b0/best.pt", "segformer_b0"),
          ("U-Net (classic, from scratch)", "work_v2/baselines/unet_classic/best.pt", "unet_classic")]


def load(path, arch, dev):
    ck = load_checkpoint(path, map_location=dev)
    m = build_model(int(ck["meta"]["num_classes"]), arch=arch, pretrained=False).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(memory_format=torch.channels_last), ck["meta"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work_v2")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskP")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    set_seed(0)
    dev = get_device("auto")
    cap_gpu_memory(3.3)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.work, "baselines", "masks"), exist_ok=True)
    seqs = sorted(discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",)), key=lambda s: s.index)
    frames = [(s.key, p, l) for s in seqs for p, l in zip(s.frames, s.labels)]
    keys = [k for k, _, _ in frames]
    R = {"protocol": __doc__, "n_frames": len(frames), "models": [], "unavailable": []}
    from torch.utils.flop_counter import FlopCounterMode
    for label, path, arch in MODELS:
        if not os.path.exists(path):
            R["unavailable"].append(dict(label=label, reason=f"{path} does not exist (training did not finish)"))
            print(f"  {label}: MISSING ({path})")
            continue
        model, meta = load(path, arch, dev)
        mpath = os.path.join(a.work, "baselines", "masks", arch + ".npy")
        if os.path.exists(mpath):
            masks = np.load(mpath)
        else:
            out = []
            with torch.no_grad():
                for k, fp, lp in frames:
                    x = C.load_image(fp, dev)
                    with torch.autocast("cuda"):
                        lo, _ = model(x)
                    out.append(lo.float().argmax(1)[0].cpu().numpy().astype(np.uint8))
            masks = np.stack(out)
            np.save(mpath, masks)
        # cost
        x = C.load_image(frames[0][1], dev)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        gflops = fc.get_total_flops() / 1e9
        pick = np.random.default_rng(0).choice(len(frames), 40, replace=False)
        xs = [C.load_image(frames[i][1], dev) for i in pick]
        ts = []
        with torch.no_grad():
            for i, xi in enumerate(xs + xs[:0]):
                if i < 3:
                    with torch.autocast("cuda"):
                        model(xi)
            for xi in xs:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast("cuda"):
                    model(xi)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        R["models"].append(dict(label=label, arch=arch, checkpoint=path, params_millions=params, gflops=gflops, ms_per_frame=float(np.mean(ts)), ms_p95=float(np.percentile(ts, 95)),
                                epochs=int(meta.get("epoch", -1)) if isinstance(meta.get("epoch", None), (int, float)) else None, _masks=mpath))
        print(f"  {label:34s} params {params:6.2f} M  GFLOPs {gflops:6.1f}  {np.mean(ts):6.1f} ms", flush=True)
        del model
        torch.cuda.empty_cache()
    # official metrics, CPU
    for m in R["models"]:
        masks = np.load(m["_masks"])
        jobs = [(l, {"x": masks[i]}) for i, (_, _, l) in enumerate(frames)]
        with mp.Pool(a.workers) as pool:
            res = pool.map(job, jobs, chunksize=4)
        f = np.array([np.nan if r["x"]["frame"] is None else r["x"]["frame"] for r in res])
        pc = {}
        for c in range(1, NC):
            v = [r["x"]["cls"][c] for r in res if c in r["x"]["cls"]]
            pc[CLASS_NAMES[c]] = dict(iou=(float(np.mean(v)) if v else None), n_frames=len(v))
        sc = [v["iou"] for v in pc.values() if v["iou"] is not None]
        bf = [r["x"]["bf1"] for r in res if r["x"]["bf1"] is not None]
        m.update(mIoU=float(np.nanmean(f)), mcIoU=float(np.mean(sc)), n_classes_scored=len(sc), per_class=pc, boundary_f1_instruments=float(np.mean(bf)),
                 cldice={n: float(np.mean([r["x"]["cl"][n] for r in res if r["x"]["cl"].get(n) is not None] or [np.nan])) for n in THIN.values()},
                 per_sequence={k: float(np.nanmean(f[np.array(keys) == k])) for k in sorted(set(keys))}, _frame=f.tolist())
    ours = next((m for m in R["models"] if m["label"].startswith("Ours")), None)
    for m in R["models"]:
        if ours is not None and m is not ours:
            d = np.array(m["_frame"]) - np.array(ours["_frame"])
            m["difference_from_ours_mIoU_points"] = 100 * float(np.nanmean(d))
            m["difference_from_ours_ci95_points"] = [100 * x for x in cluster_ci(d, keys)]
    for m in R["models"]:
        m.pop("_masks", None)
        m.pop("_frame", None)
    json.dump(R, open(os.path.join(a.out, "baselines.json"), "w"), indent=1)
    for m in R["models"]:
        print(f"{m['label']:34s} mIoU {m['mIoU']:.4f}  mcIoU {m['mcIoU']:.4f}  bF1 {m['boundary_f1_instruments']:.3f}  clDice " + "/".join(f"{v:.3f}" for v in m["cldice"].values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
