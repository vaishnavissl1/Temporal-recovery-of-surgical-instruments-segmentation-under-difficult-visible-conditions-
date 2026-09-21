#!/usr/bin/env python3
"""
STEP 2a — the bridge between Stage 1 and Stage 2.

With the base model FROZEN, run one pass over every sequence in temporal order
and cache, per frame, everything the reliability estimator needs:

    q      mean max-posterior over the instrument foreground   eq. (6)
    a      |Omega^fg_t|                                        eq. (6)
    E      masked-average-pooled decoder embedding             eq. (9)
    S      predicted label map (downsampled, for T_t)          eq. (5)
    r*     IoU(S_t, Y_t) — the training target                 eq. (13)
    ent    entropy-based confidence      ] extended indicators, all from the
    margin top-1 minus top-2            ] SAME forward pass, so they cost
    bnd    compactness (1 - perimeter/area) ] essentially nothing on top of
    frag   largest-connected-component share ] the segmentation itself

Why cache at all: T_t and A_t depend on which past frame was judged reliable,
which depends on the estimator being trained. Re-running the backbone inside
that loop would be pure waste, because a frozen model's outputs cannot change.
One pass turns Stage 2 training into arithmetic over a few thousand cached rows
and is what makes hundreds of scheduled-sampling epochs affordable.

    python -m tools.cache_features --prepared data/prep_256x320_instruments \
        --checkpoint runs/stage1/best.pt --splits train test --out runs/cache
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import (SequenceDataset, discover_sequences,  # noqa: E402
                          load_prepared)
from sitr.labels import ClassMap  # noqa: E402
from sitr.metrics import foreground_area, instrument_iou  # noqa: E402
from sitr.models import build_model  # noqa: E402
from sitr.reliability import SequenceCache  # noqa: E402
from sitr.utils import (amp_enabled, get_device, load_checkpoint,  # noqa: E402
                        set_seed, write_json)


def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("data source (give --prepared or --root)")
    src.add_argument("--prepared", default=None)
    src.add_argument("--root", default=None)
    ap.add_argument("--splits", nargs="+", default=["train"],
                    choices=["train", "test"])
    ap.add_argument("--checkpoint", required=True, help="Stage 1 checkpoint")
    ap.add_argument("--out", default="runs/cache")
    ap.add_argument("--class-map", default=None)
    ap.add_argument("--calibration", default=None,
                    help="calibration.json from tools/calibrate.py; the "
                         "temperature is applied to the logits before the "
                         "posterior-based indicators are computed")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mask-scale", type=int, default=2,
                    help="downsample factor for the cached masks used by T_t")
    ap.add_argument("--store-gray", action="store_true", default=True,
                    help="cache grayscale frames (only the flow ablation needs them)")
    ap.add_argument("--no-store-gray", dest="store_gray", action="store_false")
    ap.add_argument("--empty-both-iou", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the calibration temperature stored in the "
                         "checkpoint (1.0 disables scaling)")
    ap.add_argument("--tta", action="store_true",
                    help="horizontal-flip TTA; doubles c_seg, off by default")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--channels-last", action="store_true", default=True)
    ap.add_argument("--no-channels-last", dest="channels_last",
                    action="store_false")
    ap.add_argument("--overwrite", action="store_true",
                    help="clear pre-existing caches in --out first")
    args = ap.parse_args()
    if not args.prepared and not args.root:
        ap.error("give --prepared (recommended) or --root")
    return args


def compactness_and_fragmentation(pred: np.ndarray, fg: np.ndarray):
    """Two shape-based failure detectors the report's four indicators miss.

    A correct instrument mask is compact and (mostly) one piece. Under occlusion
    and blur the prediction shatters: perimeter-to-area explodes and the largest
    component stops dominating. Both are O(HW) on arrays already in memory.
    """
    n_fg = int(fg.sum())
    if n_fg == 0:
        return 0.0, 0.0

    # class boundaries, 4-neighbourhood
    b = np.zeros_like(fg)
    d_v = pred[1:, :] != pred[:-1, :]
    d_h = pred[:, 1:] != pred[:, :-1]
    b[1:, :] |= d_v
    b[:-1, :] |= d_v
    b[:, 1:] |= d_h
    b[:, :-1] |= d_h
    perimeter_ratio = float((b & fg).sum()) / float(n_fg)
    compact = float(np.clip(1.0 - perimeter_ratio, 0.0, 1.0))

    n_lab, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    if n_lab <= 1:
        return compact, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    frag = float(sizes.max()) / float(max(sizes.sum(), 1))
    return compact, frag


@torch.no_grad()
def cache_sequence(model, seq, class_map, meta, args, device, use_amp):
    ds = SequenceDataset(seq, class_map, meta["task"], meta["height"], meta["width"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    num_classes = int(meta["num_classes"])
    instrument_ids = [int(c) for c in meta["instrument_ids"]]
    sh = max(1, meta["height"] // args.mask_scale)
    sw = max(1, meta["width"] // args.mask_scale)
    log_c = math.log(max(num_classes, 2))
    temp = float(getattr(args, "_temperature", 1.0))

    out = {k: [] for k in ("q", "area", "emb", "masks", "r", "gray",
                           "ent", "margin", "bnd", "frag")}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if getattr(args, "channels_last", False) and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", enabled=use_amp):
            logits, feats = model(images)
        T = float(getattr(args, "_temperature", 1.0))
        probs = torch.softmax(logits.float() / T, dim=1)          # eq. (4)
        if args.tta:
            flip_logits, _ = model(torch.flip(images, dims=[3]))
            probs = 0.5 * (probs + torch.flip(
                torch.softmax(flip_logits.float() / T, 1), dims=[3]))

        conf, pred = probs.max(dim=1)                             # eq. (5)
        feats = feats.float()

        # Omega^fg = pixels predicted as an INSTRUMENT class. EndoVis 2018
        # annotates anatomy too, so "not background" is not "instrument".
        inst = torch.zeros_like(pred, dtype=torch.bool)
        for c in instrument_ids:
            inst |= (pred == c)
        fgm = inst.to(probs.dtype)
        denom = fgm.sum(dim=(1, 2)).clamp_min(1e-6)               # eq. (6) eps

        q = (conf * fgm).sum(dim=(1, 2)) / denom
        emb = (feats * fgm.unsqueeze(1)).sum(dim=(2, 3)) / denom.unsqueeze(1)

        # normalised entropy -> confidence, and top1-top2 margin
        ent_map = -(probs * probs.clamp_min(1e-8).log()).sum(1) / log_c
        ent_conf = 1.0 - (ent_map * fgm).sum(dim=(1, 2)) / denom
        top2 = probs.topk(2, dim=1).values
        margin = ((top2[:, 0] - top2[:, 1]) * fgm).sum(dim=(1, 2)) / denom

        pred_np = pred.cpu().numpy().astype(np.uint8)
        inst_np = inst.cpu().numpy()
        q_np = q.cpu().numpy().astype(np.float32)
        emb_np = emb.cpu().numpy().astype(np.float32)
        ent_np = ent_conf.cpu().numpy().astype(np.float32)
        mar_np = margin.cpu().numpy().astype(np.float32)
        gt_np = batch["mask"].numpy().astype(np.uint8)
        imgs_np = images.cpu().numpy()

        for i in range(pred_np.shape[0]):
            p, fg = pred_np[i], inst_np[i]
            a = foreground_area(p, instrument_ids)
            empty = a <= 0
            # eq. (10): an empty predicted foreground makes every
            # foreground-conditioned indicator undefined -> minimising values
            out["q"].append(0.0 if empty else float(q_np[i]))
            out["ent"].append(0.0 if empty else float(ent_np[i]))
            out["margin"].append(0.0 if empty else float(mar_np[i]))
            out["emb"].append(np.zeros_like(emb_np[i]) if empty else emb_np[i])
            cmp_, frg = (0.0, 0.0) if empty else compactness_and_fragmentation(p, fg)
            out["bnd"].append(cmp_)
            out["frag"].append(frg)
            out["area"].append(int(a))
            out["r"].append(instrument_iou(p, gt_np[i], num_classes,
                                           instrument_ids,
                                           empty_value=args.empty_both_iou))
            out["masks"].append(cv2.resize(p, (sw, sh),
                                           interpolation=cv2.INTER_NEAREST))
            if args.store_gray:
                x = imgs_np[i].transpose(1, 2, 0)
                x = x * np.array([0.229, 0.224, 0.225], np.float32) + \
                    np.array([0.485, 0.456, 0.406], np.float32)
                g = cv2.cvtColor(np.clip(x * 255.0, 0, 255).astype(np.uint8),
                                 cv2.COLOR_RGB2GRAY)
                out["gray"].append(cv2.resize(g, (sw, sh),
                                              interpolation=cv2.INTER_AREA))

    f32 = lambda k: np.asarray(out[k], dtype=np.float32)          # noqa: E731
    return SequenceCache(
        name=seq.key, q=f32("q"),
        area=np.asarray(out["area"], dtype=np.int64),
        emb=np.stack(out["emb"]).astype(np.float32),
        masks=np.stack(out["masks"]).astype(np.uint8),
        r_star=f32("r"),
        gray=np.stack(out["gray"]).astype(np.uint8) if out["gray"] else None,
        ent=f32("ent"), margin=f32("margin"), bnd=f32("bnd"), frag=f32("frag"),
        num_classes=num_classes, instrument_ids=tuple(instrument_ids))


def main() -> int:
    args = build_args()
    set_seed(args.seed)
    device = get_device(args.device)
    use_amp = amp_enabled(device, args.amp)
    os.makedirs(args.out, exist_ok=True)

    # A cache directory holding output from an earlier run is a silent hazard:
    # stale .npz files are loaded alongside the new ones and the same sequence
    # can appear twice under two naming schemes, contaminating the split.
    stale = sorted(f for f in os.listdir(args.out) if f.endswith(".npz"))
    if stale:
        if not args.overwrite:
            raise SystemExit(
                f"{args.out!r} already contains {len(stale)} cached sequences "
                f"(e.g. {stale[0]}). They would be mixed with this run's output "
                "and can duplicate sequences across the train/val split. "
                "Re-run with --overwrite, or choose an empty --out.")
        for f in stale:
            os.remove(os.path.join(args.out, f))
        print(f"--overwrite: removed {len(stale)} stale cache files")

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    meta = ckpt["meta"]
    arch = meta.get("args", {}).get("arch", meta.get("arch", "unet11"))
    cm_path = args.class_map or meta.get("class_map_path")
    if not (cm_path and os.path.isfile(cm_path)):
        raise FileNotFoundError(
            f"class map not found ({cm_path!r}); pass --class-map explicitly")
    class_map = ClassMap.from_json(cm_path)

    args._temperature = 1.0
    if args.calibration:
        import json
        with open(args.calibration) as f:
            calib = json.load(f)
        args._temperature = float(calib["temperature"])
        print(f"applying temperature T={args._temperature:.4f} from "
              f"{args.calibration} (argmax, and therefore every mask and IoU, "
              "is unchanged)")

    model = build_model(int(meta["num_classes"]), arch=arch,
                        pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    for p in model.parameters():        # frozen from here on
        p.requires_grad_(False)

    args._temperature = (args.temperature if args.temperature is not None
                         else float(meta.get("temperature", 1.0)))
    print(f"calibration temperature: {args._temperature:.3f}")
    print(f"device={device} amp={use_amp} arch={arch} task={meta['task']} "
          f"classes={meta['num_classes']} instrument_ids={meta['instrument_ids']}"
          f"{' TTA' if args.tta else ''}")

    if args.prepared:
        seqs, _ = load_prepared(args.prepared, tuple(args.splits), meta["task"],
                                (int(meta["height"]), int(meta["width"])))
    else:
        seqs = discover_sequences(args.root, require_labels=True,
                                  apply_repairs=True, splits=tuple(args.splits))

    index, t0 = [], time.time()
    for s in seqs:
        cache = cache_sequence(model, s, class_map, meta, args, device, use_amp)
        path = os.path.join(args.out, f"{s.split}__{s.name}.npz")
        cache.save(path)
        index.append({"name": s.name, "key": s.key, "split": s.split,
                      "index": s.index, "frames": len(cache),
                      "path": os.path.abspath(path),
                      "mean_r_star": float(cache.r_star.mean()),
                      "frac_r_star_below_0.5": float((cache.r_star < 0.5).mean()),
                      "frac_empty_foreground": float((cache.area == 0).mean())})
        print(f"  {s.key:<16} frames={len(cache):<5} "
              f"mean r*={cache.r_star.mean():.4f}  "
              f"hard(r*<0.5)={float((cache.r_star < 0.5).mean()):.3f}  "
              f"-> {os.path.basename(path)}")

    write_json(os.path.join(args.out, "cache_index.json"), {
        "sequences": index,
        "meta": {k: meta[k] for k in ("task", "num_classes", "target_names",
                                      "instrument_ids", "height", "width")},
        "arch": arch, "extended_indicators": True, "tta": bool(args.tta),
        "temperature": float(args._temperature),
        "temperature": float(args._temperature),
        "calibration": os.path.abspath(args.calibration) if args.calibration else None,
        "checkpoint": os.path.abspath(args.checkpoint),
        "class_map_path": os.path.abspath(cm_path),
        "mask_scale": args.mask_scale, "store_gray": bool(args.store_gray),
        "seconds": round(time.time() - t0, 1)})
    print(f"\nCached {len(index)} sequences in {time.time()-t0:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
