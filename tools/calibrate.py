#!/usr/bin/env python3
"""
STEP 1b — temperature scaling of the segmentation posteriors.

Every reliability indicator that reads the posterior — q (eq. 6), entropy
confidence, and the top-1/top-2 margin — inherits whatever miscalibration the
segmentation head has. Neural networks are systematically overconfident, so an
uncalibrated model reports high q on frames it is about to get wrong, which is
exactly the frame the gate must catch.

Temperature scaling (Guo et al., 2017) fits ONE scalar T on held-out data and
replaces p = softmax(z) with p = softmax(z / T). It cannot change which class
wins, so **segmentation accuracy is mathematically unchanged** — mIoU, Dice and
every mask are identical. It only reshapes the confidence, which is precisely
what Stage 2 consumes.

Fitted on the validation sequences, by minimising NLL over a random sample of
pixels (a full-resolution NLL over 2235 frames is unnecessary and slow).

    python -m tools.calibrate --prepared data/prep_256x320_instruments \
        --checkpoint runs/stage1/best.pt --val-seqs 2 5 9 15 \
        --out runs/calibration.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import (FrameDataset, discover_sequences,  # noqa: E402
                          load_prepared, split_sequences)
from sitr.labels import IGNORE_INDEX, ClassMap  # noqa: E402
from sitr.models import build_model  # noqa: E402
from sitr.utils import (amp_enabled, get_device, load_checkpoint,  # noqa: E402
                        set_seed, write_json)


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               bins: int = 15) -> float:
    """ECE: mean |accuracy - confidence| over equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, n = 0.0, conf.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("data source")
    src.add_argument("--prepared", default=None)
    src.add_argument("--root", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--class-map", default=None)
    ap.add_argument("--val-seqs", nargs="+", type=int, default=[2, 5, 9, 15])
    ap.add_argument("--pixels-per-frame", type=int, default=4000,
                    help="random valid pixels sampled per frame")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/calibration.json")
    args = ap.parse_args()
    if not args.prepared and not args.root:
        ap.error("give --prepared or --root")

    set_seed(args.seed)
    device = get_device(args.device)
    use_amp = amp_enabled(device, args.amp)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    meta = ckpt["meta"]
    arch = meta.get("args", {}).get("arch", meta.get("arch", "unet11"))
    cm_path = args.class_map or meta.get("class_map_path")
    class_map = ClassMap.from_json(cm_path)

    model = build_model(int(meta["num_classes"]), arch=arch,
                        pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.prepared:
        seqs, _ = load_prepared(args.prepared, ("train",), meta["task"],
                                (int(meta["height"]), int(meta["width"])))
    else:
        seqs = discover_sequences(args.root, require_labels=True,
                                  apply_repairs=True, splits=("train",))
    _train, val = split_sequences(seqs, args.val_seqs)
    print(f"fitting temperature on {len(val)} held-out sequences: "
          f"{[s.name for s in val]}")

    ds = FrameDataset(val, class_map, meta["task"], int(meta["height"]),
                      int(meta["width"]), augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    rng = np.random.default_rng(args.seed)
    logit_chunks, label_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits, _ = model(images)
            logits = logits.float()
            B, C = logits.shape[0], logits.shape[1]
            lg = logits.permute(0, 2, 3, 1).reshape(B, -1, C)
            lb = masks.reshape(B, -1)
            for b in range(B):
                valid = (lb[b] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                if valid.numel() == 0:
                    continue
                k = min(args.pixels_per_frame, valid.numel())
                pick = valid[torch.from_numpy(
                    rng.choice(valid.numel(), size=k, replace=False)).to(valid.device)]
                logit_chunks.append(lg[b, pick].cpu())
                label_chunks.append(lb[b, pick].cpu())

    if not logit_chunks:
        raise SystemExit("no valid pixels sampled — check the validation split")

    Z = torch.cat(logit_chunks).double()
    Y = torch.cat(label_chunks).long()
    print(f"sampled {Z.shape[0]:,} pixels x {Z.shape[1]} classes")

    def stats(t: float):
        p = torch.softmax(Z / t, dim=1)
        conf, pred = p.max(1)
        nll = float(F.nll_loss(torch.log(p.clamp_min(1e-12)), Y))
        ece = expected_calibration_error(conf.numpy(), (pred == Y).numpy().astype(float))
        return nll, ece

    nll0, ece0 = stats(1.0)

    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        # exp() keeps T strictly positive; a negative temperature would flip
        # the distribution and is never what we want.
        loss = F.cross_entropy(Z / torch.exp(log_t), Y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_t.detach()))
    T = float(np.clip(T, 0.05, 20.0))
    nll1, ece1 = stats(T)

    # argmax is invariant to a positive scalar divide -> accuracy cannot move
    acc0 = float((torch.softmax(Z, 1).argmax(1) == Y).float().mean())
    acc1 = float((torch.softmax(Z / T, 1).argmax(1) == Y).float().mean())
    assert abs(acc0 - acc1) < 1e-9, "temperature changed the argmax — impossible"

    print(f"\n  temperature T        = {T:.4f}  "
          f"({'over' if T > 1 else 'under'}confident before scaling)")
    print(f"  NLL   {nll0:.4f} -> {nll1:.4f}   ({100*(nll0-nll1)/max(nll0,1e-9):+.1f}%)")
    print(f"  ECE   {ece0:.4f} -> {ece1:.4f}   ({100*(ece0-ece1)/max(ece0,1e-9):+.1f}%)")
    print(f"  pixel accuracy {acc0:.6f} -> {acc1:.6f} (unchanged by construction)")

    if ece1 > ece0:
        print("  NOTE: calibration did not improve ECE on this split. Pass "
              "--calibration '' to cache_features to skip it.")

    write_json(args.out, {
        "temperature": T, "fitted_on_val_seqs": args.val_seqs,
        "pixels": int(Z.shape[0]), "checkpoint": os.path.abspath(args.checkpoint),
        "nll_before": nll0, "nll_after": nll1,
        "ece_before": ece0, "ece_after": ece1,
        "pixel_accuracy_before": acc0, "pixel_accuracy_after": acc1})
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
