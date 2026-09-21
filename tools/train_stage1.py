#!/usr/bin/env python3
"""
STAGE 1 — train the base segmentation model f_seg (report Sec. IV-B, V-C).

No temporal component at all. This is the frame-wise baseline ("Baseline A",
Sec. V-J.1) against which selective temporal recovery is later measured, and the
frozen backbone every later stage builds on.

    python -m tools.train_stage1 \
        --prepared data/prepared_256x320_instruments \
        --class-map runs/class_map.json --task instruments \
        --val-seqs 2 5 9 15 --out runs/stage1

Defaults are tuned for this dataset (15 sequences, 2235 frames, 1280x1024
downscaled to 256x320) rather than copied from a generic recipe; see README
"Hyperparameters and why" for the reasoning behind each one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import (FrameDataset, class_pixel_counts,  # noqa: E402
                          discover_sequences, inverse_sqrt_weights,
                          load_prepared, save_manifest, split_sequences)
from sitr.labels import ClassMap, build_class_map  # noqa: E402
from sitr.calibration import calibration_report, fit_temperature  # noqa: E402
from sitr.losses import SegLoss  # noqa: E402
from sitr.metrics import ConfusionMatrix, frame_iou  # noqa: E402
from sitr.models import (build_model, count_parameters,  # noqa: E402
                         encoder_param_names, predict_with_tta)
from sitr.utils import (AverageMeter, amp_enabled, cosine_lr, get_device,  # noqa: E402
                        save_checkpoint, set_seed, write_json)


def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("data source (give --prepared or --root)")
    src.add_argument("--prepared", default=None,
                     help="cache from tools/prepare_dataset.py (much faster)")
    src.add_argument("--root", default=None, help="raw dataset root")
    ap.add_argument("--class-map", default=None)
    ap.add_argument("--task", default="instruments",
                    choices=["binary", "parts", "instruments", "full"])
    ap.add_argument("--instrument-classes", nargs="*", default=None)
    ap.add_argument("--val-seqs", nargs="+", type=int, default=[2, 5, 9, 15],
                    help="held-out sequence indices (sequence-level split)")

    ap.add_argument("--arch", default="resnet34_unet",
                    choices=["resnet34_unet", "convnext_unet", "unet11"],
                    help="unet11 = the report's TernausNet baseline, for "
                         "comparability; the others are stronger")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr-scale", type=float, default=0.1,
                    help="the pretrained VGG encoder is fine-tuned at lr*scale")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--min-lr-factor", type=float, default=0.01)
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--lambda-dice", type=float, default=1.0)
    ap.add_argument("--lambda-tversky", type=float, default=0.5,
                    help="Focal-Tversky term for thin structures; 0 gives "
                         "exactly the report's eq. (29)")
    ap.add_argument("--tta", action="store_true",
                    help="flip TTA during validation only; doubles c_seg")
    ap.add_argument("--class-weights", default="inv_sqrt",
                    choices=["none", "inv_sqrt"],
                    help="CE class weighting; Dice already counters imbalance")
    ap.add_argument("--clip-grad", type=float, default=5.0)
    ap.add_argument("--early-stop", type=int, default=15,
                    help="stop after this many epochs without a new best (0=off)")

    ap.add_argument("--channels-last", action="store_true", default=True,
                    help="NHWC memory format; ~1.2-1.4x faster with AMP on "
                         "Ampere and newer, no effect on results")
    ap.add_argument("--no-channels-last", dest="channels_last",
                    action="store_false")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (Linux/WSL; flaky on Windows)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/stage1")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--limit-frames", type=int, default=0, help="debug only")
    args = ap.parse_args()

    if not args.prepared and not args.root:
        ap.error("give --prepared (recommended) or --root")
    return args


@torch.no_grad()
def evaluate(model, loader, device, num_classes, names, instrument_ids, use_amp,
             tta: bool = False):
    model.eval()
    cm = ConfusionMatrix(num_classes)
    union, gt_only, inst = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].numpy().astype(np.uint8)
        with torch.autocast("cuda", enabled=use_amp):
            if tta:
                probs, _ = predict_with_tta(model, images)
            else:
                logits, _ = model(images)
                probs = logits.float()
        preds = probs.float().argmax(1).cpu().numpy().astype(np.uint8)
        for p, g in zip(preds, masks):
            cm.update(p, g)
            union.append(frame_iou(p, g, num_classes, present="union"))
            gt_only.append(frame_iou(p, g, num_classes, present="gt"))
            inst.append(frame_iou(p, g, num_classes, class_ids=instrument_ids,
                                  present="union"))
    out = cm.summary(names)
    out["challenge_IoU_union"] = float(np.mean(union))
    out["challenge_IoU_gt"] = float(np.mean(gt_only))
    out["instrument_IoU_per_frame"] = float(np.mean(inst))
    return out


def main() -> int:
    args = build_args()
    set_seed(args.seed)
    device = get_device(args.device)
    use_amp = amp_enabled(device, args.amp)
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------ class map
    cm_path = args.class_map
    if cm_path and os.path.isfile(cm_path):
        class_map = ClassMap.from_json(cm_path)
    else:
        src = args.root or args.prepared
        class_map = build_class_map(src, args.instrument_classes)
        cm_path = os.path.join(args.out, "class_map.json")
        class_map.to_json(cm_path)

    # --------------------------------------------------------------- data
    if args.prepared:
        seqs, pmeta = load_prepared(args.prepared, ("train",), args.task,
                                    (args.height, args.width))
        print(f"prepared cache: {args.prepared} (task={pmeta['task']}, "
              f"{pmeta['height']}x{pmeta['width']})")
    else:
        seqs = discover_sequences(args.root, require_labels=True,
                                  apply_repairs=True, splits=("train",))
        print(f"raw dataset: {args.root}  "
              f"(repaired frames: {sum(len(s.repaired) for s in seqs)})")
        print("  tip: tools/prepare_dataset.py makes epochs ~10x faster")

    train_seqs, val_seqs = split_sequences(seqs, args.val_seqs)
    print(f"device={device}  amp={use_amp}")
    print(f"train seqs ({len(train_seqs)}): {[s.name for s in train_seqs]}")
    print(f"val   seqs ({len(val_seqs)}): {[s.name for s in val_seqs]}")

    train_ds = FrameDataset(train_seqs, class_map, args.task, args.height,
                            args.width, augment=True, seed=args.seed)
    val_ds = FrameDataset(val_seqs, class_map, args.task, args.height,
                          args.width, augment=False)
    if args.limit_frames:
        train_ds.items = train_ds.items[:args.limit_frames]
        val_ds.items = val_ds.items[:args.limit_frames]

    num_classes, names = train_ds.num_classes, train_ds.target_names
    instrument_ids = train_ds.instrument_ids
    print(f"task={args.task}  classes={num_classes} {names}")
    print(f"instrument ids={instrument_ids}")
    print(f"frames: train={len(train_ds)}  val={len(val_ds)}")

    # ------------------------------------------------------- class balance
    weights = None
    if args.class_weights == "inv_sqrt" and num_classes > 2:
        counts = class_pixel_counts(train_seqs, class_map, args.task,
                                    args.height, args.width, stride=7)
        w = inverse_sqrt_weights(counts)
        print("class pixel share / CE weight:")
        tot = max(counts.sum(), 1)
        for n, c, wi in zip(names, counts, w):
            print(f"  {n:<24} {100.0*c/tot:7.4f}%   w={wi:.3f}")
        weights = torch.tensor(w, dtype=torch.float32, device=device)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin,
        drop_last=len(train_ds) > args.batch_size,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None)
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch_size), shuffle=False,
        num_workers=max(2, args.workers // 2), pin_memory=pin,
        persistent_workers=args.workers > 0)

    # -------------------------------------------------------------- model
    model = build_model(num_classes, arch=args.arch,
                        pretrained=not args.no_pretrained).to(device)
    print(f"arch={args.arch}  model params: {count_parameters(model):,}")
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        print("  channels_last memory format enabled")
    if args.compile:
        try:
            model = torch.compile(model)
            print("  torch.compile enabled (first epoch will be slower)")
        except Exception as e:  # noqa: BLE001
            print(f"  torch.compile unavailable, continuing eagerly: {e}")

    # A pretrained ImageNet encoder is fine-tuned an order of magnitude slower
    # than the randomly-initialised decoder; a single lr destroys the encoder
    # features in the first epochs, which costs several IoU points.
    enc_names = encoder_param_names(args.arch)
    enc, dec = [], []
    for n, p in model.named_parameters():
        (enc if n.split(".")[0] in enc_names else dec).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": enc, "lr": args.lr * args.encoder_lr_scale, "name": "encoder"},
         {"params": dec, "lr": args.lr, "name": "decoder"}],
        weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    criterion = SegLoss(args.lambda_ce, args.lambda_dice, args.lambda_tversky,
                        class_weights=weights)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup = int(max(0, args.warmup_frac) * total_steps)

    start_epoch, best, bad_epochs, history, gstep = 0, -1.0, 0, [], 0
    if args.resume and os.path.isfile(args.resume):
        from sitr.utils import load_checkpoint
        ck = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck["meta"].get("epoch", -1)) + 1
        best = float(ck["meta"].get("stats", {}).get("mIoU", -1.0))
        gstep = start_epoch * steps_per_epoch
        print(f"resumed from {args.resume} at epoch {start_epoch} (best={best:.4f})")

    save_manifest(os.path.join(args.out, "data_manifest.json"), seqs,
                  extra={"val_seqs": args.val_seqs, "task": args.task})

    # -------------------------------------------------------------- train
    for epoch in range(start_epoch, args.epochs):
        model.train()
        meter, ce_m, dc_m, tv_m = (AverageMeter(), AverageMeter(),
                                   AverageMeter(), AverageMeter())
        t0 = time.time()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            if args.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            masks = batch["mask"].to(device, non_blocking=True)

            lrs = cosine_lr(optimizer, base_lrs, gstep, total_steps, warmup,
                            args.min_lr_factor)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits, _ = model(images)
                loss, parts = criterion(logits, masks)
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            n = images.size(0)
            meter.update(float(loss.detach()), n)
            ce_m.update(parts["ce"], n)
            dc_m.update(parts["dice"], n)
            tv_m.update(parts.get("tversky", 0.0), n)
            gstep += 1

        stats = evaluate(model, val_loader, device, num_classes, names,
                         instrument_ids, use_amp, tta=args.tta)
        row = {"epoch": epoch, "lr_decoder": lrs[-1], "loss": meter.avg,
               "ce": ce_m.avg, "dice_loss": dc_m.avg, "tversky": tv_m.avg,
               "secs": round(time.time() - t0, 1), **stats}
        history.append(row)
        print(f"[{epoch+1:3d}/{args.epochs}] loss={meter.avg:.4f} "
              f"(ce={ce_m.avg:.4f} dice={dc_m.avg:.4f})  "
              f"val mIoU={stats['mIoU']:.4f}  "
              f"chIoU={stats['challenge_IoU_union']:.4f}  "
              f"instIoU={stats['instrument_IoU_per_frame']:.4f}  {row['secs']}s")

        meta = {"args": vars(args), "epoch": epoch, "stats": stats,
                "num_classes": num_classes, "target_names": names,
                "instrument_ids": instrument_ids, "task": args.task,
                "arch": args.arch,
                "height": args.height, "width": args.width,
                "class_map_path": os.path.abspath(cm_path)}
        save_checkpoint(os.path.join(args.out, "last.pt"), model, meta, optimizer)
        if stats["mIoU"] > best:
            best, bad_epochs = stats["mIoU"], 0
            save_checkpoint(os.path.join(args.out, "best.pt"), model, meta)
            print(f"        new best mIoU={best:.4f} -> best.pt")
        else:
            bad_epochs += 1
            if args.early_stop and bad_epochs >= args.early_stop:
                print(f"        early stop: {bad_epochs} epochs without "
                      f"improvement")
                break

        write_json(os.path.join(args.out, "stage1_history.json"),
                   {"history": history, "best_mIoU": best, "args": vars(args)})

    write_json(os.path.join(args.out, "stage1_history.json"),
               {"history": history, "best_mIoU": best, "args": vars(args)})

    # ---- temperature scaling on the validation split -------------------
    # Fitted AFTER training, on the best checkpoint. It cannot change argmax,
    # so every Stage 1 metric above is unaffected; it only makes the posterior
    # honest, which is what Stage 2's q/H/M indicators are read from.
    from sitr.utils import load_checkpoint
    best_path = os.path.join(args.out, "best.pt")
    ck = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(ck["model"])
    T = fit_temperature(model, val_loader, device, use_amp,
                        channels_last=args.channels_last)
    if T is None:
        print("\ncalibration: fit did not converge; using T=1.0")
        T = 1.0
    rep = calibration_report(model, val_loader, device, T, use_amp,
                             channels_last=args.channels_last)
    if rep:
        print(f"\ncalibration: T={T:.3f}  "
              f"NLL {rep['nll_before']:.4f} -> {rep['nll_after']:.4f}  "
              f"ECE {rep['ece_before']:.4f} -> {rep['ece_after']:.4f}"
              f"  ({'overconfident' if T > 1.05 else 'already calibrated'})")
    ck["meta"]["temperature"] = float(T)
    torch.save(ck, best_path)
    write_json(os.path.join(args.out, "calibration.json"),
               rep or {"temperature": float(T), "note": "no report available"})

    print(f"\nDone. best val mIoU={best:.4f}. Checkpoints in {args.out}")
    print("Next:  python -m tools.cache_features --checkpoint "
          f"{os.path.join(args.out, 'best.pt')} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
