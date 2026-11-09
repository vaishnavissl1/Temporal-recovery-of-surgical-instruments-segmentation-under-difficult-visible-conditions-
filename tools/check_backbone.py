#!/usr/bin/env python3
"""
PRE-FLIGHT for tools/run_v2.py -- run this first, it takes ~2 minutes.

  1. downloads the encoder weights and PROVES they load (every tensor matched;
     it stops with an error rather than silently training from random weights)
  2. finds the largest batch size that fits your GPU at 512x640, with and
     without gradient checkpointing
  3. measures real training speed and estimates the whole run_v2 duration
  4. prints the exact run_v2 command to use

    python -m tools.check_backbone
    python -m tools.check_backbone --pretrained-path C:\\path\\to\\SurgeNetXL_checkpoint_epoch0050_teacher.pth
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.losses import SegLoss  # noqa: E402
from sitr.models import build_model, count_parameters  # noqa: E402


def probe(arch, bs, H, W, gc, device, pretrained_path, steps=4):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(12, arch, pretrained=False, grad_checkpoint=gc).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = SegLoss(1.0, 1.0, 0.5)
    x = torch.randn(bs, 3, H, W, device=device)
    y = torch.randint(0, 12, (bs, H, W), device=device)
    times = []
    try:
        for i in range(steps):
            torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda"):
                lo, _ = model(x)
                loss, _ = crit(lo, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i:
                times.append(time.time() - t)
        peak = torch.cuda.max_memory_allocated() / 2**30
        # Windows spills to system RAM instead of raising OOM, so a "fit" is
        # only real if the memory held stays inside the GPU's own free memory.
        usable_gb = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()) / 2**30
        if torch.cuda.max_memory_reserved() / 2**30 > usable_gb:
            return False, None, None
        return True, peak, sum(times) / len(times)
    except torch.cuda.OutOfMemoryError:
        return False, None, None
    finally:
        del model, opt
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="caformer_surgenet",
                    choices=["caformer_surgenet", "caformer_imagenet"])
    ap.add_argument("--pretrained-path", default=None)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--train-frames", type=int, default=2235,
                    help="EndoVis 2018 training frames")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--effective-batch", type=int, default=8)
    args = ap.parse_args()

    print("1. weights")
    m = build_model(12, args.arch, pretrained=True, pretrained_path=args.pretrained_path)
    print(f"   {args.arch}: {count_parameters(m)/1e6:.1f}M parameters, "
          "pretrained encoder verified")
    del m

    if not torch.cuda.is_available():
        print("\nNo CUDA GPU visible to PyTorch. Check `nvidia-smi` and that you "
              "installed the CUDA build of torch.")
        return 1
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 2**30
    print(f"\n2. GPU: {props.name}, {total:.1f} GB")

    best = None
    for gc in (False, True):
        for bs in (4, 2, 1):
            ok, peak, sec = probe(args.arch, bs, args.height, args.width, gc, dev,
                                  args.pretrained_path)
            tag = f"batch {bs}{' + grad-checkpoint' if gc else ''}"
            if ok and peak < 0.90 * total:
                print(f"   {tag:<28} fits: peak {peak:.2f} GB, {sec:.2f} s/step "
                      f"({bs/sec:.1f} img/s)")
                cand = (bs / sec, bs, gc, sec)
                if best is None or cand[0] > best[0]:
                    best = cand
                break
            print(f"   {tag:<28} does not fit")
    if best is None:
        print("\nEven batch 1 with gradient checkpointing does not fit at "
              f"{args.height}x{args.width}. Use --height 384 --width 480 "
              "(still a multiple of 32, same 5:4 aspect).")
        return 1

    ips, bs, gc, sec = best
    accum = max(1, args.effective_batch // bs)
    k = args.folds
    frames_per_run = [args.train_frames * (k - 1) / k] * k + [args.train_frames]
    hours = sum(frames_per_run) * args.epochs / ips / 3600
    hours *= 1.15   # evaluation, data loading, caching
    print(f"\n3. estimate: {ips:.1f} img/s -> about {hours:.0f} h for "
          f"{k} folds + final model at {args.epochs} epochs "
          f"(~{hours/24:.1f} days)")
    if hours > 72:
        print("   That exceeds 3 days. Options: --folds 3 (~40% less), or "
              "--epochs 30.")

    print("\n4. command (from your project folder):\n")
    cmd = (f'python -m tools.run_v2 --root "<your endovis2018_cvdataset path>" '
           f"--out work_v2 --class-map work\\class_map.json --arch {args.arch} "
           f"--batch-size {bs} --accum-steps {accum}"
           f"{' --grad-checkpoint' if gc else ''}")
    if args.pretrained_path:
        cmd += f' --pretrained-path "{args.pretrained_path}"'
    print("   " + cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
