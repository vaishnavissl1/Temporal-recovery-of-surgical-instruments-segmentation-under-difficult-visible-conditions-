"""Measure peak VRAM and training throughput of the segmentation model.

Why this exists: before any long run we must KNOW what fits on this GPU, not
guess. Each (resolution, batch) is run in its OWN process so a failed attempt
cannot leave fragmented memory behind and distort the next measurement.

A real training step is timed: forward, SegLoss (CE + Dice), AMP backward,
AdamW step, channels_last -- the same ingredients as tools/train_stage1.py.
Synthetic random tensors are used because the SHAPE, not the content,
determines memory and speed. Nothing here is evaluated or reported as a
segmentation result.

    python -m tools.bench_vram --sizes 384x480 448x576 512x640 --max-batch 8

Usable VRAM is measured (torch.cuda.mem_get_info before the model is built),
not assumed, so memory held by other applications is accounted for.
"""
import argparse
import json
import subprocess
import sys
import time

GB = 1024 ** 3


def single(args) -> None:
    """Run ONE configuration and print one JSON line."""
    import torch
    from sitr.losses import SegLoss
    from sitr.models import build_model

    dev = torch.device("cuda")
    free0, total = torch.cuda.mem_get_info()
    out = dict(h=args.height, w=args.width, batch=args.batch,
               ckpt=bool(args.ckpt), free_before_gb=free0 / GB,
               total_gb=total / GB, status="ok")
    try:
        torch.manual_seed(0)
        model = build_model(args.classes, args.arch, pretrained=False,
                            grad_checkpoint=bool(args.ckpt)).to(dev)
        model = model.to(memory_format=torch.channels_last)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        crit = SegLoss(lambda_tversky=0.0).to(dev)
        x = torch.randn(args.batch, 3, args.height, args.width, device=dev) \
            .contiguous(memory_format=torch.channels_last)
        y = torch.randint(0, args.classes, (args.batch, args.height, args.width),
                          device=dev)
        torch.cuda.reset_peak_memory_stats()

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(x)
            loss, _ = crit(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / GB
        out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / GB
        out["img_per_s"] = args.batch * args.steps / dt
        out["sec_per_step"] = dt / args.steps
        out["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    except torch.cuda.OutOfMemoryError:
        out["status"] = "OOM"
    except Exception as e:  # report, never hide
        out["status"] = f"ERROR {type(e).__name__}: {str(e)[:200]}"
    print("RESULT " + json.dumps(out), flush=True)


def driver(args) -> None:
    rows = []
    for size in args.sizes:
        h, w = (int(v) for v in size.lower().split("x"))
        for ckpt in args.ckpt_modes:
            for b in range(1, args.max_batch + 1):
                cmd = [sys.executable, "-m", "tools.bench_vram", "--single",
                       "--height", str(h), "--width", str(w), "--batch", str(b),
                       "--ckpt", str(ckpt), "--arch", args.arch,
                       "--classes", str(args.classes),
                       "--steps", str(args.steps), "--warmup", str(args.warmup)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                if not line:
                    r = dict(h=h, w=w, batch=b, ckpt=ckpt, status="CRASH",
                             detail=(p.stderr or p.stdout)[-300:])
                else:
                    r = json.loads(line[0][7:])
                rows.append(r)
                if r["status"] == "ok":
                    print(f"{h}x{w} ckpt={ckpt} batch={b}: peak alloc "
                          f"{r['peak_alloc_gb']:.2f} GB, reserved "
                          f"{r['peak_reserved_gb']:.2f} GB, "
                          f"{r['img_per_s']:.2f} img/s "
                          f"(free before {r['free_before_gb']:.2f} of "
                          f"{r['total_gb']:.2f} GB)", flush=True)
                    if r["peak_reserved_gb"] > r["free_before_gb"]:
                        # Windows (WDDM) does not raise OOM here: it silently
                        # spills into system RAM and gets several times slower.
                        print("   -> exceeds free VRAM: spilling into system "
                              "memory, treated as NOT FITTING", flush=True)
                        break
                else:
                    print(f"{h}x{w} ckpt={ckpt} batch={b}: {r['status']}", flush=True)
                    break  # larger batches cannot fit either
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--sizes", nargs="+", default=["384x480", "448x576", "512x640"])
    ap.add_argument("--ckpt-modes", nargs="+", type=int, default=[1],
                    help="1 = gradient checkpointing ON, 0 = OFF")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--arch", default="caformer_surgenet")
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ckpt", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    single(a) if a.single else driver(a)
