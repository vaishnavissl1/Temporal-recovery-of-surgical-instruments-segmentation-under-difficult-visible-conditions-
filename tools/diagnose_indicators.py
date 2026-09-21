#!/usr/bin/env python3
"""
WHY does the replayed routing rate disagree with the reported one?

`tools/verify_release.py` compares one number: the fraction of frames routed to
recovery, observed on replay against the rho in stage2_report.json. When those
disagree, the cause is always the same in kind -- an indicator is computed
differently at inference than it was when Stage 2 was fitted -- but guessing
WHICH indicator wastes hours. This measures it.

For every validation frame it computes the indicator vector twice:

    cached  : exactly what tools/train_stage2.py saw (sitr.reliability.run_sequence
              over the .npz cache, self-predicted history, sched_prob=1.0)
    replay  : exactly what your teammate will see (sitr.runtime.ReleaseBundle.stream,
              recomputed from the frozen backbone)

and reports, per indicator, the mean absolute difference and the correlation
between the two. An indicator that agrees is not the problem; one with a large
mean difference is.

    python -m tools.diagnose_indicators --release work/release

Known causes, in the order they are worth checking:

  mask_scale   T_temporal is an IoU between MASKS. tools/cache_features.py
               stores masks downsampled by --mask-scale (default 2), so Stage 2
               was fitted on a T computed at half resolution. Recomputing it at
               full resolution shifts every T_t.
  tta          --tta averages a horizontal flip into the cached posterior.
               The runtime does not, so q/H/M all shift.
  amp          the cache is built under autocast; the runtime runs fp32. This
               is small but nonzero and shows up as a tiny difference in the
               posterior-derived indicators.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.reliability import StreamConfig, run_sequence  # noqa: E402
from sitr.runtime import ReleaseBundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    b = ReleaseBundle.load(args.release, device=args.device)
    m = b.manifest
    names = list(m["indicator_names"])
    print(f"mask_scale={b.mask_scale}  tau={b.tau:.3f}  "
          f"tau_memory={b.tau_memory:.3f}  K={b.memory_size}")
    idx = os.path.join(args.release, "reports", "cache_index.json")
    if os.path.isfile(idx):
        with open(idx) as f:
            ci = json.load(f)
        print(f"cache was built with: mask_scale={ci.get('mask_scale')}  "
              f"tta={ci.get('tta')}  temperature={ci.get('temperature')}")
        if ci.get("tta"):
            print("  !! the cache used TTA and the runtime does not — that "
                  "alone will shift every posterior-derived indicator")

    val = set(m.get("val_seqs", []))
    seqs = [s for s in b.sequences("train") if s.index in val][:args.sequences]
    print(f"\ncomparing {len(seqs)} sequences: {[s.name for s in seqs]}\n")

    def predict(vec):
        with torch.no_grad():
            return float(b.reliability(torch.from_numpy(
                np.asarray(vec, np.float32)).view(1, -1).to(b.device)).item())

    A, B, rA, rB = [], [], [], []
    cfg = StreamConfig(memory_size=b.memory_size, tau=b.tau_memory,
                       indicators=m["indicator_set"])
    for s in seqs:
        cache = b.cache(s)
        if cache is None:
            print(f"  ! no cache for {s.name}, skipping")
            continue
        out = run_sequence(cache, cfg, predict, sched_prob=1.0,
                           rng=np.random.default_rng(0))
        st = [x for x in b.stream(s, batch_size=args.batch_size, with_gt=False)]
        n = min(len(out["feats"]), len(st))
        A.append(out["feats"][:n])
        B.append(np.array([x.indicators for x in st[:n]], np.float32))
        rA.append(out["r_hat"][:n])
        rB.append(np.array([x.reliability for x in st[:n]], np.float32))

    A, B = np.concatenate(A), np.concatenate(B)
    rA, rB = np.concatenate(rA), np.concatenate(rB)

    print(f"{'indicator':<20}{'mean|diff|':>11}{'max|diff|':>11}{'corr':>8}"
          f"{'cached mu':>11}{'replay mu':>11}")
    print("-" * 72)
    worst = []
    for i, nm in enumerate(names):
        d = np.abs(A[:, i] - B[:, i])
        sa, sb = A[:, i].std(), B[:, i].std()
        c = (float(np.corrcoef(A[:, i], B[:, i])[0, 1])
             if sa > 1e-9 and sb > 1e-9 else float("nan"))
        print(f"{nm:<20}{d.mean():>11.5f}{d.max():>11.5f}{c:>8.4f}"
              f"{A[:, i].mean():>11.5f}{B[:, i].mean():>11.5f}")
        worst.append((d.mean(), nm))

    print("-" * 72)
    print(f"{'r_t':<20}{np.abs(rA-rB).mean():>11.5f}"
          f"{np.abs(rA-rB).max():>11.5f}"
          f"{float(np.corrcoef(rA, rB)[0,1]):>8.4f}"
          f"{rA.mean():>11.5f}{rB.mean():>11.5f}")
    print(f"\nrouting rate at tau={b.tau:.2f}:  cached {float((rA < b.tau).mean()):.3f}"
          f"   replay {float((rB < b.tau).mean()):.3f}")

    worst.sort(reverse=True)
    print("\nlargest disagreement:")
    for d, nm in worst[:3]:
        print(f"  {nm:<20} mean |diff| = {d:.5f}")
    top = worst[0][1]
    if worst[0][0] < 1e-4:
        print("\nEvery indicator agrees. The disagreement is not in the "
              "indicators — check tau, tau_memory and the sequence subset.")
    elif top == "T_temporal":
        print("\nT_temporal dominates -> a MASK RESOLUTION mismatch. The cache "
              "stores masks at height//mask_scale; the runtime must compute "
              "T at that same scale.")
    elif top in ("q_confidence", "H_entropy_conf", "M_margin"):
        print(f"\n{top} dominates -> a POSTERIOR mismatch: TTA, temperature or "
              "autocast differ between cache_features.py and runtime.stream().")
    else:
        print(f"\n{top} dominates. Compare how tools/cache_features.py and "
              "sitr/runtime.py compute it; they must match exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
