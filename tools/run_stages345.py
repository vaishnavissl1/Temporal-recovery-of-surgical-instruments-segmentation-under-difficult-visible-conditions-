#!/usr/bin/env python3
"""
End-to-end evaluation of Stages 3–5 on the release data.

    python -m tools.run_stages345 --release <path>/release

Reports THREE metrics to validate the selective recovery claim:

    1. IoU_base       — Stage 1 segmentation without any recovery
    2. IoU_always     — recovery applied to EVERY frame (Baseline B)
    3. IoU_selective  — recovery only when r_t < tau (the proposed method)

This is critical because the research claim is NOT "temporal recovery improves
segmentation" but "SELECTIVE temporal recovery is preferable to blindly
applying it."  If IoU_always >= IoU_selective, selectivity adds nothing and
Gap 1 is not substantiated.

Also reports:
    - Routing rate (should match Stage 2's reported rho)
    - Cold-start rate and unrecoverable rate
    - Identity switches
    - Per-sequence breakdown
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.align import IdentityAlign, build_align  # noqa: E402
from sitr.identity import HungarianIdentityVerifier  # noqa: E402
from sitr.metrics import frame_iou  # noqa: E402
from sitr.recovery import TemporalRecovery  # noqa: E402
from sitr.runtime import ReleaseBundle  # noqa: E402


def run_sequence_eval(bundle, seq, mode, aligner, gamma, verbose=False):
    """Run one sequence and return per-frame IoU and stats.

    mode: 'base' | 'always' | 'selective'
    """
    tau = bundle.tau
    num_classes = bundle.num_classes
    instrument_ids = bundle.instrument_ids

    if mode == "base":
        # No recovery at all
        runner = bundle.runner(recovery=None, identity=None, aligner=None)
    elif mode == "always":
        # Recovery on EVERY frame (Baseline B).
        # We use the same recovery module but override state.reliable = False
        # before each step() so SelectiveInference routes ALL frames through
        # recovery (except cold starts, which have no memory).
        rec = TemporalRecovery(
            aligner=aligner, tau=tau, gamma=gamma,
            num_classes=num_classes, instrument_ids=instrument_ids)
        iv = HungarianIdentityVerifier(instrument_ids=instrument_ids)
        runner = bundle.runner(recovery=rec, identity=iv)
    else:  # selective
        rec = TemporalRecovery(
            aligner=aligner, tau=tau, gamma=gamma,
            num_classes=num_classes, instrument_ids=instrument_ids)
        iv = HungarianIdentityVerifier(instrument_ids=instrument_ids)
        runner = bundle.runner(recovery=rec, identity=iv)

    runner.reset()
    ious = []
    n_frames = 0

    for state in bundle.stream(seq, with_gt=True):
        if mode == "always":
            # Force ALL frames through recovery path by overriding routing.
            # Recalculate cold_start based on the RUNNER's actual memory,
            # not the stream's internal memory.
            state.reliable = False
            state.cold_start = (len(runner.memory) == 0)
        final_mask, _ = runner.step(state)
        if state.ground_truth is not None:
            iou = frame_iou(final_mask, state.ground_truth, num_classes,
                            class_ids=instrument_ids)
            ious.append(iou)
        n_frames += 1

    stats = dict(runner.stats)
    stats["n_frames"] = n_frames
    stats["mean_iou"] = float(np.mean(ious)) if ious else 0.0
    return ious, stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stages 3-5 on the release data.")
    parser.add_argument("--release", required=True,
                        help="Path to the release folder")
    parser.add_argument("--split", default="train",
                        help="Split to evaluate (default: train)")
    parser.add_argument("--aligner", default="identity",
                        choices=["identity", "farneback", "region"],
                        help="Alignment operator (default: identity)")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Temporal decay factor (default: 0.95)")
    parser.add_argument("--sequences", type=int, default=None,
                        help="Max sequences to evaluate (default: all)")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON report")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Loading release from {args.release}...")
    bundle = ReleaseBundle.load(args.release)
    print(f"  num_classes={bundle.num_classes}, tau={bundle.tau:.4f}, "
          f"tau_memory={bundle.tau_memory:.4f}, K={bundle.memory_size}")
    print(f"  instrument_ids={bundle.instrument_ids}")
    print(f"  reported routing_rate={bundle.manifest.get('routing_rate_rho', '?')}")

    aligner_kwargs = {}
    if args.aligner == "region":
        aligner_kwargs["instrument_ids"] = bundle.instrument_ids
    aligner = build_align(args.aligner, **aligner_kwargs)
    print(f"  aligner={aligner.name}, gamma={args.gamma}")

    seqs = bundle.sequences(split=args.split)
    if args.sequences:
        seqs = seqs[:args.sequences]
    print(f"\nEvaluating {len(seqs)} sequences on split={args.split}")
    print("=" * 70)

    results = {}
    for mode in ["base", "selective", "always"]:
        print(f"\n--- Mode: {mode.upper()} ---")
        all_ious = []
        all_stats = {"frames": 0, "routed": 0, "unrecoverable": 0,
                     "degenerate": 0, "identity_switches": 0}
        seq_results = []

        for seq in seqs:
            t0 = time.time()
            ious, stats = run_sequence_eval(
                bundle, seq, mode, aligner, args.gamma, args.verbose)
            elapsed = time.time() - t0
            all_ious.extend(ious)

            for k in all_stats:
                all_stats[k] += stats.get(k, 0)

            seq_info = {
                "name": seq.name,
                "n_frames": stats["n_frames"],
                "mean_iou": stats["mean_iou"],
                "routed": stats.get("routed", 0),
                "time_s": round(elapsed, 1),
            }
            seq_results.append(seq_info)
            print(f"  {seq.name}: IoU={stats['mean_iou']:.4f}, "
                  f"frames={stats['n_frames']}, "
                  f"routed={stats.get('routed', 0)}, "
                  f"{elapsed:.1f}s")

        total = all_stats["frames"]
        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
        routing_rate = all_stats["routed"] / max(total, 1)
        cold_rate = all_stats["unrecoverable"] / max(total, 1)

        summary = {
            "mean_iou": mean_iou,
            "n_frames": total,
            "routing_rate": routing_rate,
            "cold_start_rate": cold_rate,
            "degenerate_rate": all_stats["degenerate"] / max(total, 1),
            "identity_switches": all_stats["identity_switches"],
            "sequences": seq_results,
        }
        results[mode] = summary

        print(f"\n  >> {mode.upper()} overall: IoU={mean_iou:.4f}, "
              f"routing={routing_rate:.4f}, "
              f"cold_start={cold_rate:.4f}, "
              f"switches={all_stats['identity_switches']}")

    # Final comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    for mode in ["base", "selective", "always"]:
        r = results[mode]
        print(f"  {mode:12s}  IoU={r['mean_iou']:.4f}  "
              f"routing={r['routing_rate']:.4f}  "
              f"cold={r['cold_start_rate']:.4f}  "
              f"switches={r['identity_switches']}")

    sel = results["selective"]["mean_iou"]
    base = results["base"]["mean_iou"]
    always = results["always"]["mean_iou"]
    print(f"\n  Selective vs Base:   {sel - base:+.4f}")
    print(f"  Selective vs Always: {sel - always:+.4f}")
    print(f"  Always vs Base:     {always - base:+.4f}")

    if sel > base:
        print("\n  [PASS] Selective recovery improves over baseline")
    else:
        print("\n  [FAIL] Selective recovery does NOT improve over baseline")

    if sel > always:
        print("  [PASS] Selectivity matters: selective > always-on")
    else:
        print("  [FAIL] Selectivity does not help: always-on >= selective")

    # Save report
    if args.output:
        report = {
            "release": os.path.abspath(args.release),
            "split": args.split,
            "aligner": args.aligner,
            "gamma": args.gamma,
            "tau": bundle.tau,
            "tau_memory": bundle.tau_memory,
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                     exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
