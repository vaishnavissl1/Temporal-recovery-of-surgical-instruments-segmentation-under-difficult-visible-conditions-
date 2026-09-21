#!/usr/bin/env python3
"""
PRE-HANDOVER CHECK — run this on the release folder before giving it to anyone.

The tests in tests/test_pipeline.py prove the CODE is correct. This proves that
the ARTEFACTS YOU ACTUALLY PRODUCED are correct and mutually consistent, by
loading the release exactly as your teammate will and replaying real sequences
through it.

It checks the things that, if wrong, would make every later stage wrong:

  1. the release loads, and the model / estimator / manifest agree with each other
  2. every frame yields a complete FrameState: I_t, P_t, S_t, Phi_t, E_t, r_t
  3. P_t is a genuine probability distribution (sums to 1 everywhere)
  4. the reliable memory B_t is complete (M_s = {S,P,E,ID,r}) and bounded by K
  5. memory never crosses a sequence boundary
  6. recovery is offered ONLY on unreliable, non-cold-start frames
  7. the routing rate observed here matches the rho in stage2_report.json
     -- this is the strongest single check: it proves the shipped estimator,
     the shipped tau and the reported numbers are the same system
  8. ground truth never influences an inference-time indicator

    python -m tools.verify_release --release work/release

Exit code 0 means the release is internally consistent and safe to hand over.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.runtime import ReleaseBundle  # noqa: E402
from sitr.stage34_interface import RecoveryResult  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--sequences", type=int, default=3,
                    help="how many validation sequences to replay")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--data", default=None,
                    help="prepared image folder, if not at <release>/prepared")
    args = ap.parse_args()

    problems, notes = [], []

    # ---------------------------------------------------------------- load
    print("1. loading the release exactly as your teammate will ...")
    bundle = ReleaseBundle.load(args.release, device=args.device,
                                data_root=args.data)
    m = bundle.manifest
    data_path = bundle.resolve_data_root()
    inside = os.path.abspath(data_path).startswith(
        os.path.abspath(args.release) + os.sep)
    print(f"   image data: {data_path}"
          f"{'  (inside the release: portable)' if inside else ''}")
    if not inside and args.data is None:
        problems.append(
            "the release is NOT self-contained: the image data it needs is at "
            f"{data_path}, outside the release folder, so it will not exist on "
            "your teammate's machine. Copy it in:  xcopy /e /i /y "
            f"\"{data_path}\" \"{os.path.join(args.release, 'prepared')}\"")
    print(f"   task={m['task']}  arch={m['arch']}  classes={bundle.num_classes}")
    print(f"   tau={bundle.tau:.3f} (routing)  tau_memory={bundle.tau_memory:.3f}"
          f" (admission)  K={bundle.memory_size}")
    print(f"   T={bundle.temperature:.3f}  indicators={m['indicator_set']}")
    if "tau_train" not in m:
        notes.append("manifest has no tau_train, so memory admission falls "
                     "back to the routing tau. Regenerate the release with a "
                     "current tools/run_all.py.")
    print(f"   Stage 1 mIoU={m.get('stage1_best_mIoU', float('nan')):.4f}  "
          f"Stage 2 AUC={m['stage2'].get('auc_detect_bad_frames', float('nan')):.4f}")

    # ------------------------------------------------------------- replay
    val = set(m.get("val_seqs", []))
    seqs = [s for s in bundle.sequences("train") if s.index in val][:args.sequences]
    if not seqs:
        seqs = bundle.sequences("train")[:args.sequences]
    print(f"\n2. replaying {len(seqs)} validation sequences "
          f"({[s.name for s in seqs]}) ...")

    offered, n_frames = [], 0
    seen_reliable, mem_max = 0, 0

    class Spy:
        def __call__(self, state):
            offered.append(state.t)
            if not state.needs_recovery:
                problems.append(f"recovery offered on frame {state.t} which is "
                                "reliable or cold-start")
            for e in state.memory:
                if e.probs is None:
                    problems.append("a memory entry has no P_s — eq. (18) "
                                    "aggregates probability maps and cannot run")
                    break
            return RecoveryResult(mask=state.mask)

    runner = bundle.runner(recovery=Spy())
    for s in seqs:
        runner.reset()
        if runner.memory:
            problems.append("reset() did not clear the memory")
        prev_t = -1
        for st in bundle.stream(s, batch_size=args.batch_size):
            n_frames += 1
            if st.t <= prev_t:
                problems.append(f"frames out of temporal order at t={st.t}")
            prev_t = st.t

            if st.probs.shape[0] != bundle.num_classes:
                problems.append(f"P_t has {st.probs.shape[0]} channels, "
                                f"expected {bundle.num_classes}")
            if st.probs.shape[1:] != st.mask.shape:
                problems.append("P_t and S_t disagree on spatial size")
            tot = st.probs.sum(0)
            if not np.allclose(tot, 1.0, atol=1e-3):
                problems.append(f"P_t is not a distribution at t={st.t} "
                                f"(sums {tot.min():.3f}..{tot.max():.3f})")
            if st.features.ndim != 3:
                problems.append("Phi_t is not a dense feature map")
            if st.embedding.ndim != 1:
                problems.append("E_t is not a vector")
            if not (0.0 <= st.reliability <= 1.0):
                problems.append(f"r_t out of range at t={st.t}: {st.reliability}")
            if len(st.indicators) != len(m["indicator_names"]):
                problems.append("indicator count disagrees with the manifest")
            if len(st.memory) > bundle.memory_size:
                problems.append(f"|B_t| = {len(st.memory)} exceeds K="
                                f"{bundle.memory_size}")
            mem_max = max(mem_max, len(st.memory))
            if any(e.t >= st.t or e.t < st.t - bundle.memory_size
                   for e in st.memory):
                problems.append(f"memory at t={st.t} holds a frame outside "
                                f"[t-K, t-1]")
            seen_reliable += int(st.reliable)
            runner.step(st)

    print(f"   {n_frames} frames replayed, max |B_t| = {mem_max} (K="
          f"{bundle.memory_size})")

    # -------------------------------------------------- routing agreement
    print("\n3. does the shipped system reproduce the reported routing rate?")
    observed = 1.0 - seen_reliable / max(n_frames, 1)
    reported = m.get("routing_rate_rho")
    print(f"   observed rho = {observed:.3f}   reported rho = "
          f"{reported if reported is None else f'{reported:.3f}'}")
    if reported is None:
        notes.append("manifest has no routing_rate_rho to compare against")
    elif abs(observed - reported) > 0.10:
        problems.append(
            f"routing rate disagrees: replay gives {observed:.3f}, the report "
            f"says {reported:.3f}. The shipped estimator, tau and report are "
            "not the same system. Check first that tau_memory (eq. 15 "
            "admission) and tau (eq. 2 routing) are not conflated, then that "
            "--sequences covers every sequence in the manifest's val_seqs.")
    else:
        print("   -> agree (the shipped estimator, tau and report are consistent)")

    if offered:
        print(f"   recovery was offered on {len(offered)} frames, all of them "
              "unreliable and non-cold-start")
    else:
        notes.append("recovery was never offered — every replayed frame was "
                     "reliable or cold-start; try more sequences")

    # ------------------------------------------------- ground-truth safety
    print("\n4. can ground truth reach an inference-time indicator?")
    s = seqs[0]
    a = [st.indicators for st in bundle.stream(s, batch_size=args.batch_size,
                                               with_gt=True)]
    b = [st.indicators for st in bundle.stream(s, batch_size=args.batch_size,
                                               with_gt=False)]
    if a != b:
        problems.append("indicators change when ground truth is withheld — "
                        "label information is leaking into inference")
    else:
        print("   -> no: indicators are identical with and without labels")

    # --------------------------------------------------------------- files
    print("\n5. release contents")
    need = ["manifest.json", "class_map.json", "stage1_best.pt",
            "stage2_reliability.pt", "README_FOR_STAGE3.md"]
    for f in need:
        ok = os.path.exists(os.path.join(args.release, f))
        print(f"   {'OK ' if ok else 'MISSING'}  {f}")
        if not ok:
            problems.append(f"missing {f}")
    ncache = len([f for f in os.listdir(os.path.join(args.release, "caches"))
                  if f.endswith(".npz")]) if os.path.isdir(
                      os.path.join(args.release, "caches")) else 0
    print(f"   OK   caches/ ({ncache} sequences)")
    if ncache == 0:
        problems.append("no per-sequence caches in the release")

    # -------------------------------------------------------------- verdict
    print("\n" + "=" * 70)
    if notes:
        for n in notes:
            print(f"NOTE: {n}")
    if problems:
        print(f"NOT READY TO HAND OVER — {len(problems)} problem(s):")
        for p in dict.fromkeys(problems):
            print(f"  - {p}")
        return 1
    print("RELEASE VERIFIED — internally consistent and safe to hand over.")
    print("Stages 3-5 can be built on this without re-running Stages 1-2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
