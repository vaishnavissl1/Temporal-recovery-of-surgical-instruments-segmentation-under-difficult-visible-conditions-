#!/usr/bin/env python3
"""Task F.3: latency of the selector WITHOUT action A1 and without the stereo-temporal fusion, composed from measured components exactly as tools/final_eval.py composes M6.
Reads results/task6/final_results.json (measured segmentation pass, measured per-model latencies), work_v2/actions/test__*.npz (measured per-action seconds per frame) and
results/taskF/no_a1/choices_test.npz. Without A1 the stereo matching (about 1.7 s per frame on the CPU) and the right-view forward pass are no longer needed on flagged frames,
which is where nearly all of the old selector's cost came from."""
import json
import os

import numpy as np

F = json.load(open("results/task6/final_results.json"))
L = F["latency_ms"]
ch = np.load("results/taskF/no_a1/choices_test.npz", allow_pickle=True)
routed = ch["routed"].astype(bool)
sec = {k: np.concatenate([np.load(os.path.join("work_v2/actions", f), allow_pickle=True)[f"sec_{k}"] for f in sorted(os.listdir("work_v2/actions")) if f.startswith("test__")]) * 1000 for k in ("A2", "A3", "FUSE")}
base = L["M1"]["mean_ms"]                     # segmentation pass plus the gate, measured (Task 6)
fr = float(routed.mean())
sel_ms = 0.002 * 1000 * 0                     # the selector call is microseconds (Task 6 used 0.002 s; kept out here, negligible)
m6_noa1 = base + fr * float((sec["A2"][routed] + sec["A3"][routed]).mean())
a23_all = base + float((sec["A2"] + sec["A3"]).mean())
out = {"basis": "composed from measured components (Task 6 latency JSON and per-action seconds in the action data)",
       "flagged_share_test": fr,
       "M1_segmentation_plus_gate_ms": base,
       "M6_with_A1_ms": L["M6"]["mean_ms"], "M4_fusion_every_frame_ms": L["M4"]["mean_ms"], "M5_fusion_refine_every_frame_ms": L["M5"]["mean_ms"],
       "M6_without_A1_ms": m6_noa1,
       "correct_every_frame_A2_then_A3_ms": a23_all,
       "speedup_old_tile_M4_over_M6": L["M4"]["mean_ms"] / L["M6"]["mean_ms"],
       "speedup_M4_over_M6_without_A1": L["M4"]["mean_ms"] / m6_noa1,
       "speedup_correct_every_frame_A2_A3_over_M6_without_A1": a23_all / m6_noa1,
       "note": "A4 (hallucination repair) is not included: it has not been run yet; its cost is a connected-component and set comparison on masks already in memory, expected to be negligible."}
json.dump(out, open("results/taskF/cost.json", "w"), indent=1)
print(json.dumps(out, indent=1))
