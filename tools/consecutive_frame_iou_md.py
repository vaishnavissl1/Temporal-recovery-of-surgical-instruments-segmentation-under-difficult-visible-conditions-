#!/usr/bin/env python3
"""Write results/CONSECUTIVE_FRAME_IOU.md from results/taskK/consecutive_frame_iou.json (every number is read from the JSON)."""
import json
import os
import sys

R = json.load(open("results/taskK/consecutive_frame_iou.json"))
f3 = lambda v: f"{v:.3f}"  # noqa: E731
pc = lambda v: f"{100*v:.0f}%"  # noqa: E731


def pooled_row(label, d):
    lo, hi = d["mean_ci95_sequence_bootstrap"]
    return (f"| {label} | {d['n']} | {f3(d['mean'])} ({f3(lo)} to {f3(hi)}) | {f3(d['median'])} | {f3(d['p05'])} | {f3(d['p25'])} | {f3(d['p75'])} | "
            f"{f3(d['p95'])} | {pc(d['frac_below_0_5'])} |")


L = ["# Consecutive-frame ground-truth IoU",
     "",
     "How much does the ground-truth instrument mask change from one frame to the next in EndoVis 2018? This bounds what any method that "
     "carries a mask forward in time (copy the previous mask, warp it, vote over a memory) can achieve at this frame rate: it is the "
     "overlap of the two TRUE masks, so a perfect propagation method could not exceed it by copying. Produced by `tools/consecutive_frame_iou.py`; "
     "every number below is read from `results/taskK/consecutive_frame_iou.json`.",
     "",
     "## Why the frames are so different: how the dataset was built",
     "",
     "The challenge paper (Allan et al., *2018 Robotic Scene Segmentation Challenge*, arXiv:2001.11190) describes the sampling. Two sentences, "
     "checked against the paper's HTML rendering: sections were \"extracted and subsampled to 1 Hz\", and \"Similar frames were manually removed "
     "until the sequence contained 300 frames.\" The dataset is therefore a 1 Hz sample that was then thinned of near-duplicates on purpose, "
     "which removes exactly the frame pairs that would have overlapped most.",
     "",
     "What the paper does and does not say about participants: in the rendering checked, it does not state that any team used information "
     "from other time steps, and one method (Method 4, NTNU) takes a left/right frame pair as input, so stereo input was used by at least one "
     "team. The absence of a statement is not proof that no team used temporal information; a claim that none did should be checked against "
     "the full method descriptions before it is cited.",
     "",
     "## Definitions",
     "",
     f"- **Instrument IoU**: {R['definitions']['instrument_iou']} (the definition the gate's target uses, and DATASET_FACTS.md section 6).",
     f"- **Region IoU**: {R['definitions']['region_iou']}.",
     f"- Resolution {R['definitions']['resolution']}; pairs are (t-1, t) of one sequence, used only when the two frame numbers differ by exactly 1 "
     "(no pair had to be skipped in any sequence).",
     f"- Random pairs: for every frame, one partner drawn uniformly from the same sequence at least 20 frames away (seed 0), for scale.",
     f"- Intervals: {R['definitions']['interval']}. \"<0.5\" is the share of pairs with IoU below the repo's standing 0.5 (not tuned).",
     ""]
for sp, title in (("train", "Training sequences (15)"), ("test", "Official test sequences (4)")):
    G = R["groups"][sp]
    P = G["pooled"]
    L += [f"## {title}: pooled", "",
          "| pairs | n | mean (95%) | median | 5th pct | 25th pct | 75th pct | 95th pct | < 0.5 |", "|---|---|---|---|---|---|---|---|---|",
          pooled_row("consecutive, instrument IoU", P["consecutive_instrument_iou"]),
          pooled_row("consecutive, region IoU", P["consecutive_region_iou"]),
          pooled_row("random (20+ apart), instrument IoU", P["random_pairs_20_apart_instrument_iou"]),
          pooled_row("random (20+ apart), region IoU", P["random_pairs_20_apart_region_iou"]),
          "", "Histogram of consecutive instrument IoU (10 equal bins from 0 to 1, counts): "
          + ", ".join(str(c) for c in P["consecutive_instrument_iou"]["histogram_10_bins"]) + ".", "",
          f"### Per sequence ({sp}), consecutive pairs", "",
          "| sequence | instrument IoU mean | median | 25th | 75th | < 0.5 | region IoU mean |", "|---|---|---|---|---|---|---|"]
    for k, v in G["sequences"].items():
        i, r = v["instrument_iou"], v["region_iou"]
        L.append(f"| {k} | {f3(i['mean'])} | {f3(i['median'])} | {f3(i['p25'])} | {f3(i['p75'])} | {pc(i['frac_below_0_5'])} | {f3(r['mean'])} |")
    L.append("")
L += ["## Reading it", "",
      "- The distribution is wide and has a heavy low end, so the mean alone hides it: use the percentiles and the histogram above. "
      "A large share of consecutive pairs overlap by less than half, and a visible share overlap by almost nothing.",
      "- Consecutive masks overlap far more than random pairs from the same sequence, so there is temporal information; it is just not enough "
      "for copying a mask forward to be reliable at this rate. This is consistent with the measured temporal-recovery result "
      "(`results/taskA/SUMMARY.md`): the temporal action leaves most flagged frames unchanged.",
      "- Sequences differ a lot. The ranking of sequences here and the ranking of segmentation difficulty are different questions; this file "
      "measures only how fast the scene moves between samples.",
      "- The test sequences are scored on the same definitions and are shown for completeness; the measurement described in the task is the "
      "training-sequence block.",
      "- Ground truth is used here, and only here, to measure the dataset. Nothing in this file feeds any gate input, action input or model.",
      ""]
open("results/CONSECUTIVE_FRAME_IOU.md", "w", encoding="utf-8").write("\n".join(L))
print("wrote results/CONSECUTIVE_FRAME_IOU.md")
