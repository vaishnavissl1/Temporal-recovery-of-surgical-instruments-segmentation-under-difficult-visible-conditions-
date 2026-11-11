#!/usr/bin/env python3
"""
Task B support: why does an action gain a lot of 'instrument IoU' on flagged frames but almost nothing on the
official metric?  Scores the SAME saved masks (A0 = base, and the chosen action) on the frames the gate flagged,
under three definitions, and counts predicted-but-absent (hallucinated) instrument classes.

    python -m tools.metric_gap --actions work_v2/actions --choices results/task5/choices_test.npz \
        --root <endovis2018_cvdataset> --action A3 --out results/taskB/metric_gap_A3.json

  instr_union   mean IoU over instrument classes that appear in the ground truth OR the prediction
                (the definition the selector's 'delta' and the earlier tables used; present="union")
  instr_gtonly  the same but a class counts only if it is in the ground truth
  official      the challenge's per-frame score (all classes, ground-truth-only rule)
Ground truth is used for metrics only. Test frames only (the saved masks exist only for test).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sitr.dataset import discover_sequences  # noqa: E402
from sitr.metrics import frame_iou, official_frame_iou  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402
from tools.final_eval import INSTR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--choices", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--action", default="A3", choices=["A1", "A2", "A3", "FUSE", "FUSE_REFINE"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seqs = {s.key: s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",))}
    ch = np.load(a.choices, allow_pickle=True)
    keys = [str(k) for k in ch["seq"]]
    ts = ch["t"]
    routed = ch["routed"].astype(bool)
    rows = []
    for i in np.where(routed)[0]:
        s = seqs[keys[i]]
        stem = os.path.splitext(os.path.basename(s.frames[int(ts[i])]))[0]
        z = np.load(os.path.join(a.actions, "test_masks", f"{s.split}_{s.name}", stem + ".npz"))
        gt = read_mask(s.labels[int(ts[i])])
        r = {}
        for tag, k in (("before", "A0"), ("after", a.action)):
            m = cv2.resize(z[k], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            r[tag] = dict(un=frame_iou(m, gt, 12, class_ids=INSTR, present="union"),
                          gtonly=frame_iou(m, gt, 12, class_ids=INSTR, present="gt"),
                          off=official_frame_iou(m, gt, 12),
                          halluc=sorted(int(c) for c in INSTR if (m == c).sum() > 0 and (gt == c).sum() == 0))
        rows.append(r)

    def mean(f):
        return float(np.nanmean([f(r) for r in rows]))

    out = dict(action=a.action, n_total_test_frames=int(len(routed)), n_flagged=len(rows),
               instrument_iou_union_change=mean(lambda r: r["after"]["un"] - r["before"]["un"]),
               instrument_iou_gt_only_change=mean(lambda r: r["after"]["gtonly"] - r["before"]["gtonly"]),
               official_frame_iou_change=mean(lambda r: np.nan if r["before"]["off"] is None else r["after"]["off"] - r["before"]["off"]),
               flagged_frames_with_hallucinated_instrument_class_before=sum(1 for r in rows if r["before"]["halluc"]),
               flagged_frames_with_hallucinated_instrument_class_after=sum(1 for r in rows if r["after"]["halluc"]),
               hallucinated_instrument_classes_removed=sum(len(set(r["before"]["halluc"]) - set(r["after"]["halluc"])) for r in rows),
               hallucinated_instrument_classes_added=sum(len(set(r["after"]["halluc"]) - set(r["before"]["halluc"])) for r in rows),
               instrument_class_ids=INSTR)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
