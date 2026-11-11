#!/usr/bin/env python3
"""
TASK Q.2 / Q.3 (CPU part) -- how much could identity relabelling gain at most, and where does identity confusion sit?

    python -m tools.identity_ceiling --actions work_v2/actions --choices results/task5/choices_test.npz --root <endovis2018_cvdataset> --out results/taskQ

Code audit first (sitr/identity.py, the teammate's Stage 5). HungarianIdentityVerifier.__call__ returns IdentityResult(mask=mask, ...) with the INPUT mask: by its own stated design
"identity uncertainty does NOT modify the segmentation mask" (lines 12-19 and 177-181). It also constructs `switches = 0` and never increments it, so the identity-switch metric it
declares is never computed. Consequence, by construction: M7 (= M6 + Stage 5) has exactly the masks of M6, so its mIoU, mcIoU, per-class IoU and class sets are identical to M6's, whatever
the data. The code is not changed here (contract in sitr/stage34_interface.py is respected; the two points are recorded in RUNLOG.md).

What can be measured without changing it:
  1. A GROUND-TRUTH CEILING for any identity-relabelling stage (it uses ground truth, so it is an upper bound for the literature's mechanism, never a method): every connected region of
     predicted instrument pixels (one class, 8-connected) is relabelled to the instrument class that covers most of it in the ground truth, if the ground truth has any instrument there.
     The change in official mIoU and mcIoU is the most that perfect identity assignment could add on top of the M6 masks, region geometry unchanged.
  2. The same ceiling restricted to the clasper/wrist pair (the pair the official run.py and labels.json disagree about), and to nothing else.
  3. The instrument confusion matrix of M6 (pixels, ground-truth class in rows, predicted class in columns), before relabelling and after the ceiling relabelling.
Scored on the 997 test frames at the 512x640 working grid (the ground truth nearest-neighbour downsampled; Task I measured that this changes the official mIoU by +0.04 points).
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
from sitr.metrics import official_class_ious  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

NC = 12
INSTR = [1, 2, 3, 6, 7, 8, 9, 11]
CLASP, WRIST = 2, 3
NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma", "covered-kidney", "thread", "clamps", "suturing-needle",
         "suction-instrument", "intestine", "ultrasound-probe"]


def relabel(pred, gt, classes):
    """Relabel every 8-connected single-class region of `pred` whose class is in `classes` to the class of `classes` covering most of it in `gt` (if any covers it)."""
    out = pred.copy()
    for c in classes:
        m = (pred == c).astype(np.uint8)
        if not m.any():
            continue
        n, lab = cv2.connectedComponents(m, connectivity=8)
        for k in range(1, n):
            reg = lab == k
            cnt = np.bincount(gt[reg].astype(np.int64), minlength=256)[:NC]
            best = max(classes, key=lambda x: cnt[x])
            if cnt[best] > 0:
                out[reg] = best
    return out


def score(frames):
    fi, per = [], {c: [] for c in range(1, NC)}
    for cls in frames:
        if cls:
            fi.append(float(np.mean(list(cls.values()))))
            for c, v in cls.items():
                per[c].append(v)
    pc = {NAMES[c]: (float(np.mean(v)) if v else None) for c, v in per.items()}
    return dict(mIoU=float(np.mean(fi)), mcIoU=float(np.mean([v for v in pc.values() if v is not None])), per_class=pc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", default="work_v2/actions")
    ap.add_argument("--choices", default="results/task5/choices_test.npz")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskQ")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ch = np.load(a.choices, allow_pickle=True)
    keys, ts, choice = [str(k) for k in ch["seq"]], ch["t"], ch["choice"].astype(str)
    seqs = {s.key: s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",))}
    base, full, pair = [], [], []
    conf_b = np.zeros((NC, NC), np.int64)
    conf_a = np.zeros((NC, NC), np.int64)
    changed_region_frames = changed_pixel = tot_instr = 0
    for i in range(len(ts)):
        s = seqs[keys[i]]
        stem = os.path.splitext(os.path.basename(s.frames[int(ts[i])]))[0]
        z = np.load(os.path.join(a.actions, "test_masks", f"{s.split}_{s.name}", stem + ".npz"))
        pred = z[choice[i]] if choice[i] != "A0" else z["A0"]
        gt = cv2.resize(read_mask(s.labels[int(ts[i])]), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        r_full = relabel(pred, gt, INSTR)
        r_pair = relabel(pred, gt, [CLASP, WRIST])
        base.append(official_class_ious(pred, gt, NC))
        full.append(official_class_ious(r_full, gt, NC))
        pair.append(official_class_ious(r_pair, gt, NC))
        valid = gt < NC
        np.add.at(conf_b, (gt[valid].astype(int), pred[valid].astype(int)), 1)
        np.add.at(conf_a, (gt[valid].astype(int), r_full[valid].astype(int)), 1)
        d = int((r_full != pred).sum())
        changed_pixel += d
        changed_region_frames += int(d > 0)
        tot_instr += int(np.isin(pred, INSTR).sum())
    R = {"audit": {"verifier_returns_input_mask": True, "switches_hard_coded_zero": True,
                   "consequence": "M7 has exactly the masks of M6, so mIoU, mcIoU, per-class IoU and class sets are identical by construction"},
         "grid": "512x640 (ground truth nearest-neighbour downsampled)", "n_frames": len(ts),
         "M6": score(base), "ceiling_all_instrument_classes": score(full), "ceiling_clasper_wrist_only": score(pair),
         "ceiling_frames_with_any_relabelled_pixel_pct": 100 * changed_region_frames / len(ts),
         "ceiling_relabelled_pixels_share_of_predicted_instrument_pixels": changed_pixel / max(tot_instr, 1)}
    for k in ("ceiling_all_instrument_classes", "ceiling_clasper_wrist_only"):
        R[k]["mIoU_points_over_M6"] = 100 * (R[k]["mIoU"] - R["M6"]["mIoU"])
        R[k]["mcIoU_points_over_M6"] = 100 * (R[k]["mcIoU"] - R["M6"]["mcIoU"])
    inst = INSTR
    R["confusion_M6_instrument_pixels"] = {"classes": [NAMES[c] for c in inst], "rows_ground_truth_cols_predicted_other_included_as_last": 1,
                                           "before": [[int(conf_b[g, p]) for p in inst] + [int(conf_b[g].sum() - conf_b[g, inst].sum())] for g in inst],
                                           "after_ceiling": [[int(conf_a[g, p]) for p in inst] + [int(conf_a[g].sum() - conf_a[g, inst].sum())] for g in inst]}
    cl_to_wr = conf_b[CLASP, WRIST] / max(conf_b[CLASP].sum(), 1)
    wr_to_cl = conf_b[WRIST, CLASP] / max(conf_b[WRIST].sum(), 1)
    R["clasper_wrist"] = {"gt_clasper_pixels_predicted_wrist": float(cl_to_wr), "gt_wrist_pixels_predicted_clasper": float(wr_to_cl),
                          "gt_clasper_pixels_predicted_clasper": float(conf_b[CLASP, CLASP] / max(conf_b[CLASP].sum(), 1)),
                          "gt_wrist_pixels_predicted_wrist": float(conf_b[WRIST, WRIST] / max(conf_b[WRIST].sum(), 1))}
    json.dump(R, open(os.path.join(a.out, "identity_headroom.json"), "w"), indent=1)
    print(f"M6                   mIoU {R['M6']['mIoU']:.4f}  mcIoU {R['M6']['mcIoU']:.4f}")
    for k in ("ceiling_all_instrument_classes", "ceiling_clasper_wrist_only"):
        print(f"{k:32s} mIoU {R[k]['mIoU']:.4f} ({R[k]['mIoU_points_over_M6']:+.2f} pts)  mcIoU {R[k]['mcIoU']:.4f} ({R[k]['mcIoU_points_over_M6']:+.2f} pts)")
    print("per class M6 -> ceiling:", {n: (round(R['M6']['per_class'][n], 3) if R['M6']['per_class'][n] is not None else None, round(R['ceiling_all_instrument_classes']['per_class'][n], 3) if R['ceiling_all_instrument_classes']['per_class'][n] is not None else None) for n in NAMES[1:4] + NAMES[6:9]})
    print("clasper/wrist:", R["clasper_wrist"])
    print("frames with any relabel:", round(R["ceiling_frames_with_any_relabelled_pixel_pct"], 1), "% ; relabelled share of predicted instrument pixels:", round(100 * R["ceiling_relabelled_pixels_share_of_predicted_instrument_pixels"], 2), "%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
