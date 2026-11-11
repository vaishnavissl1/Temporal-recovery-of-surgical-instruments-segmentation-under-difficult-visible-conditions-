#!/usr/bin/env python3
"""
TASK F.4 -- action A4, hallucination repair.

    python -m tools.action_a4 --work work_v2 --root <endovis2018_cvdataset> --out results/taskF/a4

Rule (stated in the task): when a class is predicted in only ONE of the two views and the temporal memory does not support it either, drop that class. Concretely, per frame:
  * a class is "predicted" in a view if its area is at least FLOOR of that view's pixels (FLOOR = 0.05%, the same floor as the hallucination rate of Task F.2);
  * the left view is the frame's base mask (A0); the right view is the right-image prediction of the same model (work_v2/right_view); class presence is a SET comparison, so no disparity
    and no geometry is involved (the two views see the same scene at the same instant);
  * the temporal memory is the reliable base masks of the last K = 3 frames (admitted when the gate's reliability >= tau_memory = 0.5, as in the stream), reconstructed from the stored
    gate reliability of each frame; a class is "supported by memory" if it is predicted in any memory entry;
  * a class predicted in the left view, absent from the right view and absent from every memory entry is DROPPED: its pixels become background (class 0). No probabilities are used.
Gating. A4 is applied only on the frames the gate flags (reliability below tau = 0.64 and not degenerate), like every other action. It is NOT a learned-selector candidate here (that would need the
selector's per-action features, which require the full action pipeline); it is a rule that fires or does not, so "gated like any other action" is implemented as "applied on flagged frames only".
That is a stated simplification, recorded in RUNLOG.md.

Measured, out of fold (15 training sequences, each predicted by the fold model that excluded it; at the 256x320 cache grid, ground truth nearest-neighbour downsampled) and on the official test
set (the saved 512x640 masks, official scoring at 1280x1024), the effect on (a) the hallucination rate (classes predicted with at least FLOOR but absent from the ground truth) and (b) the official
frame IoU / mIoU / mcIoU, SEPARATELY, plus the precision of the drops: of the (frame, class) pairs dropped, how many were truly absent from the ground truth (correct) and how many were present (harmful).
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
from sitr.reliability import SequenceCache  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

NC = 12
FLOOR = 0.0005
K_MEM = 3
TAU_MEM = 0.5
TAU = 0.64
NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma", "covered-kidney", "thread", "clamps", "suturing-needle",
         "suction-instrument", "intestine", "ultrasound-probe"]


def present(mask, floor=FLOOR):
    n = mask.size
    cnt = np.bincount(mask.ravel(), minlength=NC)[:NC]
    return {c for c in range(1, NC) if cnt[c] >= max(floor * n, 1)}


def run_sequence(masks, right, r, degenerate):
    """masks [T,H,W] base masks, right [T,H,W] right-view labels on the same grid, r [T] gate reliability. Returns new masks and per-frame dropped class lists."""
    T = len(masks)
    out = masks.copy()
    dropped = [[] for _ in range(T)]
    mem = []                                              # (t, class set) of reliable frames
    for t in range(T):
        while mem and mem[0][0] < t - K_MEM:
            mem.pop(0)
        left_set = present(masks[t])
        routed = (r[t] < TAU) and not degenerate[t]
        if routed:
            right_set = present(right[t])
            mem_set = set().union(*[s for _, s in mem]) if mem else set()
            drop = [c for c in sorted(left_set) if c not in right_set and c not in mem_set]
            if drop:
                dropped[t] = drop
                for c in drop:
                    out[t][masks[t] == c] = 0
        if r[t] >= TAU_MEM:
            mem.append((t, left_set))
    return out, dropped


def tally(base_masks, new_masks, dropped, gts, floor=FLOOR):
    """Hallucination events and official IoUs, before and after, for one sequence on one grid."""
    rows = []
    for t in range(len(base_masks)):
        gt = gts[t]
        gc = np.bincount(gt.ravel(), minlength=256)[:NC]
        pb, pa = present(base_masks[t], floor), present(new_masks[t], floor)
        hb = [c for c in pb if gc[c] == 0]
        ha = [c for c in pa if gc[c] == 0]
        ib, ia = official_class_ious(base_masks[t], gt, NC), official_class_ious(new_masks[t], gt, NC)
        rows.append(dict(hall_before=len(hb), hall_after=len(ha), pred_before=len(pb), pred_after=len(pa),
                         dropped=len(dropped[t]), dropped_absent=sum(1 for c in dropped[t] if gc[c] == 0), dropped_present=sum(1 for c in dropped[t] if gc[c] > 0),
                         iou_before=ib, iou_after=ia, changed=bool(dropped[t])))
    return rows


def summarise(rows):
    n = len(rows)
    fb = [np.mean(list(r["iou_before"].values())) for r in rows if r["iou_before"]]
    fa = [np.mean(list(r["iou_after"].values())) for r in rows if r["iou_after"]]

    def mc(key):
        per = {c: [] for c in range(1, NC)}
        for r in rows:
            for c, v in r[key].items():
                per[c].append(v)
        return float(np.mean([np.mean(v) for v in per.values() if v])), {NAMES[c]: (float(np.mean(v)) if v else None) for c, v in per.items()}
    mcb, pcb = mc("iou_before")
    mca, pca = mc("iou_after")
    d = dict(frames=n, frames_changed=int(sum(r["changed"] for r in rows)), frames_changed_pct=100 * float(np.mean([r["changed"] for r in rows])),
             classes_dropped=int(sum(r["dropped"] for r in rows)), dropped_truly_absent_from_ground_truth=int(sum(r["dropped_absent"] for r in rows)),
             dropped_present_in_ground_truth=int(sum(r["dropped_present"] for r in rows)),
             hallucination_events_before=int(sum(r["hall_before"] for r in rows)), hallucination_events_after=int(sum(r["hall_after"] for r in rows)),
             frames_with_event_pct_before=100 * float(np.mean([r["hall_before"] > 0 for r in rows])), frames_with_event_pct_after=100 * float(np.mean([r["hall_after"] > 0 for r in rows])),
             share_of_predicted_invented_before=float(sum(r["hall_before"] for r in rows) / max(sum(r["pred_before"] for r in rows), 1)),
             share_of_predicted_invented_after=float(sum(r["hall_after"] for r in rows) / max(sum(r["pred_after"] for r in rows), 1)),
             mIoU_before=float(np.mean(fb)), mIoU_after=float(np.mean(fa)), mIoU_change_points=100 * float(np.mean(fa) - np.mean(fb)),
             mcIoU_before=mcb, mcIoU_after=mca, mcIoU_change_points=100 * (mca - mcb), per_class_before=pcb, per_class_after=pca)
    d["drop_precision"] = d["dropped_truly_absent_from_ground_truth"] / max(d["classes_dropped"], 1)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work_v2")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskF/a4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = {s.key: s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("train", "test"))}
    R = {"definition": __doc__, "floor": FLOOR, "K_memory": K_MEM, "tau_memory": TAU_MEM, "tau_route": TAU}
    # ------------------------------------------------ out of fold, 256x320 cache grid
    rows = []
    for f in sorted(os.listdir(os.path.join(a.work, "cache_oof"))):
        if not f.endswith(".npz"):
            continue
        c = SequenceCache.load(os.path.join(a.work, "cache_oof", f))
        s = seqs[c.name]
        z = np.load(os.path.join(a.work, "actions", f"oof__{c.name.replace('/', '_')}.npz"), allow_pickle=True)
        meta = json.load(open(os.path.join(a.work, "stereo", c.name.replace("/", "_"), "stereo_meta.json")))
        h, w = c.masks.shape[1:]
        right = np.stack([cv2.resize(np.load(os.path.join(a.work, "right_view", c.name.replace("/", "_"), os.path.splitext(fn)[0] + ".npz"))["label"], (w, h), interpolation=cv2.INTER_NEAREST)
                          for fn in meta["frames"]])
        gts = [cv2.resize(read_mask(l), (w, h), interpolation=cv2.INTER_NEAREST) for l in s.labels]
        new, dropped = run_sequence(c.masks, right, np.asarray(z["r"], float), np.asarray(z["degenerate"], bool))
        rows += tally(c.masks, new, dropped, gts)
        print(f"  {c.name}: {sum(1 for d in dropped if d)} frames changed", flush=True)
    R["oof"] = summarise(rows)
    # ------------------------------------------------ test, saved 512x640 masks, official scoring at full resolution
    rows = []
    new_dir = os.path.join(a.work, "actions_a4", "test_masks")
    for s in sorted([x for x in seqs.values() if x.split == "test"], key=lambda x: x.index):
        z = np.load(os.path.join(a.work, "actions", f"test__{s.split}_{s.name}.npz"), allow_pickle=True)
        meta = json.load(open(os.path.join(a.work, "stereo", f"{s.split}_{s.name}", "stereo_meta.json")))
        stems = [os.path.splitext(os.path.basename(p))[0] for p in s.frames]
        base = np.stack([np.load(os.path.join(a.work, "actions", "test_masks", f"{s.split}_{s.name}", st + ".npz"))["A0"] for st in stems])
        right = np.stack([np.load(os.path.join(a.work, "right_view", f"{s.split}_{s.name}", st + ".npz"))["label"] for st in stems])
        new, dropped = run_sequence(base, right, np.asarray(z["r"], float), np.asarray(z["degenerate"], bool))
        gts = []
        up_b, up_n = [], []
        for t, l in enumerate(s.labels):
            g = read_mask(l)
            gts.append(g)
            up_b.append(cv2.resize(base[t], (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST))
            up_n.append(cv2.resize(new[t], (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST))
        rows += tally(up_b, up_n, dropped, gts)
        os.makedirs(os.path.join(new_dir, f"{s.split}_{s.name}"), exist_ok=True)
        np.savez_compressed(os.path.join(new_dir, f"{s.split}_{s.name}", "A4.npz"), A4=new, dropped=np.array([",".join(map(str, d)) for d in dropped]))
        print(f"  {s.key}: {sum(1 for d in dropped if d)} frames changed", flush=True)
    R["test"] = summarise(rows)
    json.dump(R, open(os.path.join(a.out, "a4.json"), "w"), indent=1)
    for g in ("oof", "test"):
        d = R[g]
        print(f"{g:5s} frames changed {d['frames_changed']}/{d['frames']} ({d['frames_changed_pct']:.1f}%)  dropped {d['classes_dropped']} (truly absent {d['dropped_truly_absent_from_ground_truth']}, present {d['dropped_present_in_ground_truth']}, precision {100*d['drop_precision']:.0f}%)")
        print(f"      hallucination: frames with event {d['frames_with_event_pct_before']:.1f}% -> {d['frames_with_event_pct_after']:.1f}%; invented share of predicted {100*d['share_of_predicted_invented_before']:.1f}% -> {100*d['share_of_predicted_invented_after']:.1f}%")
        print(f"      mIoU {d['mIoU_before']:.4f} -> {d['mIoU_after']:.4f} ({d['mIoU_change_points']:+.2f} pts); mcIoU {d['mcIoU_before']:.4f} -> {d['mcIoU_after']:.4f} ({d['mcIoU_change_points']:+.2f} pts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
