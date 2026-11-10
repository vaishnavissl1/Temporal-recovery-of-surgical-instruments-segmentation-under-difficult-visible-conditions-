#!/usr/bin/env python3
"""
TASK 4, part B -- is the objective difficulty index real?

    python -m tools.difficulty_report --oof work_v2/cache_oof_stereo \
        --stereo work_v2/stereo --prepared work_v2/prepared_512x640_full \
        [--test work_v2/cache_test_stereo] --out results/task4

The index (sitr/difficulty.py) uses NO labels and NO model output. Here it is
compared with segmentation error on HELD-OUT frames: out-of-fold predictions for
the training sequences, and the final model's predictions for the official test
sequences. Reference distribution for the rank-normalisation: the TRAINING frames
only (never the test frames).

Error is measured two ways: 1 - instrument IoU (the gate's target) and
1 - official frame IoU (the challenge metric, here on the 256x320 mask grid; the
full-resolution number is Task 6). Both Pearson and Spearman are reported, pooled
AND within sequences (so a sequence-level effect cannot pass for frame-level
signal), plus each cue alone and the quartile table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.difficulty import (CUE_NAMES, DifficultyIndex, IMAGE_CUE_NAMES,  # noqa: E402
                             cue_table, stereo_cues)
from sitr.metrics import official_frame_iou, pearson, spearman  # noqa: E402
from sitr.reliability import SequenceCache  # noqa: E402


def load_dir(d, prefix):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            c = SequenceCache.load(os.path.join(d, f))
            if c.name.startswith(prefix):
                out[c.name] = c
    return out


def frames_table(caches, stereo_dir, prepared):
    """Per-frame stereo/image cues and errors for a set of sequence caches."""
    rows = dict(seq=[], t=[], sl=[], oc=[], sp=[], img=[], r_star=[], off=[], q=[], area=[])
    for name, c in sorted(caches.items()):
        split, sn = name.split("/")
        sd = os.path.join(stereo_dir, f"{split}_{sn}")
        meta = json.load(open(os.path.join(sd, "stereo_meta.json")))
        assert len(meta["frames"]) == len(c), name
        for t, fn in enumerate(meta["frames"]):
            z = np.load(os.path.join(sd, os.path.splitext(fn)[0] + ".npz"))
            s = stereo_cues(z["lr_ok"], z["occluded"], z["specular"], z["in_view"])
            rows["sl"].append(s["stereo_lr_fail"]); rows["oc"].append(s["occluded"]); rows["sp"].append(s["specular"])
            rows["img"].append(z["image_cues"])
            gt = cv2.imread(os.path.join(prepared, split, sn, "labels", fn), cv2.IMREAD_UNCHANGED)
            h, w = c.masks[t].shape
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)       # nearest: label map
            rows["off"].append(official_frame_iou(c.masks[t], gt, c.num_classes))
            rows["seq"].append(name); rows["t"].append(t)
            rows["r_star"].append(float(c.r_star[t])); rows["q"].append(float(c.q[t])); rows["area"].append(int(c.area[t]))
    out = {k: np.array(v) for k, v in rows.items() if k != "off"}
    out["off"] = np.array([np.nan if v is None else v for v in rows["off"]], float)
    return out


def corr_block(idx, err, seq):
    ok = np.isfinite(idx) & np.isfinite(err)
    idx, err, seq = idx[ok], err[ok], seq[ok]
    out = dict(n=int(ok.sum()), pearson=pearson(idx, err), spearman=spearman(idx, err))
    ws = []
    for s in np.unique(seq):
        m = seq == s
        if m.sum() >= 20 and idx[m].std() > 0 and err[m].std() > 0:
            ws.append(spearman(idx[m], err[m]))
    out["within_seq_spearman_mean"] = float(np.mean(ws))
    out["within_seq_spearman_each"] = {str(s): float(v) for s, v in zip([x for x in np.unique(seq)], ws)} if len(ws) == len(np.unique(seq)) else None
    rng = np.random.default_rng(0)
    bs = [np.mean(rng.choice(ws, len(ws), replace=True)) for _ in range(2000)]
    out["within_seq_spearman_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["n_seq_positive"] = int(sum(v > 0 for v in ws)); out["n_seq"] = len(ws)
    return out


def quartile_table(score, r_star, off, edges=None):
    q = DifficultyIndex.quartile_of(score, edges)
    rows = []
    for k in range(4):
        m = q == k
        rows.append(dict(quartile=k + 1, n=int(m.sum()), mean_index=float(score[m].mean()),
                         mean_instr_iou=float(r_star[m].mean()),
                         bad_frame_rate=float((r_star[m] < 0.5).mean()),
                         mean_official_iou=float(np.nanmean(off[m]))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--stereo", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--out", default="results/task4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = frames_table(load_dir(a.oof, "train/"), a.stereo, a.prepared)
    cues_tr = cue_table(dict(stereo_lr_fail=tr["sl"], occluded=tr["oc"], specular=tr["sp"]), np.stack(tr["img"]), tr["seq"])
    idx = DifficultyIndex().fit(cues_tr)
    R = {"cues": list(CUE_NAMES), "image_cues": list(IMAGE_CUE_NAMES)}

    def analyse(tag, tab, cues, edges_from_own=True):
        score = idx.score(cues)
        ranks = idx.cue_ranks(cues)
        block = {"n_frames": int(len(score)), "mean_instr_iou": float(tab["r_star"].mean()),
                 "mean_official_iou_masks256": float(np.nanmean(tab["off"])),
                 "vs_instrument_error": corr_block(score, 1 - tab["r_star"], tab["seq"]),
                 "vs_official_error": corr_block(score, 1 - tab["off"], tab["seq"]),
                 "confidence_reference": dict(pearson_q_vs_iou=pearson(tab["q"][tab["area"] > 0], tab["r_star"][tab["area"] > 0])),
                 "per_cue": {}}
        for c in CUE_NAMES:
            block["per_cue"][c] = dict(vs_instrument_error=corr_block(ranks[c], 1 - tab["r_star"], tab["seq"]),
                                       vs_official_error=corr_block(ranks[c], 1 - tab["off"], tab["seq"]))
        block["quartiles"] = quartile_table(score, tab["r_star"], tab["off"])
        R[tag] = block
        v, o = block["vs_instrument_error"], block["vs_official_error"]
        print(f"\n[{tag}] {block['n_frames']} frames, mean instrument IoU {block['mean_instr_iou']:.3f}, official IoU (256x320) {block['mean_official_iou_masks256']:.3f}")
        print(f"  index vs (1 - instrument IoU): Pearson {v['pearson']:+.3f}  Spearman {v['spearman']:+.3f}  "
              f"within-sequence Spearman {v['within_seq_spearman_mean']:+.3f} [{v['within_seq_spearman_ci95'][0]:+.3f}, {v['within_seq_spearman_ci95'][1]:+.3f}] "
              f"({v['n_seq_positive']}/{v['n_seq']} sequences positive)")
        print(f"  index vs (1 - official IoU):   Pearson {o['pearson']:+.3f}  Spearman {o['spearman']:+.3f}  "
              f"within-sequence Spearman {o['within_seq_spearman_mean']:+.3f}")
        print(f"  (reference: model confidence q vs IoU Pearson {block['confidence_reference']['pearson_q_vs_iou']:+.3f})")
        print("  per cue, Spearman vs (1 - instrument IoU): pooled / within-sequence")
        for c in CUE_NAMES:
            p = block["per_cue"][c]["vs_instrument_error"]
            print(f"    {c:16s} {p['spearman']:+.3f} / {p['within_seq_spearman_mean']:+.3f}")
        print("  quartile (1 = easiest .. 4 = hardest): n, mean index, mean instrument IoU, bad-frame rate, mean official IoU")
        for r in block["quartiles"]:
            print(f"    Q{r['quartile']}: {r['n']:4d}  {r['mean_index']:.3f}  {r['mean_instr_iou']:.3f}  {r['bad_frame_rate']:.3f}  {r['mean_official_iou']:.3f}")

    analyse("oof_train_sequences", tr, cues_tr)
    if a.test:
        te = frames_table(load_dir(a.test, "test/"), a.stereo, a.prepared)
        cues_te = cue_table(dict(stereo_lr_fail=te["sl"], occluded=te["oc"], specular=te["sp"]), np.stack(te["img"]), te["seq"])
        analyse("official_test_sequences", te, cues_te)
    json.dump(R, open(os.path.join(a.out, "difficulty_report.json"), "w"), indent=1)
    print(f"\nwrote {a.out}/difficulty_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
