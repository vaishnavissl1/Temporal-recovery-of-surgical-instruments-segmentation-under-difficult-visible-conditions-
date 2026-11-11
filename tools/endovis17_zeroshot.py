#!/usr/bin/env python3
"""
TASK P.3 option B -- EndoVis 2017 as a ZERO-SHOT, EVALUATION-ONLY test of the segmenter and the reliability gate.

    python -m tools.endovis17_zeroshot --data <endovis2017 folder> --release work_v2/release --out results/taskP3

Rules (hard, and fixed here before any result was seen):
  * The models are never trained or fine-tuned on this data. No threshold is re-tuned: tau = 0.64, the memory threshold, the temperature 2.688 and the gate are exactly the released ones.
  * It is scored ONCE. This file is run once for the reported numbers.
  * Data: the CC BY 4.0 preprocessed copy of EndoVis 2017 on Zenodo (record 10527017): 512x512 RGB frames and instrument-TYPE label maps, left view only, no stereo and no part labels.
    Test part used: val1-val8 (75 frames each) and val10 (300 frames) = 900 frames in 9 sequences. val9 is EXCLUDED: its archive holds 300 labels but only one image.
    The training part (1800 frames) is not used, so nothing here overlaps anything the model saw (it never saw EndoVis 2017 at all).
  * Class mapping. EndoVis 2017 labels instrument TYPES (1 bipolar forceps, 2 prograsp forceps, 3 large needle driver, 4 vessel sealer, 5 grasping retractor,
    6 monopolar curved scissors, 7 other) and background; our 12 classes label instrument PARTS and anatomy. The only common ground is BINARY instrument versus background:
      ground truth: label > 0 (all seven types) -> instrument; 0 -> background.
      prediction  : our classes {1 shaft, 2 clasper, 3 wrist, 9 suction-instrument, 11 ultrasound-probe} -> instrument; everything else -> background.
    EXCLUDED, with no 2017 equivalent and therefore mapped to background: thread (6), clamps (7), suturing-needle (8) (suture material and clips are not annotated as instruments in 2017),
    and the anatomy classes background-tissue (0), kidney-parenchyma (4), covered-kidney (5), intestine (10) (2017 has no anatomy labels). The seven instrument TYPES have no counterpart in our
    classes, so no per-type or per-part score exists. Frames of this copy are labelled only for some instruments (visible in the data), a limitation of the label set.
  * Geometry. Frames are 512x512; they are resized to the segmenter's 512x640 input (bilinear), the predicted mask is resized back to 512x512 by nearest neighbour for scoring. Choice made in advance;
    the aspect ratio of the preprocessed copy is not documented, and this is a stated source of mismatch.
  * Temporal regime. Frames within a sequence are consecutive samples of a ~2 Hz release (2018: 1 Hz); the gate's temporal indicators run on them unchanged.
  * Frame IoU = binary instrument IoU; both empty -> 1.0, one empty -> 0.0 (the repo's empty-frame convention). Bad frame: IoU < 0.5. Risk = 1 - IoU.
  * Detectors compared on the SAME frames: the released gate r_t; max-softmax (the q indicator: mean top-class probability over predicted instrument pixels, at the stream temperature);
    the entropy confidence and margin indicators. AUROC for bad frames and AURC, with 95% intervals that resample whole sequences (2000 draws, seed 0; 9 clusters).
  * Reference: the SAME mapping, metric and detectors on the 997 EndoVis 2018 test frames (stored predictions and gate scores), so the drop is measured, not assumed. A large drop is the expected
    result and is reported as such.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import SequenceInfo, discover_sequences  # noqa: E402
from sitr.metrics import aurc, roc_auc  # noqa: E402
from sitr.runtime import ReleaseBundle  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

PRED_INSTR = [1, 2, 3, 9, 11]
SEQS = [f"val{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 10)]
EXCLUDED_PRED = {"thread": 6, "clamps": 7, "suturing-needle": 8, "background-tissue": 0, "kidney-parenchyma": 4, "covered-kidney": 5, "intestine": 10}


def binary_iou(pred_inst, gt_inst):
    p, g = int(pred_inst.sum()), int(gt_inst.sum())
    if p == 0 and g == 0:
        return 1.0
    return float((pred_inst & gt_inst).sum()) / float((pred_inst | gt_inst).sum())


def detector_table(rows, score_keys, n_boot=2000, seed=0):
    """rows: dict with arrays iou, seq and score arrays. Returns metrics per detector with sequence-clustered intervals."""
    iou, seq = np.asarray(rows["iou"], float), np.asarray(rows["seq"])
    risk, bad = 1.0 - iou, (iou < 0.5).astype(int)
    uniq = np.unique(seq)
    idx = {u: np.where(seq == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)]) for _ in range(n_boot)]
    out = {}
    for k in score_keys:
        s = np.asarray(rows[k], float)
        A, U = [], []
        for ii in draws:
            if bad[ii].min() == bad[ii].max():
                continue
            A.append(roc_auc(-s[ii], bad[ii]))
            U.append(aurc(s[ii], risk[ii]))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        out[k] = dict(auroc=roc_auc(-s, bad), auroc_ci95=ci(A), aurc=aurc(s, risk), aurc_ci95=ci(U))
    out["_aurc_floor"] = aurc(-risk, risk)
    out["_aurc_random"] = float(risk.mean())
    out["_bad_rate"] = float(bad.mean())
    out["_n_frames"] = int(len(iou))
    out["_n_sequences"] = int(len(uniq))
    return out


def seg_summary(rows):
    iou, seq = np.asarray(rows["iou"], float), np.asarray(rows["seq"])
    return dict(mean_binary_iou=float(iou.mean()), median=float(np.median(iou)), frac_below_0_5=float((iou < 0.5).mean()),
                frames_with_no_gt_instrument_pct=100 * float(np.mean(rows["gt_area"] == 0)),
                per_sequence={str(u): float(iou[seq == u].mean()) for u in np.unique(seq)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder 'endovis2017' holding val1..val10")
    ap.add_argument("--release", default="work_v2/release")
    ap.add_argument("--root18", default=None, help="EndoVis 2018 dataset root (for the reference on the 2018 test frames)")
    ap.add_argument("--actions", default="work_v2/actions")
    ap.add_argument("--out", default="results/taskP3")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-frames", type=int, default=0, help="debug only: do not use for the reported run")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bundle = ReleaseBundle.load(a.release, device=a.device)
    names = list(bundle.manifest["indicator_names"])
    ix = {n: i for i, n in enumerate(names)}
    R17 = {k: [] for k in ("seq", "iou", "gt_area", "pred_area", "r", "q", "ent", "margin", "frame")}
    t0 = time.time()
    for sq in SEQS:
        d = os.path.join(a.data, sq)
        files = sorted(os.listdir(os.path.join(d, "image")))
        if a.limit_frames:
            files = files[: a.limit_frames]
        k = int(sq.replace("val", ""))
        seq = SequenceInfo(name=f"seq_{k}", split="test", index=k, frame_dir=os.path.join(d, "image"), label_dir=os.path.join(d, "label"),
                           frames=[os.path.join(d, "image", f) for f in files], labels=[None] * len(files), prepared=True)
        for t, st in enumerate(bundle.stream(seq, batch_size=4, with_gt=False)):
            gt = cv2.imread(os.path.join(d, "label", files[t]), cv2.IMREAD_UNCHANGED) > 0
            pi = np.isin(st.mask, PRED_INSTR).astype(np.uint8)
            pi = cv2.resize(pi, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            R17["seq"].append(sq); R17["frame"].append(files[t])
            R17["iou"].append(binary_iou(pi, gt)); R17["gt_area"].append(int(gt.sum())); R17["pred_area"].append(int(pi.sum()))
            R17["r"].append(float(st.reliability))
            R17["q"].append(float(st.indicators[ix["q_confidence"]])); R17["ent"].append(float(st.indicators[ix["H_entropy_conf"]]))
            R17["margin"].append(float(st.indicators[ix["M_margin"]]))
        print(f"  {sq}: {len(files)} frames   {(time.time()-t0)/60:.1f} min", flush=True)
    R17 = {k: np.asarray(v) for k, v in R17.items()}
    keys = ["r", "q", "ent", "margin"]
    out = {"definition": __doc__, "tau": bundle.tau, "excluded_sequence": "val9 (300 labels, 1 image in the archive)", "pred_instrument_classes": PRED_INSTR,
           "excluded_predicted_classes_mapped_to_background": EXCLUDED_PRED, "endovis2017": dict(segmentation=seg_summary(R17), detectors=detector_table(R17, keys),
                                                                                              flagged_pct=100 * float((R17["r"] < bundle.tau).mean()))}
    # reference on the EndoVis 2018 test frames, same mapping, same detectors
    if a.root18:
        seqs18 = sorted(discover_sequences(a.root18, require_labels=True, apply_repairs=True, splits=("test",)), key=lambda s: s.index)
        R18 = {k: [] for k in ("seq", "iou", "gt_area", "r", "q", "ent", "margin")}
        for s in seqs18:
            z = np.load(os.path.join(a.actions, f"test__{s.split}_{s.name}.npz"), allow_pickle=True)
            for t, (fp, lp) in enumerate(zip(s.frames, s.labels)):
                stem = os.path.splitext(os.path.basename(fp))[0]
                m = np.load(os.path.join(a.actions, "test_masks", f"{s.split}_{s.name}", stem + ".npz"))["A0"]
                gt = read_mask(lp)
                gt = cv2.resize(gt, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
                gi = np.isin(gt, [1, 2, 3, 6, 7, 8, 9, 11])          # 2018 ground truth instrument classes (same convention as the rest of this study)
                gi = np.isin(gt, PRED_INSTR)                           # restricted to the classes that map to the 2017 notion of instrument
                R18["seq"].append(s.key); R18["iou"].append(binary_iou(np.isin(m, PRED_INSTR), gi)); R18["gt_area"].append(int(gi.sum()))
                ind = z["ind9"][t]
                R18["r"].append(float(z["r"][t])); R18["q"].append(float(ind[ix["q_confidence"]])); R18["ent"].append(float(ind[ix["H_entropy_conf"]])); R18["margin"].append(float(ind[ix["M_margin"]]))
        R18 = {k: np.asarray(v) for k, v in R18.items()}
        out["endovis2018_reference"] = dict(segmentation=seg_summary(R18), detectors=detector_table(R18, keys), flagged_pct=100 * float((R18["r"] < bundle.tau).mean()))
    if not a.limit_frames:
        json.dump(out, open(os.path.join(a.out, "endovis17_zeroshot.json"), "w"), indent=1)
        import csv
        with open(os.path.join(a.out, "endovis17_per_frame.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "frame", "binary_iou", "gt_instrument_pixels", "predicted_instrument_pixels", "gate_r", "max_softmax_q", "entropy_conf", "margin"])
            for i in range(len(R17["iou"])):
                w.writerow([R17["seq"][i], R17["frame"][i], f"{R17['iou'][i]:.5f}", R17["gt_area"][i], R17["pred_area"][i], f"{R17['r'][i]:.5f}", f"{R17['q'][i]:.5f}", f"{R17['ent'][i]:.5f}", f"{R17['margin'][i]:.5f}"])
    for g in ("endovis2017", "endovis2018_reference"):
        if g in out:
            s = out[g]["segmentation"]; d = out[g]["detectors"]
            print(f"{g}: mean binary IoU {s['mean_binary_iou']:.4f}  frames<0.5 {100*s['frac_below_0_5']:.1f}%  flagged {out[g]['flagged_pct']:.1f}%  frames={d['_n_frames']}")
            for k in keys:
                print(f"   {k:7s} AUROC {d[k]['auroc']:.3f} {d[k]['auroc_ci95']}  AURC {d[k]['aurc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
