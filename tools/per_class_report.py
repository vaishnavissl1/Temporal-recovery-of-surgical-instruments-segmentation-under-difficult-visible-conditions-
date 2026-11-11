#!/usr/bin/env python3
"""
TASK J -- per-class, per-sequence results and the rig-leakage limitation.

    python -m tools.per_class_report --actions work_v2/actions --root <endovis2018_cvdataset> --out results/taskJ [--cldice]

Part 1 (test, M1 = the single-model system of record; same official rule as everywhere: full 1280x1024, class 0 never scored, a class
counted only if it is in the ground truth). For every class: IoU over the frames in which it is present, per sequence and pooled; per-sequence
mIoU (mean over frames of the per-frame class mean) and mcIoU (per class over frames, then over classes); which classes drop most on test
sequence 4 relative to the other three. With --cldice, centreline Dice per class (skimage skeletonize, full resolution) as well.

Part 2 (rig leakage, analysis only, no retraining). Camera rigs are identified from the calibration files themselves (parsed values, not file
names). For every training sequence: its rig and the number of frames in which each class appears. For every leave-one-rig-out split of the
training sequences: which classes are left with no training frame at all (the rare-class constraint of DATASET_FACTS.md section 3). Which
training rig each test sequence shares a rig with. No fold is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.metrics import _class_counts  # noqa: E402
from sitr.stereo import calibration_path, parse_calibration  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

NC = 12
NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma", "covered-kidney", "thread",
         "clamps", "suturing-needle", "suction-instrument", "intestine", "ultrasound-probe"]


def cl_dice(pred_bin, gt_bin):
    from skimage.morphology import skeletonize
    if not gt_bin.any():
        return None
    if not pred_bin.any():
        return 0.0
    sp, sg = skeletonize(pred_bin), skeletonize(gt_bin)
    tp = float((sp & gt_bin).sum() / max(sp.sum(), 1))
    ts = float((sg & pred_bin).sum() / max(sg.sum(), 1))
    return 0.0 if tp + ts == 0 else 2 * tp * ts / (tp + ts)


def cl_job(args):
    mask_path, label_path = args
    pred = np.load(mask_path)["A0"]
    gt = read_mask(label_path)
    full = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return {c: cl_dice(full == c, gt == c) for c in range(1, NC) if (gt == c).any()}


RIG_TOL = 0.05     # a stated choice: two calibration files describe the same rig if every parameter agrees within 0.05 (px or mm)


def calib_vector(calib_path):
    c = parse_calibration(calib_path)
    keys = ("Width", "Height", "Camera-0-F", "Camera-0-C", "Camera-1-F", "Camera-1-C", "Camera-1-K", "Extrinsic-Omega", "Extrinsic-T")
    return np.array([float(x) for k in keys for x in c.get(k, [])], float)


def cluster_rigs(vectors):
    """Greedy grouping by the largest absolute parameter difference. Returns (group index per key, within-group max difference,
    smallest difference between two different groups). The last two numbers show that RIG_TOL separates the groups cleanly."""
    keys = list(vectors)
    group, reps = {}, []
    for k in keys:
        for gi, r in enumerate(reps):
            if vectors[k].shape == r.shape and float(np.abs(vectors[k] - r).max()) <= RIG_TOL:
                group[k] = gi
                break
        else:
            reps.append(vectors[k])
            group[k] = len(reps) - 1
    within = max((float(np.abs(vectors[k] - reps[group[k]]).max()) for k in keys), default=0.0)
    between = min((float(np.abs(reps[i] - reps[j]).max()) if reps[i].shape == reps[j].shape else float("inf")
                   for i in range(len(reps)) for j in range(i + 1, len(reps))), default=float("inf"))
    return group, within, between


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", default="work_v2/actions")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/taskJ")
    ap.add_argument("--cldice", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("train", "test"))
    seqs = sorted(seqs, key=lambda s: (s.split != "train", s.index))
    R = {}

    # ------------------------------------------------------------------ part 1: test
    test = [s for s in seqs if s.split == "test"]
    per = {s.key: {c: [] for c in range(1, NC)} for s in test}
    frame_mean = {s.key: [] for s in test}
    jobs = []
    for s in test:
        for t, (fp, lp) in enumerate(zip(s.frames, s.labels)):
            stem = os.path.splitext(os.path.basename(fp))[0]
            mp_ = os.path.join(a.actions, "test_masks", f"{s.split}_{s.name}", stem + ".npz")
            pred = np.load(mp_)["A0"]
            gt = read_mask(lp)
            full = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            inter, pr, gc = _class_counts(full, gt, NC)
            ious = {}
            for c in range(1, NC):
                if gc[c] > 0:
                    ious[c] = float(inter[c]) / float(gc[c] + pr[c] - inter[c])
                    per[s.key][c].append(ious[c])
            if ious:
                frame_mean[s.key].append(float(np.mean(list(ious.values()))))
            jobs.append((s.key, mp_, lp))
        print(f"  {s.key}: scored", flush=True)
    keys = [s.key for s in test]
    pooled = {c: [v for k in keys for v in per[k][c]] for c in range(1, NC)}
    R["test_per_class"] = {NAMES[c]: dict(
        pooled_iou=(float(np.mean(pooled[c])) if pooled[c] else None), n_frames=len(pooled[c]),
        per_sequence={k: (dict(iou=float(np.mean(per[k][c])), n_frames=len(per[k][c])) if per[k][c] else None) for k in keys})
        for c in range(1, NC)}
    scored = [c for c in range(1, NC) if pooled[c]]
    R["test_active_classes"] = [NAMES[c] for c in scored]
    R["test_classes_with_no_ground_truth_pixels"] = [NAMES[c] for c in range(1, NC) if not pooled[c]]
    R["test_per_sequence"] = {}
    for k in keys:
        cls = {c: float(np.mean(per[k][c])) for c in range(1, NC) if per[k][c]}
        R["test_per_sequence"][k] = dict(mIoU=float(np.mean(frame_mean[k])), n_frames=len(frame_mean[k]), mcIoU=float(np.mean(list(cls.values()))),
                                         n_classes_scored=len(cls))
    R["test_pooled"] = dict(mIoU=float(np.average([R["test_per_sequence"][k]["mIoU"] for k in keys], weights=[len(frame_mean[k]) for k in keys])),
                            mcIoU=float(np.mean([np.mean(pooled[c]) for c in scored])), n_classes_scored=len(scored))
    # sequence 4 against the others
    others = [k for k in keys if k != "test/seq_4"]
    drops = []
    for c in range(1, NC):
        if per["test/seq_4"][c]:
            o = [v for k in others for v in per[k][c]]
            drops.append(dict(cls=NAMES[c], seq4_iou=float(np.mean(per["test/seq_4"][c])), seq4_frames=len(per["test/seq_4"][c]),
                              others_iou=(float(np.mean(o)) if o else None), others_frames=len(o),
                              drop=(float(np.mean(o) - np.mean(per["test/seq_4"][c])) if o else None)))
    R["test_seq4_vs_others"] = sorted(drops, key=lambda d: -(d["drop"] if d["drop"] is not None else -9))
    if a.cldice:
        with mp.Pool(a.workers) as pool:
            res = pool.map(cl_job, [(m, l) for _, m, l in jobs], chunksize=4)
        cl = {k: {c: [] for c in range(1, NC)} for k in keys}
        for (k, _, _), r in zip(jobs, res):
            for c, v in r.items():
                if v is not None:
                    cl[k][c].append(v)
        R["test_cldice"] = {NAMES[c]: dict(pooled=(float(np.mean([v for k in keys for v in cl[k][c]])) if any(cl[k][c] for k in keys) else None),
                                           per_sequence={k: (float(np.mean(cl[k][c])) if cl[k][c] else None) for k in keys}) for c in range(1, NC)}

    # ------------------------------------------------------------------ part 2: rigs
    import hashlib
    vecs = {s.key: calib_vector(calibration_path(s.frame_dir)) for s in seqs}
    raw_md5 = {s.key: hashlib.md5(open(calibration_path(s.frame_dir), "rb").read()).hexdigest()[:8] for s in seqs}
    grp, within, between = cluster_rigs(vecs)
    order = sorted(set(grp.values()), key=lambda g: min(int(k.split("seq_")[1]) + (100 if k.startswith("test") else 0) for k in grp if grp[k] == g))
    labels = {g: "ABCDEFGHIJ"[i] for i, g in enumerate(order)}
    by_seq = {k: labels[g] for k, g in grp.items()}
    rigs = {}
    for k, r in by_seq.items():
        rigs.setdefault(r, []).append(k)
    class_frames = {}
    for s in seqs:
        if s.split != "train":
            continue
        cnt = np.zeros(NC, int)
        for lp in s.labels:
            gt = read_mask(lp)
            cnt += np.bincount(np.unique(gt[gt < NC]), minlength=NC)[:NC]
        class_frames[s.key] = cnt
    train_keys = [k for k in class_frames]
    rig_groups_train = {r: [k for k in v if k in class_frames] for r, v in rigs.items()}
    R["rigs"] = dict(
        tolerance=RIG_TOL, max_within_group_parameter_difference=within, min_between_group_parameter_difference=between,
        n_distinct_calibration_files=len(set(raw_md5.values())), calibration_file_md5_per_sequence=raw_md5,
        n_rigs=len(rigs), n_rigs_with_a_training_sequence=sum(1 for v in rig_groups_train.values() if v),
        groups={r: sorted(v) for r, v in rigs.items()},
        train_groups={g: sorted(v) for g, v in rig_groups_train.items() if v},
        test_sequence_rig={k: dict(rig=by_seq[k], training_sequences_on_the_same_rig=sorted(x for x in rigs[by_seq[k]] if x.startswith("train")))
                           for k in keys},
        train_frames_per_class_per_sequence={k: {NAMES[c]: int(v[c]) for c in range(1, NC)} for k, v in class_frames.items()})
    loro = {}
    for g, held in rig_groups_train.items():
        if not held:
            continue
        rest = [k for k in train_keys if k not in held]
        left = {NAMES[c]: int(sum(class_frames[k][c] for k in rest)) for c in range(1, NC)}
        loro[g] = dict(held_out_sequences=sorted(held), training_sequences_left=len(rest),
                       classes_with_zero_training_frames=[n for n, v in left.items() if v == 0],
                       classes_in_held_out_rig_but_not_elsewhere=[NAMES[c] for c in range(1, NC)
                                                                 if sum(class_frames[k][c] for k in held) > 0 and left[NAMES[c]] == 0],
                       classes_below_20_training_frames=[n for n, v in left.items() if 0 < v < 20])
    R["leave_one_rig_out"] = loro
    R["leave_one_rig_out_feasible"] = all(not v["classes_with_zero_training_frames"] for v in loro.values())
    json.dump(R, open(os.path.join(a.out, "per_class_and_rigs.json"), "w"), indent=1)
    print(json.dumps({k: R[k] for k in ("test_per_sequence", "test_pooled", "test_active_classes", "test_classes_with_no_ground_truth_pixels")}, indent=1))
    print("rigs:", json.dumps(R["rigs"]["groups"]))
    print("LORO:", json.dumps({g: (v["classes_with_zero_training_frames"], v["classes_below_20_training_frames"]) for g, v in loro.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
