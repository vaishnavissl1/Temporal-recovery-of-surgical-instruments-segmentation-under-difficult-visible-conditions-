#!/usr/bin/env python3
"""
TASK F.1 and F.2 -- mcIoU beside mIoU for every model, and the hallucination rate. CPU only, from files that already exist.

    python -m tools.mciou_hallucination --actions work_v2/actions --root <endovis2018_cvdataset> --out results/taskF

F.1 mcIoU = per class over the frames where the class is in the ground truth, then the mean over classes (results/task6/final_results.json already holds the per-class IoU of M1..M6). Under the
official rule class 0 is never scored and suction-instrument has no test pixels, so 10 of the 11 foreground classes are scored (11 of 12 only if background were counted).

F.2 Hallucination. The official metric counts a class only if it is in the ground truth (utils.py:143), so a class the model INVENTS costs nothing. A HALLUCINATION EVENT is a (frame, class)
pair with the class absent from the ground truth and predicted with area >= FLOOR pixels at 1280x1024. FLOOR = 0.05% of the frame (655 px, about a 26x26 patch), a stated choice; the
rates at 0 (any pixel) and 0.5% of the frame are stored as a sensitivity check and never substituted. Reported per model on the 997 test frames:
  frames_with_event_pct   share of frames with at least one invented class
  events_per_frame        mean number of invented classes per frame
  pooled_share_of_predicted   events / all predicted (frame, class) pairs with area >= FLOOR   (what fraction of what the model says is present is invented)
  per_class_false_presence    for each class, the share of the frames where it is ABSENT from the ground truth in which it is nevertheless predicted with area >= FLOOR
M1..M6 are scored from the masks saved by tools/make_action_data.py (A0, A1, A2, A3, FUSE, FUSE+refine; M6 uses the selector's choice per frame from results/task5/choices_test.npz).
Only the test frames have saved masks, so this is a test-set measurement.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.metrics import official_class_ious  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

NC = 12
KEY_OF = {"M1": "A0", "M2": "A1", "M3": "A2", "M4": "FUSE", "M5": "FUSE_REFINE"}
FLOORS = {"main_0.05pct": 0.0005, "any_pixel": 0.0, "0.5pct": 0.005}
NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma", "covered-kidney", "thread", "clamps", "suturing-needle",
         "suction-instrument", "intestine", "ultrasound-probe"]


def job(args):
    label_path, npz, keys, iou_models = args
    gt = read_mask(label_path)
    z = np.load(npz)
    gc = np.bincount(gt.ravel(), minlength=256)[:NC]
    area = gt.shape[0] * gt.shape[1]
    out = {}
    cache = {}
    for model, key in keys.items():
        if key not in cache:
            full = cv2.resize(z[key], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            cache[key] = np.bincount(full.ravel(), minlength=256)[:NC]
        pc = cache[key]
        extra = {}
        if model in iou_models:                      # official class IoUs for models that final_results.json does not hold
            full = cv2.resize(z[key], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            extra = {"cls": {int(c): float(v) for c, v in official_class_ious(full, gt, NC).items()}}
        out[model] = {"_extra": extra, **{f: dict(events=[int(c) for c in range(1, NC) if gc[c] == 0 and pc[c] > fl * area and pc[c] > 0],
                              predicted=[int(c) for c in range(1, NC) if pc[c] > fl * area and pc[c] > 0],
                              absent=[int(c) for c in range(1, NC) if gc[c] == 0]) for f, fl in FLOORS.items()}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", default="work_v2/actions")
    ap.add_argument("--choices", default="results/task5/choices_test.npz")
    ap.add_argument("--root", required=True)
    ap.add_argument("--final", default="results/task6/final_results.json")
    ap.add_argument("--out", default="results/taskF")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra", nargs="*", default=[], help="NAME=path/to/choices_test.npz : extra selector systems (e.g. M6_noA1) scored the same way")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    F = json.load(open(a.final))
    R = {"mcIoU": {}, "hallucination": {}, "definitions": __doc__}
    for m, d in F["models"].items():
        v = [c["iou"] for c in d["per_class"].values() if c["iou"] is not None]
        R["mcIoU"][m] = dict(mIoU=d["mIoU"], mcIoU=float(np.mean(v)), n_classes_scored=len(v),
                             per_class={k: dict(iou=c["iou"], n_frames=c["n_frames"]) for k, c in d["per_class"].items()})
    ch = np.load(a.choices, allow_pickle=True)
    keys_seq, ts, choice = [str(k) for k in ch["seq"]], ch["t"], ch["choice"].astype(str)
    seqs = {s.key: s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",))}
    extras = {}
    for e in a.extra:
        name, path = e.split("=", 1)
        z = np.load(path, allow_pickle=True)
        assert [str(k) for k in z["seq"]] == keys_seq and np.array_equal(z["t"], ts), f"{name}: frame order differs from {a.choices}"
        extras[name] = z["choice"].astype(str)
    jobs = []
    for i in range(len(ts)):
        s = seqs[keys_seq[i]]
        stem = os.path.splitext(os.path.basename(s.frames[int(ts[i])]))[0]
        kmap = dict(KEY_OF)
        kmap["M6"] = choice[i] if choice[i] != "A0" else "A0"
        for name, ch2 in extras.items():
            kmap[name] = ch2[i] if ch2[i] != "A0" else "A0"
        jobs.append((s.labels[int(ts[i])], os.path.join(a.actions, "test_masks", f"{s.split}_{s.name}", stem + ".npz"), kmap, set(extras)))
    with mp.Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=8)
    n = len(res)
    for name in extras:                       # official mIoU / mcIoU of the extra systems, from the same masks
        fr = [np.mean(list(r[name]["_extra"]["cls"].values())) for r in res if r[name]["_extra"]["cls"]]
        pc = {}
        for c in range(1, NC):
            v = [r[name]["_extra"]["cls"][c] for r in res if c in r[name]["_extra"]["cls"]]
            pc[NAMES[c]] = dict(iou=(float(np.mean(v)) if v else None), n_frames=len(v))
        sc = [v["iou"] for v in pc.values() if v["iou"] is not None]
        R["mcIoU"][name] = dict(mIoU=float(np.mean(fr)), mcIoU=float(np.mean(sc)), n_classes_scored=len(sc), per_class=pc,
                                changed_frames_pct=100 * float(np.mean([c != "A0" for c in extras[name]])))
    for m in ["M1", "M2", "M3", "M4", "M5", "M6"] + list(extras):
        H = {}
        for f in FLOORS:
            ev = [r[m][f]["events"] for r in res]
            pr = [r[m][f]["predicted"] for r in res]
            ab = [r[m][f]["absent"] for r in res]
            n_ev, n_pr = sum(len(e) for e in ev), sum(len(p) for p in pr)
            per = {}
            for c in range(1, NC):
                absent_frames = sum(1 for x in ab if c in x)
                inv = sum(1 for e in ev if c in e)
                per[NAMES[c]] = dict(frames_absent_from_ground_truth=absent_frames, frames_predicted_while_absent=inv,
                                     false_presence_rate=(inv / absent_frames if absent_frames else None))
            H[f] = dict(frames=n, events=n_ev, frames_with_event_pct=100 * sum(1 for e in ev if e) / n, events_per_frame=n_ev / n,
                        pooled_share_of_predicted=(n_ev / n_pr if n_pr else None), predicted_class_instances=n_pr, per_class=per)
        R["hallucination"][m] = H
    R["floor_px_at_full_resolution"] = {k: int(v * 1280 * 1024) for k, v in FLOORS.items()}
    json.dump(R, open(os.path.join(a.out, "mciou_hallucination.json"), "w"), indent=1)
    print(f"{'model':5s} {'mIoU':>7s} {'mcIoU':>7s} | frames w/ invented class  events/frame  share of predicted classes invented   (floor {R['floor_px_at_full_resolution']['main_0.05pct']} px)")
    for m in R["mcIoU"]:
        h = R["hallucination"][m]["main_0.05pct"]
        print(f"{m:5s} {R['mcIoU'][m]['mIoU']:7.4f} {R['mcIoU'][m]['mcIoU']:7.4f} | {h['frames_with_event_pct']:8.1f}%  {h['events_per_frame']:12.3f}  {100*h['pooled_share_of_predicted']:10.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
