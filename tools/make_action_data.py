#!/usr/bin/env python3
"""
Build the per-frame ACTION DATA for Tasks 5 and 6.

    python -m tools.make_action_data --release work_v2/release --work work_v2 \
        --root <endovis2018_cvdataset> --stereo work_v2/stereo --right-view work_v2/right_view \
        --groups oof test --out work_v2/actions

For every frame of every HELD-OUT sequence (training sequences via their fold
model, so the data is honest; test sequences via the final model) it computes
  A0 keep, A1 stereo, A2 temporal, A3 refine, and the two baselines FUSE (M4,
  stereo+temporal on every frame) and FUSE_REFINE (M5),
the observable features the selector may use, and -- the only use of ground
truth, after the masks exist -- the measured IoU change of every action.

Design choices stated, not hidden:
  * Temporal memory holds the BASE (uncorrected) predictions, so every frame's
    data is independent of any policy. A deployed system would store corrected
    masks; the difference is a limitation, noted in FINDINGS.
  * Everything is at the working resolution (512x640). Full-resolution official
    numbers for the test frames are produced in Task 6 from the saved masks.
  * Per-action wall-clock time of the correction step is recorded (CPU seconds
    per frame) for the compute report.
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

from sitr.actions import (ACTIONS, BASELINES, ActionContext, ActionParams,  # noqa: E402
                          action_features, delta_scores)
from sitr.align import build_align  # noqa: E402
from sitr.dataset import discover_sequences  # noqa: E402
from sitr.difficulty import stereo_cues  # noqa: E402
from sitr.recovery import TemporalRecovery  # noqa: E402
from sitr.runtime import ReleaseBundle  # noqa: E402
from sitr.stereo import (VerticalCorrection, calibration_path, get_rectifier,  # noqa: E402
                         instrument_stereo_indicators, right_to_left_grid)
from sitr.utils import keep_awake  # noqa: E402
from tools.measure_recovery import fold_bundles  # noqa: E402

DELTA_KEYS = ("A1", "A2", "A3", "FUSE", "FUSE_REFINE")
FEAT_KEYS = ("A1", "A2", "A3", "FUSE")
CUE_RAW_NAMES = ("stereo_lr_fail", "occluded", "specular", "lap_var", "brightness", "contrast",
                 "dark_channel", "under_frac", "over_frac")


def run_group(bundle, seqs, orig, a, group_name, save_masks):
    inst, nc = bundle.instrument_ids, bundle.num_classes
    names = bundle.target_names
    params = ActionParams()
    rec = TemporalRecovery(aligner=build_align("identity"), tau=bundle.tau, gamma=a.gamma,
                           num_classes=nc, instrument_ids=inst)
    for seq in seqs:
        out_path = os.path.join(a.out, f"{group_name}__{seq.split}_{seq.name}.npz")
        if os.path.exists(out_path):
            print(f"  {seq.key}: exists, skipping", flush=True); continue
        sd = os.path.join(a.stereo, f"{seq.split}_{seq.name}")
        rd = os.path.join(a.right_view, f"{seq.split}_{seq.name}")
        meta = json.load(open(os.path.join(sd, "stereo_meta.json")))
        assert len(meta["frames"]) == len(seq), (seq.key, len(meta["frames"]), len(seq))
        o = orig[seq.key]
        dist, rot = meta["convention"]
        rect = get_rectifier(calibration_path(o.frame_dir), meta["alpha"], dist, rot)
        corr = VerticalCorrection(**{k: meta["correction"][k] for k in
                                     ("a", "b", "c", "n_matches", "mad", "max_abs_px", "applied")}) \
            if meta.get("correction") else None
        rows = {k: [] for k in ("t", "ind9", "st3", "r", "reliable", "cold", "degenerate", "cue_raw",
                                "instr_before", "off_before")}
        rows.update({f"d_{k}": [] for k in DELTA_KEYS}); rows.update({f"o_{k}": [] for k in DELTA_KEYS})
        rows.update({f"f_{k}": [] for k in FEAT_KEYS}); rows.update({f"avail_{k}": [] for k in DELTA_KEYS})
        rows.update({f"sec_{k}": [] for k in DELTA_KEYS})
        t0 = time.time()
        for state in bundle.stream(seq, with_gt=True):
            t = state.t
            fn = meta["frames"][t]; stem = os.path.splitext(fn)[0]
            z = np.load(os.path.join(sd, stem + ".npz"))
            r = np.load(os.path.join(rd, stem + ".npz"))
            H, W = state.mask.shape
            evid = dict(lr_ok=z["lr_ok"].astype(bool), occluded=z["occluded"].astype(bool),
                        confidence=z["confidence"], in_view=z["in_view"].astype(bool))
            disp = z["disparity"].astype(np.float32)
            rl = right_to_left_grid(r["label"], disp, rect, corr, (H, W), cv2.INTER_NEAREST, 255)
            rc = right_to_left_grid(r["conf"], disp, rect, corr, (H, W), cv2.INTER_NEAREST, 0)
            ctx = ActionContext(state=state, evidence=evid, right_label=rl, right_conf=rc,
                                target_names=names, instrument_ids=inst, recovery=rec)
            base = state.mask
            gt = state.ground_truth                     # read only below, to LABEL the actions
            masks, secs = {}, {}
            for k in ("A1", "A2", "A3"):
                t1 = time.perf_counter(); masks[k] = ACTIONS[k](ctx, params); secs[k] = time.perf_counter() - t1
            for k, fnc in BASELINES.items():
                t1 = time.perf_counter(); masks[k] = fnc(ctx, params); secs[k] = time.perf_counter() - t1
            d, o_, av = {}, {}, {}
            for k in DELTA_KEYS:
                m = masks[k]
                av[k] = m is not None
                if m is None:
                    d[k] = o_[k] = np.nan
                else:
                    s = delta_scores(base, m, gt, nc, inst)
                    d[k] = s["instr"]; o_[k] = np.nan if s["official"] is None else s["official"]
            b = delta_scores(base, base, gt, nc, inst)
            rows["t"].append(t); rows["r"].append(state.reliability)
            rows["reliable"].append(state.reliable); rows["cold"].append(state.cold_start)
            rows["degenerate"].append(state.degenerate)
            rows["ind9"].append(np.asarray(state.indicators, np.float32))
            rows["st3"].append(np.asarray(instrument_stereo_indicators(
                base, evid["lr_ok"], evid["occluded"], evid["confidence"], evid["in_view"], inst), np.float32))
            sc = stereo_cues(evid["lr_ok"], evid["occluded"], evid["specular"] if "specular" in evid else z["specular"].astype(bool),
                             evid["in_view"])
            rows["cue_raw"].append(np.array([sc["stereo_lr_fail"], sc["occluded"], sc["specular"], *z["image_cues"]], np.float32))
            from sitr.metrics import official_frame_iou
            rows["instr_before"].append(b["instr_before"])
            ob = official_frame_iou(base, gt, nc)
            rows["off_before"].append(np.nan if ob is None else ob)
            for k in DELTA_KEYS:
                rows[f"d_{k}"].append(d[k]); rows[f"o_{k}"].append(o_[k]); rows[f"avail_{k}"].append(av[k])
                rows[f"sec_{k}"].append(secs[k])
            for k in FEAT_KEYS:
                rows[f"f_{k}"].append(action_features(ctx, masks[k]) if masks[k] is not None
                                      else np.full(8, np.nan, np.float32))
            if save_masks:
                md = os.path.join(a.out, "test_masks", f"{seq.split}_{seq.name}")
                os.makedirs(md, exist_ok=True)
                pk = {"A0": base}
                for k in ("A1", "A2", "A3", "FUSE", "FUSE_REFINE"):
                    pk[k] = masks[k] if masks[k] is not None else base
                np.savez_compressed(os.path.join(md, stem + ".npz"), **pk,
                                    avail=np.array([masks["A1"] is not None, masks["A2"] is not None]))
        arr = {k: np.array(v) for k, v in rows.items()}
        np.savez_compressed(out_path, seq=np.array(seq.key), **arr)
        n = len(arr["t"])
        print(f"  {seq.key:<12} {n} frames  {time.time()-t0:.0f}s  "
              f"mean instrument IoU {np.nanmean(arr['instr_before']):.3f}  "
              f"mean delta A1 {np.nanmean(arr['d_A1']):+.4f} A2 {np.nanmean(arr['d_A2']):+.4f} "
              f"A3 {np.nanmean(arr['d_A3']):+.4f} FUSE {np.nanmean(arr['d_FUSE']):+.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--stereo", required=True)
    ap.add_argument("--right-view", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="+", default=["oof", "test"], choices=["oof", "test"])
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--limit-seqs", type=int, default=0)
    a = ap.parse_args()
    keep_awake()
    os.makedirs(a.out, exist_ok=True)
    orig = {s.key: s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True,
                                                 splits=("train", "test"))}
    if "oof" in a.groups:
        print("OUT-OF-FOLD training sequences (per-fold models)")
        for i, b, seqs in fold_bundles(a.release, a.work):
            print(f" fold {i}: {[s.key for s in seqs]}")
            run_group(b, seqs[:a.limit_seqs] if a.limit_seqs else seqs, orig, a, "oof", save_masks=False)
    if "test" in a.groups:
        print("OFFICIAL TEST sequences (final model)")
        b = ReleaseBundle.load(a.release)
        seqs = b.sequences("test")
        run_group(b, seqs[:a.limit_seqs] if a.limit_seqs else seqs, orig, a, "test", save_masks=True)
    print("done ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
