#!/usr/bin/env python3
"""
Cache the stereo evidence for every frame, ONCE, to disk (CPU only, no GPU).

    python -m tools.cache_stereo --root <endovis2018_cvdataset> --out work_v2/stereo \
        --sgbm-json results/task2/stereo_check.json --splits train

Per frame (one .npz): disparity (float16), lr_ok, occluded, specular, in_view
(bool), confidence (uint8) -- all on the ORIGINAL-LEFT working grid (default
512x640, the same grid as the segmentation masks), so evidence and masks line up
pixel for pixel. Per sequence: stereo_meta.json with the label-free vertical
self-calibration (estimated from frames 0,5,...,35 only) and the SGBM settings.

Why 512x640 and not full resolution: the three frame-level stereo indicators are
fractions/means over the predicted-instrument region, so a 2x downsample does not
change them; full resolution would be ~4x the disk for no decision it affects.
(Disparity VALUES stay in full-resolution rectified pixels.)

Resumable: frames whose file exists are skipped. Run a second copy with
`--splits test` in parallel; they never touch the same files.
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

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.stereo import (SGBMParams, calibration_path, compute_evidence,  # noqa: E402
                         estimate_vertical_correction, get_rectifier, right_path)
from sitr.utils import keep_awake  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sgbm-json", required=True,
                    help="results/task2/stereo_check.json: the SGBM settings and the "
                         "distortion/rotation convention chosen there")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--refine-frames", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 35])
    ap.add_argument("--no-self-calibration", action="store_true",
                    help="calibration only (for the comparison in Task 4)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: frames per sequence")
    a = ap.parse_args()
    cv2.setNumThreads(a.threads)
    keep_awake()

    with open(a.sgbm_json) as f:
        t2 = json.load(f)
    params = SGBMParams(**t2["sgbm_chosen"])
    dist, rot = t2["chosen_convention"].split("/")
    print(f"SGBM {params}\nconvention distortion={dist} rotation={rot}")

    seqs = discover_sequences(a.root, require_labels=True, apply_repairs=True,
                              splits=tuple(a.splits))
    seqs.sort(key=lambda s: (s.split != "train", s.index))
    out_hw = (a.height, a.width)
    total = sum(len(s) for s in seqs)
    done = 0
    t0 = time.time()
    for s in seqs:
        sd = os.path.join(a.out, f"{s.split}_{s.name}")
        os.makedirs(sd, exist_ok=True)
        rc = get_rectifier(calibration_path(s.frame_dir), a.alpha, dist, rot)
        meta_path = os.path.join(sd, "stereo_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            from sitr.stereo import VerticalCorrection
            corr = VerticalCorrection(**{k: meta["correction"][k] for k in
                                         ("a", "b", "c", "n_matches", "mad", "max_abs_px", "applied")})
        else:
            if a.no_self_calibration:
                corr = None
            else:
                corr = estimate_vertical_correction(
                    rc, [s.frames[i] for i in a.refine_frames if i < len(s)])
            meta = dict(seq=s.key, n_frames=len(s), out_hw=list(out_hw), alpha=a.alpha,
                        convention=[dist, rot], sgbm=params.__dict__,
                        correction=(corr.to_dict() if corr is not None else None),
                        self_calibration=not a.no_self_calibration,
                        rectified_focal=rc.focal, baseline_mm=rc.baseline_mm,
                        calibration_file=os.path.abspath(calibration_path(s.frame_dir)),
                        frames=[os.path.basename(p) for p in s.frames])
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=1)
            if corr is not None:
                print(f"{s.key}: self-calibration N={corr.n_matches} a={corr.a:.2f} b={corr.b:.2f} "
                      f"c={corr.c:.2f} max|dy|={corr.max_abs_px:.1f}px "
                      f"{'APPLIED' if corr.applied else 'identity'}", flush=True)
        n = len(s) if not a.limit else min(a.limit, len(s))
        for i in range(n):
            stem = os.path.splitext(os.path.basename(s.frames[i]))[0]
            path = os.path.join(sd, stem + ".npz")
            done += 1
            if os.path.exists(path):
                continue
            L = cv2.imread(s.frames[i], cv2.IMREAD_COLOR)
            R = cv2.imread(right_path(s.frames[i]), cv2.IMREAD_COLOR)
            ev = compute_evidence(L, R, rc, corr, params, out_hw)
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, disparity=ev.disparity, lr_ok=ev.lr_ok, occluded=ev.occluded,
                                specular=ev.specular, confidence=ev.confidence, in_view=ev.in_view,
                                image_cues=ev.image_cues)
            os.replace(tmp, path)                       # atomic: a crash never leaves a half file
            if done % 25 == 0:
                el = time.time() - t0
                print(f"  {done}/{total} frames  {el/60:.1f} min  ETA {(el/done)*(total-done)/60:.1f} min", flush=True)
    print(f"finished {done} frames in {(time.time()-t0)/60:.1f} min -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
