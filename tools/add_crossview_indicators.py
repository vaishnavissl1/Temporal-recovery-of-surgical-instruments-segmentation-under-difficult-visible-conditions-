#!/usr/bin/env python3
"""
Add the six cross-view indicators to Stage 2 caches (Task D).

    python -m tools.add_crossview_indicators --cache work_v2/cache_oof  --stereo work_v2/stereo \
        --right-view work_v2/right_view --out work_v2/cache_oof_xv  --report results/taskD/crossview_oof.json
    python -m tools.add_crossview_indicators --cache work_v2/cache_test --stereo work_v2/stereo \
        --right-view work_v2/right_view --out work_v2/cache_test_xv --report results/taskD/crossview_test.json

For every frame of every cached sequence: take the LEFT predicted mask stored in the cache and the RIGHT-view
prediction already computed by tools/cache_right_view.py (same segmenter, out-of-fold honest for training sequences),
bring both onto the rectified grid with nearest-neighbour warps, drop the field-of-view strips, and compute the six
indicators of sitr.crossview. No ground truth is read; no disparity map is used. Nothing else in the cache changes.

The right-view labels are stored on the 512x640 grid and the cached left masks on the 2x coarser 256x320 grid; the right
labels are brought to the left grid by nearest neighbour (the same operation that produced the cached left masks).

Frame order is VERIFIED, not assumed, exactly as in tools/add_stereo_indicators.py: the cache, the stereo metadata and the
right-view folder must agree on the frame count and on every file name, and the tool refuses to continue on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr import crossview as XV  # noqa: E402
from sitr.reliability import SequenceCache  # noqa: E402
from sitr.stereo import VerticalCorrection, get_rectifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="directory of Stage 2 .npz caches")
    ap.add_argument("--stereo", required=True, help="directory written by tools/cache_stereo.py (calibration + vertical correction + frame list)")
    ap.add_argument("--right-view", required=True, help="directory written by tools/cache_right_view.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None, help="write per-sequence summary JSON here")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.cache) if f.endswith(".npz"))
    if not files:
        raise SystemExit(f"no caches in {a.cache}")
    report = {"guard_full_px": XV.DISPARITY_P99_FULL_PX, "empty_both_constants": dict(zip(XV.NAMES, XV.EMPTY_BOTH)),
              "indicator_names": list(XV.NAMES), "sequences": {}}
    for f in files:
        c = SequenceCache.load(os.path.join(a.cache, f))
        split, name = c.name.split("/")
        sd_dir = os.path.join(a.stereo, f"{split}_{name}")
        rv_dir = os.path.join(a.right_view, f"{split}_{name}")
        meta_path = os.path.join(sd_dir, "stereo_meta.json")
        if not os.path.exists(meta_path):
            raise SystemExit(f"{c.name}: stereo metadata missing ({sd_dir}). Run tools/cache_stereo.py first.")
        if not os.path.isdir(rv_dir):
            raise SystemExit(f"{c.name}: right-view predictions missing ({rv_dir}). Run tools/cache_right_view.py first.")
        meta = json.load(open(meta_path))
        frames = meta["frames"]
        if len(frames) != len(c):
            raise SystemExit(f"{c.name}: cache has {len(c)} frames but the stereo metadata lists {len(frames)}")
        have = sorted(os.path.splitext(x)[0] for x in os.listdir(rv_dir) if x.endswith(".npz"))
        want = sorted(os.path.splitext(x)[0] for x in frames)
        if have != want:
            miss = sorted(set(want) - set(have))[:3]
            extra = sorted(set(have) - set(want))[:3]
            raise SystemExit(f"{c.name}: right-view files do not match the frame list "
                             f"({len(have)} files vs {len(want)} frames; missing {miss}, unexpected {extra})")
        dist, rot = meta["convention"]
        rect = get_rectifier(meta["calibration_file"], meta["alpha"], dist, rot)
        corr = None
        if meta.get("correction"):
            corr = VerticalCorrection(**{k: meta["correction"][k] for k in
                                         ("a", "b", "c", "n_matches", "mad", "max_abs_px", "applied")})
        h, w = c.masks.shape[1:]
        xv = np.zeros((len(c), len(XV.NAMES)), np.float32)
        surv_l, surv_r, n_empty = [], [], 0
        for t, fn in enumerate(frames):
            z = np.load(os.path.join(rv_dir, os.path.splitext(fn)[0] + ".npz"))
            right = z["label"]
            if right.shape != (h, w):
                right = cv2.resize(right, (w, h), interpolation=cv2.INTER_NEAREST)
            vals, info = XV.crossview_indicators(c.masks[t], right, rect, corr, c.instrument_ids, with_info=True)
            if not np.all(np.isfinite(vals)):
                raise SystemExit(f"{c.name} frame {t}: non-finite indicator {vals}")
            xv[t] = vals
            surv_l.append(info["surviving_left"]); surv_r.append(info["surviving_right"])
            n_empty += int(tuple(vals) == XV.EMPTY_BOTH)
        c.xv = xv
        if c.seq_index is None:
            c.seq_index = np.arange(len(c), dtype=np.int32)
        c.save(os.path.join(a.out, f))
        report["sequences"][c.name] = dict(
            frames=len(c), grid=[int(h), int(w)], guard_px=info["guard_px"],
            surviving_fraction_left=float(np.mean(surv_l)), surviving_fraction_right=float(np.mean(surv_r)),
            frames_empty_in_both_views=n_empty, vertical_correction_applied=bool(corr is not None and corr.applied),
            mean={n: float(xv[:, j].mean()) for j, n in enumerate(XV.NAMES)})
        print(f"  {c.name:<14} frames={len(c):<4} guard={info['guard_px']}px surviving L/R="
              f"{np.mean(surv_l):.3f}/{np.mean(surv_r):.3f} empty-both={n_empty:<3} "
              + " ".join(f"{n.replace('xv_', '')}={xv[:, j].mean():.3f}" for j, n in enumerate(XV.NAMES)), flush=True)
    for extra in ("cache_index.json",):
        src = os.path.join(a.cache, extra)
        if os.path.exists(src):
            idx = json.load(open(src))
            idx["crossview_indicators_added"] = True
            json.dump(idx, open(os.path.join(a.out, extra), "w"), indent=2)
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        json.dump(report, open(a.report, "w"), indent=1)
    print(f"wrote {len(files)} caches with cross-view indicators -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
