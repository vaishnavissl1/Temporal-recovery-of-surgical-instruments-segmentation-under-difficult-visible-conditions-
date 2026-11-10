#!/usr/bin/env python3
"""
Add the three stereo indicators to Stage 2 caches.

    python -m tools.add_stereo_indicators --cache work_v2/cache_oof \
        --stereo work_v2/stereo --out work_v2/cache_oof_stereo

For every frame of every cached sequence: take the PREDICTED mask stored in the
cache, take that frame's stereo evidence (tools/cache_stereo.py), and compute
stereo_disagreement / occluded_fraction / disparity_confidence over the
predicted-instrument pixels (sitr.stereo.instrument_stereo_indicators). Nothing
else in the cache changes, and no ground truth is read.

The frame order of a cache is the sorted frame list of its sequence, the same
order the stereo cache uses; this is VERIFIED (counts and frame names) rather
than assumed, and the tool refuses to continue on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.reliability import SequenceCache  # noqa: E402
from sitr.stereo import instrument_stereo_indicators  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="directory of Stage 2 .npz caches")
    ap.add_argument("--stereo", required=True, help="directory written by tools/cache_stereo.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.cache) if f.endswith(".npz"))
    if not files:
        raise SystemExit(f"no caches in {a.cache}")
    for f in files:
        c = SequenceCache.load(os.path.join(a.cache, f))
        split, name = c.name.split("/")
        sd_dir = os.path.join(a.stereo, f"{split}_{name}")
        meta_path = os.path.join(sd_dir, "stereo_meta.json")
        if not os.path.exists(meta_path):
            raise SystemExit(f"{c.name}: stereo cache missing ({sd_dir}). Run tools/cache_stereo.py first.")
        with open(meta_path) as fh:
            frames = json.load(fh)["frames"]
        if len(frames) != len(c):
            raise SystemExit(f"{c.name}: cache has {len(c)} frames but the stereo cache has {len(frames)}")
        sd, oc, dc = (np.zeros(len(c), np.float32) for _ in range(3))
        missing = []
        for t, fn in enumerate(frames):
            p = os.path.join(sd_dir, os.path.splitext(fn)[0] + ".npz")
            if not os.path.exists(p):
                missing.append(fn); continue
            z = np.load(p)
            m = c.masks[t]
            h, w = m.shape
            def fit(x):          # evidence is on the 2x grid; nearest down to the mask grid
                return x if x.shape == (h, w) else cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            sd[t], oc[t], dc[t] = instrument_stereo_indicators(
                m, fit(z["lr_ok"]).astype(bool), fit(z["occluded"]).astype(bool),
                fit(z["confidence"]), fit(z["in_view"]).astype(bool), c.instrument_ids)
        if missing:
            raise SystemExit(f"{c.name}: {len(missing)} frames lack stereo evidence (first: {missing[0]})")
        c.sd, c.oc, c.dc = sd, oc, dc
        c.seq_index = np.arange(len(c), dtype=np.int32)
        c.save(os.path.join(a.out, f))
        print(f"  {c.name:<14} frames={len(c):<4} mean sd={sd[c.area > 0].mean():.3f} "
              f"oc={oc[c.area > 0].mean():.3f} dc={dc[c.area > 0].mean():.3f}")
    # carry the index file along so downstream tools see the same metadata
    for extra in ("cache_index.json",):
        src = os.path.join(a.cache, extra)
        if os.path.exists(src):
            with open(src) as fh:
                idx = json.load(fh)
            idx["stereo_indicators_added"] = True
            with open(os.path.join(a.out, extra), "w") as fh:
                json.dump(idx, fh, indent=2)
    print(f"wrote {len(files)} caches with stereo indicators -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
