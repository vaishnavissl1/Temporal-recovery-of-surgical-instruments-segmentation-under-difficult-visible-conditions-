#!/usr/bin/env python3
"""
STEP 0b (optional but strongly recommended) — bake a resized training cache.

The raw release is 2235 training frames of 1280x1024 PNG. Every epoch otherwise
re-decodes ~3 GB of PNG and re-runs the 24-bit colour LUT on 1.3 M pixels per
frame, which on most machines makes the *data loader*, not the GPU, the
bottleneck. This writes each frame once at the training resolution, with labels
already decoded to single-channel class ids:

    <out>/<split>/<seq>/left_frames/frame000.png    resized RGB
    <out>/<split>/<seq>/labels/frame000.png         uint8 target ids (+255 ignore)

Measured effect: ~8-12x faster epochs and ~40x less disk read, with no change
to the numbers — the same resize and the same nearest-neighbour label
interpolation the online pipeline would have applied, just done once.

The cache is task-specific (ids are TARGET ids, not canonical ids), so re-run it
if you change --task.

    python -m tools.prepare_dataset --root /data/endovis2018_cvdataset \
        --class-map runs/class_map.json --task instruments \
        --height 256 --width 320 --out data/prepared_256x320_instruments
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

from sitr.dataset import (Preprocessor, discover_sequences,  # noqa: E402
                          save_manifest)
from sitr.labels import ClassMap, IGNORE_INDEX, decode_label_image  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="instruments",
                    choices=["binary", "parts", "instruments", "full"])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--splits", nargs="+", default=["train"],
                    choices=["train", "test"])
    ap.add_argument("--no-repairs", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cm = ClassMap.from_json(args.class_map)
    from sitr.labels import task_remap
    remap, names, instrument_ids = task_remap(cm, args.task)
    rgb_lut = cm.rgb_lut() if cm.encoding == "rgb" else None
    id_lut = cm.id_lut() if cm.encoding == "id" else None
    pre = Preprocessor(args.height, args.width)

    seqs = discover_sequences(args.root, require_labels=True,
                              apply_repairs=not args.no_repairs,
                              splits=tuple(args.splits))
    print(f"preparing {len(seqs)} sequences "
          f"({sum(len(s) for s in seqs)} frames) -> {args.out}")
    print(f"task={args.task} classes={len(names)} {names}")

    t0 = time.time()
    written = skipped = 0
    for s in seqs:
        out_img = os.path.join(args.out, s.split, s.name, "left_frames")
        out_lbl = os.path.join(args.out, s.split, s.name, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        for i, (fp, lp) in enumerate(zip(s.frames, s.labels)):
            stem = os.path.splitext(os.path.basename(fp))[0] + ".png"
            dst_i, dst_l = os.path.join(out_img, stem), os.path.join(out_lbl, stem)
            if not args.overwrite and os.path.exists(dst_i) and os.path.exists(dst_l):
                skipped += 1
                continue

            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"unreadable frame {fp}")
            cv2.imwrite(dst_i, pre.resize_image(img))   # stays BGR on disk

            raw = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"unreadable label {lp}")
            if raw.ndim == 3 and raw.shape[2] >= 3:
                raw = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
            canonical = decode_label_image(raw, cm, rgb_lut=rgb_lut, id_lut=id_lut)
            tgt = np.full(canonical.shape, IGNORE_INDEX, dtype=np.uint8)
            known = canonical != IGNORE_INDEX
            tgt[known] = remap[canonical[known]]
            # Resize the DECODED ids with nearest neighbour: resizing the RGB
            # first can blend two colours into a third that maps to nothing.
            cv2.imwrite(dst_l, pre.resize_mask(tgt))
            written += 1

        print(f"  {s.key:<16} {len(s):>5} frames  "
              f"(repaired {len(s.repaired)})")

    meta = {"task": args.task, "target_names": names,
            "instrument_ids": instrument_ids,
            "height": args.height, "width": args.width,
            "class_map": os.path.abspath(args.class_map),
            "source_root": os.path.abspath(args.root),
            "prepared": True}
    with open(os.path.join(args.out, "prepared_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    save_manifest(os.path.join(args.out, "manifest.json"), seqs, extra=meta)

    print(f"\nwrote {written} frames ({skipped} already present) in "
          f"{time.time()-t0:.1f}s")
    print(f"Now train with:  --prepared {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
