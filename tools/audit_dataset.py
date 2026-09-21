#!/usr/bin/env python3
"""
STEP 0 — audit the EndoVis 2018 download and derive the class table.

Run this before anything else. It answers the question "does any preprocessing
need to happen before the architecture is implemented?" with evidence from your
files rather than an assumption:

  * which sequences exist, in which release, how many frames each
  * that __MACOSX phantom sequences were excluded
  * that the 7 corrected labels in repairs/ were matched and applied
  * every frame has a matching label, and every file actually opens
  * frame resolutions (expects 1280x1024)
  * the class table, unioned over all four labels.json files
  * whether any colour present in the label PNGs is NOT in that class table
    (an unmapped colour becomes ignore-index and silently deletes supervision)

    python -m tools.audit_dataset --root /data/endovis2018_cvdataset \
        --out runs/class_map.json --scan-colors --check-readable
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences, find_repairs, save_manifest  # noqa: E402
from sitr.labels import (OFFICIAL_NUM_CLASSES, build_class_map,  # noqa: E402
                         find_labels_json, scan_label_colors, task_remap)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="runs/class_map.json")
    ap.add_argument("--manifest", default="runs/dataset_manifest.json")
    ap.add_argument("--instrument-classes", nargs="*", default=None)
    ap.add_argument("--scan-colors", action="store_true",
                    help="sample label PNGs and compare against the class table")
    ap.add_argument("--scan-files", type=int, default=60)
    ap.add_argument("--check-readable", action="store_true",
                    help="open every frame and label header (slow, thorough)")
    ap.add_argument("--no-repairs", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on warnings too, not just on problems")
    args = ap.parse_args()

    problems = []   # corrupt data: training on this produces wrong numbers
    warnings = []   # informative mismatches: worth knowing, not fatal

    # ---------------------------------------------------------------- repairs
    repairs = {} if args.no_repairs else find_repairs(args.root)
    print(f"repairs/ corrected labels found: {len(repairs)}")
    for (seq, stem), p in sorted(repairs.items()):
        print(f"    {seq}/{stem}  <-  {os.path.relpath(p, args.root)}")
    if not args.no_repairs and len(repairs) != 7:
        warnings.append(
            f"expected 7 corrected labels in repairs/, found {len(repairs)}")

    # ------------------------------------------------------------- sequences
    train = discover_sequences(args.root, require_labels=True,
                               apply_repairs=not args.no_repairs, splits=("train",))
    try:
        test = discover_sequences(args.root, require_labels=True,
                                  apply_repairs=False, splits=("test",))
    except FileNotFoundError:
        test = []

    print(f"\nTRAIN sequences: {len(train)}")
    for s in train:
        rep = f"  repaired={len(s.repaired)}" if s.repaired else ""
        print(f"  {s.key:<16} idx={s.index:<3} frames={len(s):<5} "
              f"release={s.release}{rep}")
    print(f"  total train frames: {sum(len(s) for s in train)}")

    if test:
        print(f"\nTEST sequences: {len(test)}")
        for s in test:
            print(f"  {s.key:<16} idx={s.index:<3} frames={len(s):<5} "
                  f"release={s.release}")
        print(f"  total test frames: {sum(len(s) for s in test)}")

    idx = sorted(s.index for s in train)
    if idx != [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]:
        warnings.append(
            f"unexpected training sequence indices {idx}; the release provides "
            "1-7 and 9-16 (there is no seq_8)")
    n_repaired = sum(len(s.repaired) for s in train)
    if not args.no_repairs and n_repaired != len(repairs):
        warnings.append(
            f"{len(repairs)} corrected labels found but only {n_repaired} were "
            "matched to frames — check the repairs filename pattern")

    # ------------------------------------------------------------ class table
    paths = find_labels_json(args.root)
    print(f"\nlabels.json files: {len(paths)}")
    for p in paths:
        print(f"  {os.path.relpath(p, args.root)}")

    cm = build_class_map(args.root, instrument_names=args.instrument_classes)
    print(f"\nclass table (union): encoding={cm.encoding}  classes={cm.num_classes}")
    for c in cm.classes:
        flag = "INSTRUMENT" if c.instrument else "anatomy/bg "
        print(f"  id={c.id:<3} {c.name:<24} rgb={str(c.color):<18} {flag} "
              f"in {len(c.sources)} release(s)")
    if cm.num_classes != OFFICIAL_NUM_CLASSES:
        warnings.append(
            f"class table has {cm.num_classes} classes but the challenge's own "
            f"run.py declares NUM_CLASSES = {OFFICIAL_NUM_CLASSES}")

    print("\ntask presets:")
    for task in ("binary", "parts", "instruments", "full"):
        try:
            _lut, names, inst = task_remap(cm, task)
            print(f"  {task:<12} {len(names):>2} classes  instrument ids={inst}")
            print(f"               {names}")
        except ValueError as e:
            print(f"  {task:<12} unavailable: {e}")

    # ---------------------------------------------------------- colour audit
    if args.scan_colors:
        label_paths = [p for s in train for p in s.labels if p]
        step = max(1, len(label_paths) // max(1, args.scan_files))
        sample = label_paths[::step][:args.scan_files]
        info = scan_label_colors(sample, limit=len(sample))
        known = {tuple(c.color) for c in cm.classes if c.color}
        print(f"\ncolour audit over {info['files_scanned']} label files "
              f"({'RGB' if info['multichannel'] else 'single-channel'}):")
        total = sum(n for _, n in info["rgb_values"]) or 1
        unmapped = 0
        for rgb, n in info["rgb_values"]:
            hit = rgb in known
            name = next((c.name for c in cm.classes if c.color == rgb), "UNMAPPED")
            if not hit:
                unmapped += n
            print(f"  {str(rgb):<18} {100.0*n/total:6.3f}%  {name}")
        if unmapped:
            pct = 100.0 * unmapped / total
            problems.append(
                f"{pct:.3f}% of sampled label pixels have a colour absent from "
                "the class table; those pixels become ignore-index and their "
                "supervision is lost")

    # --------------------------------------------------------- readability
    if args.check_readable:
        print("\nchecking every file opens and recording resolutions ...")
        sizes = Counter()
        bad = []
        for s in train + test:
            for p in s.frames:
                im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_8)
                if im is None:
                    bad.append(p)
                else:
                    sizes[(im.shape[0] * 8, im.shape[1] * 8)] += 1
            for p in s.labels:
                if p and cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_8) is None:
                    bad.append(p)
        print(f"  approximate frame resolutions: "
              f"{ {f'{h}x{w}': n for (h, w), n in sizes.most_common(5)} }")
        if bad:
            problems.append(f"{len(bad)} unreadable files, first: {bad[0]}")

    # ------------------------------------------------------------------ out
    cm.to_json(args.out)
    save_manifest(args.manifest, train + test,
                  extra={"class_map": os.path.abspath(args.out),
                         "repairs_applied": len(repairs),
                         "root": os.path.abspath(args.root)})
    print(f"\nwrote {args.out}")
    print(f"wrote {args.manifest}")

    if warnings:
        print("\n" + "-" * 72)
        print("WARNINGS — not fatal, but know what they mean for your results:")
        for w in warnings:
            print(f"  ! {w}")
        print("-" * 72)

    if problems:
        print("\n" + "!" * 72)
        print("PROBLEMS — training on this data would produce wrong numbers:")
        for p in problems:
            print(f"  - {p}")
        print("!" * 72)
        return 1

    if args.strict and warnings:
        print("\n--strict: treating the warnings above as fatal.")
        return 1

    print("\nAudit passed. No separate preprocessing pass is required: colour "
          "decoding, the repairs override and resizing all happen inside the "
          "data pipeline at load time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
