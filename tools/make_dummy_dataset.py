#!/usr/bin/env python3
"""
Generate a miniature dataset that replicates the REAL EndoVis 2018 download
layout, including the two traps that break naive loaders:

  * `__MACOSX/` shadow trees with AppleDouble `._frameXXX.png` junk files
  * `repairs/` corrected labels that must override release-1 ground truth

plus the two labels.json schemas (RGBA + "classes" wrapper in test_data, bare
RGB list in the releases), the missing seq_8, and the train/test sequence-name
collision.

This is a PLUMBING TEST, not science. It exists so you can verify the install
and the loader in under a minute before committing GPU hours to the real 2235
frames. Never report numbers from it.

    python -m tools.make_dummy_dataset --out /tmp/fake2018 --frames 30
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

# Colours taken from the real labels.json shown in the dataset documentation.
CLASSES = [
    ("background-tissue", [0, 0, 0], 0),
    ("instrument-shaft", [0, 255, 0], 1),
    ("instrument-clasper", [0, 255, 255], 2),
    ("instrument-wrist", [125, 255, 12], 3),
    ("kidney-parenchyma", [255, 55, 0], 4),
]
RELEASES = {
    "miccai_challenge_2018_release_1": [1, 2, 3, 4],
    "miccai_challenge_release_2": [5, 6, 7],
    "miccai_challenge_release_3": [9, 10, 11, 12],      # note: no seq_8
    "miccai_challenge_release_4": [13, 14, 15, 16],
}
REPAIRS = [("seq_1", 42), ("seq_1", 43), ("seq_1", 44), ("seq_1", 73),
           ("seq_4", 135), ("seq_4", 137), ("seq_4", 138)]


def draw(h, w, t, rng, phase):
    label = np.zeros((h, w, 3), np.uint8)
    img = np.full((h, w, 3), (150, 90, 90), np.uint8)
    img = np.clip(img.astype(np.float32) +
                  rng.normal(0, 9, (h, w, 1)), 0, 255).astype(np.uint8)

    cv2.ellipse(img, (int(w * .7), int(h * .72)), (int(w * .25), int(h * .2)),
                20, 0, 360, (170, 110, 110), -1)
    cv2.ellipse(label, (int(w * .7), int(h * .72)), (int(w * .25), int(h * .2)),
                20, 0, 360, tuple(CLASSES[4][1]), -1)

    ang = phase + 0.12 * t
    cx = int(w * (0.25 + 0.4 * (0.5 + 0.5 * np.sin(ang))))
    cy = int(h * (0.30 + 0.25 * (0.5 + 0.5 * np.cos(ang * 0.8))))
    hard, mode = (t % 11) in (5, 6, 7), (t // 11) % 4
    scale = 0.45 if (hard and mode == 3) else 1.0

    L, th = int(w * .30 * scale), max(2, int(h * .055 * scale))
    p0, p1, p2 = ((cx - L, cy - int(L * .4)), (cx, cy),
                  (cx + int(L * .35), cy + int(L * .25)))
    for canvas, cols in ((img, [(210, 210, 215), (190, 190, 200), (230, 230, 235)]),
                         (label, [tuple(CLASSES[1][1]), tuple(CLASSES[3][1]),
                                  tuple(CLASSES[2][1])])):
        cv2.line(canvas, p0, p1, cols[0], th)
        cv2.circle(canvas, p1, max(2, th // 2 + 1), cols[1], -1)
        cv2.line(canvas, p1, p2, cols[2], max(2, th - 2))

    if hard:
        if mode == 0:
            cv2.circle(img, p1, int(L * .55), (140, 80, 80), -1)
            cv2.circle(label, p1, int(L * .55), tuple(CLASSES[0][1]), -1)
        elif mode == 1:
            k = max(3, int(w * .035) | 1)
            kern = np.zeros((k, k), np.float32)
            kern[k // 2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kern)
        elif mode == 2:
            ov = img.copy()
            cv2.circle(ov, p1, int(L * .7), (255, 255, 255), -1)
            img = cv2.addWeighted(ov, .65, img, .35, 0)
    return img, label


def write_seq(seq_dir, frames, h, w, rng, phase, calib=True):
    fdir, ldir, rdir = (os.path.join(seq_dir, d)
                        for d in ("left_frames", "labels", "right_frames"))
    for d in (fdir, ldir, rdir):
        os.makedirs(d, exist_ok=True)
    if calib:
        with open(os.path.join(seq_dir, "camera_calibration.txt"), "w") as f:
            f.write("# synthetic placeholder\n")
    for t in range(frames):
        img, lab = draw(h, w, t, rng, phase)
        name = f"frame{t:03d}.png"
        cv2.imwrite(os.path.join(fdir, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(ldir, name), cv2.cvtColor(lab, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(rdir, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def write_macosx_junk(release_dir, release_name, seq_names, frames):
    """AppleDouble shadow tree — the trap that creates phantom sequences."""
    base = os.path.join(release_dir, "__MACOSX", release_name)
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "._labels.json"), "wb") as f:
        f.write(b"\x00\x05\x16\x07Mac OS X        \x00\x02\x00\x00")
    for s in seq_names:
        for sub in ("left_frames", "labels", "right_frames"):
            d = os.path.join(base, s, sub)
            os.makedirs(d, exist_ok=True)
            for t in range(frames):
                with open(os.path.join(d, f"._frame{t:03d}.png"), "wb") as f:
                    f.write(b"\x00\x05\x16\x07Mac OS X        \x00\x02\x00\x00")
        with open(os.path.join(base, f"._{s}"), "wb") as f:
            f.write(b"\x00\x05\x16\x07Mac OS X ATTR com.apple.quarantine")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=30, help="per training seq")
    ap.add_argument("--test-frames", type=int, default=20)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    h, w = args.height, args.width

    # ------------------------------------------------- training releases
    for release, seq_ids in RELEASES.items():
        outer = os.path.join(args.out, release)
        inner = os.path.join(outer, release)
        os.makedirs(inner, exist_ok=True)
        # bare-list RGB schema, as the releases ship it
        with open(os.path.join(inner, "labels.json"), "w") as f:
            json.dump([{"name": n, "color": c, "classid": i}
                       for n, c, i in CLASSES], f, indent=2)
        names = []
        for sid in seq_ids:
            name = f"seq_{sid}"
            names.append(name)
            write_seq(os.path.join(inner, name), args.frames, h, w, rng,
                      float(rng.uniform(0, 6.28)))
        if release == "miccai_challenge_2018_release_1":
            write_macosx_junk(outer, release, names, args.frames)

    # ------------------------------------------------------------ repairs
    rep = os.path.join(args.out, "repairs", "repairs")
    os.makedirs(rep, exist_ok=True)
    for seq, fr in REPAIRS:
        _img, lab = draw(h, w, fr, rng, 1.0)
        cv2.imwrite(os.path.join(rep, f"{seq}_frame{fr:03d}.png"),
                    cv2.cvtColor(lab, cv2.COLOR_RGB2BGR))

    # ---------------------------------------------------------- test_data
    td = os.path.join(args.out, "test_data", "test_data")
    os.makedirs(td, exist_ok=True)
    # RGBA + "classes" wrapper + "active", as test_data ships it
    with open(os.path.join(td, "labels.json"), "w") as f:
        json.dump({"classes": [{"name": n, "active": True,
                                "color": c + [128], "classid": i}
                               for n, c, i in CLASSES]}, f, indent=2)
    for sid in (1, 2, 3, 4):
        outer = os.path.join(td, f"seq_{sid}-20260917T0335{sid:02d}Z-1-001")
        write_seq(os.path.join(outer, f"seq_{sid}"), args.test_frames, h, w,
                  rng, float(rng.uniform(0, 6.28)))

    n_train = sum(len(v) for v in RELEASES.values()) * args.frames
    print(f"Wrote replica dataset to {args.out}")
    print(f"  training: {sum(len(v) for v in RELEASES.values())} sequences "
          f"(no seq_8), {n_train} frames")
    print(f"  test:     4 sequences x {args.test_frames} frames")
    print(f"  traps:    __MACOSX shadow tree, {len(REPAIRS)} repairs, "
          f"2 labels.json schemas, train/test seq name collision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
