#!/usr/bin/env python3
"""
Write demo_subset/predicted_masks/: the system's final masks for the 24 demo frames, so the website's public
"Score masks" tool can be tried with real predictions.

    python -m tools.make_demo_predicted_masks

Each file is a grey PNG, 1280x1024, pixel value = class number (0 to 11), named like the ground-truth label
(frame171.png ...). The masks are the ones the paper's final evaluation already wrote (work_v2/actions/test_masks),
the output of the selector's choice for each frame (results/taskF/no_a1/choices_test.npz), upsampled to the label size
by nearest neighbour exactly as tools/final_eval.py does. Nothing is re-run.
"""
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "demo_subset", "predicted_masks")


def main():
    os.makedirs(OUT, exist_ok=True)
    ch = np.load(os.path.join(ROOT, "results", "taskF", "no_a1", "choices_test.npz"), allow_pickle=True)
    n = 0
    for t in range(171, 195):
        i = int(np.where((ch["seq"] == "test/seq_1") & (ch["t"] == t))[0][0])
        z = np.load(os.path.join(ROOT, "work_v2", "actions", "test_masks", "test_seq_1", f"frame{t:03d}.npz"))
        m = z[str(ch["choice"][i])]
        full = cv2.resize(m, (1280, 1024), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT, f"frame{t}.png"), full)
        n += 1
    print(f"wrote {n} masks to {OUT}")


if __name__ == "__main__":
    main()
