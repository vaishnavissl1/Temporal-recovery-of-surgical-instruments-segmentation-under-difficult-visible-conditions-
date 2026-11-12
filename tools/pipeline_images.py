#!/usr/bin/env python3
"""
Saves one image per pipeline step, for selected frames, into results/pipeline/.

Nothing is re-trained or re-tuned and no number is produced here: every picture
is drawn from files the final evaluation already wrote (prepared frames and
labels, work_v2/actions/test_masks, work_v2/stereo, results/task5/choices_test.npz).

  python -m tools.pipeline_images                 # writes results/pipeline/

Steps drawn for each frame (the order of the architecture):
  01_input            left camera frame (as given to the model, 512x640)
  02_ground_truth     the official label, drawn for comparison only
  03_stereo_disparity depth cue from the stereo pair (the second camera)
  04_segmentation     mask from the segmenter alone (M1 / A0)
  05_error_before     where that mask disagrees with the ground truth
  06_reliability      the gate's score r and its decision (flag / accept)
  07_corrected        the mask after the action the selector chose (or the same
                      mask when the system abstained)
  08_error_after      where the final mask disagrees with the ground truth
  overview.png        the eight panels on one sheet

Frames are chosen by a fixed rule, not by eye: per test sequence, the frame the
segmenter scores best on, the routed frame the correction helped most, and the
frame the segmenter scores worst on.

The pictures show EndoVis frames, so they stay in results/ (git-ignored *.png)
and are NOT published on the site until the frame-image licence is settled.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sitr.metrics import official_frame_iou  # noqa: E402
from tools.check_stereo import PALETTE  # noqa: E402

PREP = os.path.join(ROOT, "work_v2", "prepared_512x640_full")
ACT = os.path.join(ROOT, "work_v2", "actions", "test_masks")
STE = os.path.join(ROOT, "work_v2", "stereo")
OUT = os.path.join(ROOT, "results", "pipeline")
NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma",
         "covered-kidney", "thread", "clamps", "suturing-needle", "suction-instrument", "small-intestine",
         "ultrasound-probe"]
KEY_OF = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "A3", "FUSE": "FUSE"}
CHOICE_TEXT = {"A0": "no correction (accepted or abstained)", "A1": "A1 stereo evidence", "A2": "A2 temporal",
               "A3": "A3 class-conditioned refinement", "FUSE": "FUSE stereo and temporal"}

COLOUR = np.zeros((256, 3), np.uint8)
for rgb, i in PALETTE.items():
    COLOUR[i] = rgb


def paint(m):
    out = COLOUR[m]
    out[m == 255] = (255, 0, 255)
    return out


def error_map(pred, gt):
    """white = agree, red = disagree on a labelled pixel, dark = background in both."""
    out = np.full(pred.shape + (3,), 255, np.uint8)
    both_bg = (pred == 0) & (gt == 0)
    out[both_bg] = (235, 235, 235)
    out[pred != gt] = (214, 39, 40)
    return out


def load_frame(seq, t):
    d = os.path.join(PREP, "test", seq)
    img = cv2.cvtColor(cv2.imread(os.path.join(d, "left_frames", f"frame{t:03d}.png")), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(os.path.join(d, "labels", f"frame{t:03d}.png"), cv2.IMREAD_UNCHANGED)  # class ids 0..11
    return img, gt


def masks(seq, t):
    z = np.load(os.path.join(ACT, f"test_{seq}", f"frame{t:03d}.npz"))
    return {k: z[k] for k in z.files if k in ("A0", "A1", "A2", "A3", "FUSE", "FUSE_REFINE")}


def disparity_image(seq, t):
    p = os.path.join(STE, f"test_{seq}", f"frame{t:03d}.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    d = np.nan_to_num(z["disparity"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ok = z["lr_ok"] & z["in_view"]
    if ok.sum() < 10:
        return None
    lo, hi = np.percentile(d[ok], [2, 98])
    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = cv2.cvtColor(cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~ok] = (40, 40, 40)
    return rgb


def select(ch):
    """Fixed rule; returns [(seq, t, kind)]."""
    rows = []
    for s in sorted(set(ch["seq"].tolist())):
        idx = np.where(ch["seq"] == s)[0]
        seq = s.split("/")[1]
        best = (-1, None); worst = (2, None); help_ = (0, None)
        for i in idx:
            t = int(ch["t"][i])
            try:
                img, gt = load_frame(seq, t)
                m = masks(seq, t)
            except (FileNotFoundError, OSError):
                continue
            a0 = official_frame_iou(m["A0"], gt, 12)
            if a0 is None:
                continue
            c = str(ch["choice"][i])
            fin = official_frame_iou(m.get(KEY_OF.get(c, "A0"), m["A0"]), gt, 12) if c != "A0" else a0
            if a0 > best[0]:
                best = (a0, t)
            if a0 < worst[0]:
                worst = (a0, t)
            if c != "A0" and fin is not None and fin - a0 > help_[0]:
                help_ = (fin - a0, t)
        rows.append((seq, best[1], "best"))
        rows.append((seq, worst[1], "worst"))
        if help_[1] is not None:
            rows.append((seq, help_[1], "corrected"))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    ch = np.load(os.path.join(ROOT, "results", "task5", "choices_test.npz"), allow_pickle=True)
    lookup = {(str(s).split("/")[1], int(t)): i for i, (s, t) in enumerate(zip(ch["seq"], ch["t"]))}
    index = []
    for seq, t, kind in select(ch):
        i = lookup[(seq, t)]
        img, gt = load_frame(seq, t)
        m = masks(seq, t)
        choice = str(ch["choice"][i])
        r = float(ch["r"][i]); routed = bool(ch["routed"][i])
        a0 = m["A0"]
        fin = m.get(KEY_OF.get(choice, "A0"), a0)
        iou0 = official_frame_iou(a0, gt, 12); iou1 = official_frame_iou(fin, gt, 12)
        disp = disparity_image(seq, t)
        tag = f"{seq}_frame{t:03d}_{kind}"
        d = os.path.join(OUT, tag)
        os.makedirs(d, exist_ok=True)

        def save(name, arr):
            cv2.imwrite(os.path.join(d, name), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        rel = np.full((512, 640, 3), 250, np.uint8)
        fig0, ax0 = plt.subplots(figsize=(6.4, 5.12), dpi=100)
        ax0.axis("off")
        ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
        ax0.text(0.5, 0.78, "Reliability score r", ha="center", fontsize=15, color="#333")
        ax0.text(0.5, 0.55, f"{r:.3f}", ha="center", fontsize=44, weight="bold")
        status = "flagged: sent to correction" if routed else ("low score, but not routed" if r < 0.64 else "accepted as it is")
        ax0.text(0.5, 0.36, status, ha="center", fontsize=15, color="#b45309" if routed else ("#7f1d1d" if r < 0.64 else "#166534"))
        ax0.text(0.5, 0.2, "threshold 0.64", ha="center", fontsize=12, color="#555")
        fig0.canvas.draw()
        rel = np.asarray(fig0.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig0)

        panels = [("01_input", "Input frame", img),
                  ("02_ground_truth", "Ground truth (for checking only)", paint(gt)),
                  ("03_stereo_disparity", "Stereo depth cue", disp if disp is not None else np.full_like(img, 60)),
                  ("04_segmentation", f"Segmentation (official IoU {iou0:.3f})", paint(a0)),
                  ("05_error_before", "Where it is wrong (red)", error_map(a0, gt)),
                  ("06_reliability", "Reliability gate", rel),
                  ("07_corrected", f"After: {CHOICE_TEXT.get(choice, choice)} ({iou1:.3f})", paint(fin)),
                  ("08_error_after", "Where it is wrong now (red)", error_map(fin, gt))]
        for name, _, arr in panels:
            save(name + ".png", arr)
        fig, axs = plt.subplots(2, 4, figsize=(16, 7.4))
        for ax, (_, title, arr) in zip(axs.ravel(), panels):
            ax.imshow(arr); ax.set_title(title, fontsize=10); ax.axis("off")
        fig.suptitle(f"EndoVis 2018 test {seq}, frame {t}: {kind} case", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "overview.png"), dpi=110)
        plt.close(fig)
        index.append(dict(folder=tag, seq=seq, frame=t, kind=kind, reliability=round(r, 3), flagged=routed,
                          choice=choice, official_iou_before=None if iou0 is None else round(iou0, 4),
                          official_iou_after=None if iou1 is None else round(iou1, 4)))
        print(tag, "r=%.3f" % r, choice, iou0, iou1)

    # dataset overview: class colours + one frame per sequence with its label
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.6))
    for j, seq in enumerate(["seq_1", "seq_2", "seq_3", "seq_4"]):
        t = 100
        img, gt = load_frame(seq, t)
        axs[0, j].imshow(img); axs[0, j].set_title(f"test {seq}, frame {t}", fontsize=10); axs[0, j].axis("off")
        axs[1, j].imshow(paint(gt)); axs[1, j].set_title("official label", fontsize=10); axs[1, j].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[i] / 255.0) for i in range(12)]
    fig.legend(handles, NAMES, ncol=6, loc="lower center", fontsize=9, frameon=False)
    fig.suptitle("The dataset: EndoVis 2018 test sequences and their 12-class labels", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=2.0)
    fig.savefig(os.path.join(OUT, "dataset_overview.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    lines = ["# Pipeline images, one picture per step", "",
             "Drawn from files the final evaluation already wrote. No model was re-run and no number here is new;",
             "the per-frame IoU in the table is the official per-frame rule at the 512x640 working grid, so it can",
             "differ in the third decimal from the full-resolution figures in the paper.", "",
             "Each folder holds `01_input.png` ... `08_error_after.png` and `overview.png`.",
             "`dataset_overview.png` shows one frame of each test sequence with its label and the 12 class colours.", "",
             "Frames were picked by a fixed rule: per sequence the best-scored, the worst-scored, and the routed frame the correction helped most.", "",
             "These are EndoVis frames: the PNGs stay local (git-ignored) and are not on the website until the frame-image licence is settled.", "",
             "| folder | kind | gate r | flagged | action | IoU before | IoU after |", "|---|---|---|---|---|---|---|"]
    for x in index:
        lines.append(f"| {x['folder']} | {x['kind']} | {x['reliability']} | {x['flagged']} | {x['choice']} | {x['official_iou_before']} | {x['official_iou_after']} |")
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
