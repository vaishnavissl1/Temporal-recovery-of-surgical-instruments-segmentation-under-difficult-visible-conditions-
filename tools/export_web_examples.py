#!/usr/bin/env python3
"""
A few before/after pictures for the website's "My results" page, composed from the pipeline images that
tools/pipeline_images.py already wrote (results/pipeline). Nothing is re-run.

    python -m tools.export_web_examples      # writes web/public/examples/*.jpg and web/data/examples.json

Frames were chosen by the fixed rule of tools/pipeline_images.py (per test sequence the best-scored, the
worst-scored and the routed frame the correction helped most), not by eye, and the worst cases are shown beside
the best ones. IoU values are the official per-frame rule at the 512x640 working grid (results/pipeline/index.json).
Every picture is one EndoVis 2018 test frame with its label and our output; the source is credited on the page.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sitr.metrics import official_frame_iou  # noqa: E402
from tools.check_stereo import PALETTE  # noqa: E402
SRC = os.path.join(ROOT, "results", "pipeline")
OUT_IMG = os.path.join(ROOT, "web", "public", "examples")
OUT_JSON = os.path.join(ROOT, "web", "data", "examples.json")
PW, PH, HEAD = 400, 320, 30
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 17) if os.path.exists(r"C:\Windows\Fonts\arial.ttf") else ImageFont.load_default()


def compose(folder, steps, labels, out):
    sheet = Image.new("RGB", (PW * len(steps), PH + HEAD), "white")
    d = ImageDraw.Draw(sheet)
    for i, (s, lab) in enumerate(zip(steps, labels)):
        im = Image.open(os.path.join(SRC, folder, s)).convert("RGB").resize((PW - 4, PH - 4), Image.LANCZOS)
        sheet.paste(im, (i * PW + 2, HEAD + 2))
        d.text((i * PW + 8, 5), lab, fill=(20, 20, 20), font=FONT)
    sheet.save(out, "JPEG", quality=85, optimize=True)


NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma", "covered-kidney",
         "thread", "clamps", "suturing-needle", "suction-instrument", "small-intestine", "ultrasound-probe"]
COLOUR = np.zeros((256, 3), np.uint8)
for _rgb, _i in PALETTE.items():
    COLOUR[_i] = _rgb


def overlay(img, m):
    out = img.copy()
    fg = m != 0
    out[fg] = (0.45 * img[fg] + 0.55 * COLOUR[m][fg]).astype(np.uint8)
    return out


def samples(out_dir):
    """One representative frame per test sequence: the frame whose official IoU is closest to that sequence's median
    (a fixed rule, so these are typical frames, not the best ones). Frame, ground truth and my final mask, both drawn over the frame."""
    prep = os.path.join(ROOT, "work_v2", "prepared_512x640_full", "test")
    acts = os.path.join(ROOT, "work_v2", "actions", "test_masks")
    ch = np.load(os.path.join(ROOT, "results", "taskF", "no_a1", "choices_test.npz"), allow_pickle=True)
    res = []
    for seq in ("seq_1", "seq_2", "seq_3", "seq_4"):
        rows = []
        for i in np.where(ch["seq"] == "test/" + seq)[0]:
            t = int(ch["t"][i])
            gt = cv2.imread(os.path.join(prep, seq, "labels", f"frame{t:03d}.png"), cv2.IMREAD_UNCHANGED)
            z = np.load(os.path.join(acts, "test_" + seq, f"frame{t:03d}.npz"))
            m = z[str(ch["choice"][i])]
            v = official_frame_iou(m, gt, 12)
            if v is not None:
                rows.append((t, v))
        med = float(np.median([v for _, v in rows]))
        t, v = min(rows, key=lambda r: abs(r[1] - med))
        img = cv2.cvtColor(cv2.imread(os.path.join(prep, seq, "left_frames", f"frame{t:03d}.png")), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(prep, seq, "labels", f"frame{t:03d}.png"), cv2.IMREAD_UNCHANGED)
        i = int(np.where((ch["seq"] == "test/" + seq) & (ch["t"] == t))[0][0])
        m = np.load(os.path.join(acts, "test_" + seq, f"frame{t:03d}.npz"))[str(ch["choice"][i])]
        panels = [("Dataset frame", img), ("Ground truth on the frame", overlay(img, gt)), ("My result on the frame", overlay(img, m))]
        sheet = Image.new("RGB", (PW * 3, PH + HEAD), "white")
        d = ImageDraw.Draw(sheet)
        for k, (lab, arr) in enumerate(panels):
            sheet.paste(Image.fromarray(arr).resize((PW - 4, PH - 4), Image.LANCZOS), (k * PW + 2, HEAD + 2))
            d.text((k * PW + 8, 5), lab, fill=(20, 20, 20), font=FONT)
        fn = f"sample_{seq}_frame{t:03d}.jpg"
        sheet.save(os.path.join(out_dir, fn), "JPEG", quality=86, optimize=True)
        res.append(dict(file=f"/examples/{fn}", seq=seq.replace("seq_", "sequence "), frame=t, iou=round(float(v), 4), median_iou=round(med, 4)))
    # the colour key, drawn once
    key = Image.new("RGB", (PW * 3, 74), "white")
    d = ImageDraw.Draw(key)
    for k, n in enumerate(NAMES):
        x, y = 8 + (k % 6) * 196, 6 + (k // 6) * 32
        d.rectangle([x, y + 3, x + 18, y + 21], fill=tuple(int(c) for c in COLOUR[k]), outline=(120, 120, 120))
        d.text((x + 24, y + 3), n, fill=(30, 30, 30), font=FONT)
    key.save(os.path.join(out_dir, "class_key.jpg"), "JPEG", quality=90)
    return res


def main():
    idx = {x["folder"]: x for x in json.load(open(os.path.join(SRC, "index.json")))}
    os.makedirs(OUT_IMG, exist_ok=True)
    for f in os.listdir(OUT_IMG):
        os.remove(os.path.join(OUT_IMG, f))

    def item(folder, steps, labels, title):
        x = idx[folder]
        fn = folder + ".jpg"
        compose(folder, steps, labels, os.path.join(OUT_IMG, fn))
        return dict(file=f"/examples/{fn}", title=title, seq=x["seq"].replace("seq_", "sequence "), frame=x["frame"],
                    iou_before=x["official_iou_before"], iou_after=x["official_iou_after"], action=x["choice"], reliability=x["reliability"],
                    flagged=x["flagged"], kind=x["kind"])

    cmp_steps = ["01_input.png", "02_ground_truth.png", "04_segmentation.png", "05_error_before.png"]
    cmp_labels = ["Input frame", "Ground truth", "My segmentation", "Where it differs (red)"]
    ba_steps = ["04_segmentation.png", "07_corrected.png", "05_error_before.png", "08_error_after.png"]
    ba_labels = ["Before correction", "After correction", "Errors before (red)", "Errors after (red)"]

    seg = [item("seq_1_frame079_best", cmp_steps, cmp_labels, "A well-segmented frame"),
           item("seq_3_frame123_best", cmp_steps, cmp_labels, "A well-segmented frame"),
           item("seq_1_frame248_worst", cmp_steps, cmp_labels, "A poorly segmented frame"),
           item("seq_4_frame013_worst", cmp_steps, cmp_labels, "A poorly segmented frame")]
    ba = [item("seq_3_frame143_corrected", ba_steps, ba_labels, "Correction on a flagged frame"),
          item("seq_4_frame210_corrected", ba_steps, ba_labels, "Correction on a flagged frame"),
          item("seq_1_frame023_corrected", ba_steps, ba_labels, "Correction on a flagged frame")]
    # the dataset: one frame per test sequence with its label
    ds = Image.open(os.path.join(SRC, "dataset_overview.png")).convert("RGB")
    ds.thumbnail((1600, 1600), Image.LANCZOS)
    ds.save(os.path.join(OUT_IMG, "dataset_overview.jpg"), "JPEG", quality=85, optimize=True)
    smp = samples(OUT_IMG)
    out = dict(samples=smp, class_key="/examples/class_key.jpg", segmentation=seg, correction=ba, dataset=dict(file="/examples/dataset_overview.jpg"))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    n = len(os.listdir(OUT_IMG))
    print(f"wrote {n} images, {sum(os.path.getsize(os.path.join(OUT_IMG, f)) for f in os.listdir(OUT_IMG)) / 1e3:.0f} kB -> {OUT_IMG}")


if __name__ == "__main__":
    main()
