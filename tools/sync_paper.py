#!/usr/bin/env python3
"""
Copy the generated paper assets into the paper's folder and rebuild the number macros, so main.tex always reads current results.

    python -m tools.sync_paper --paper "C:/Users/Chandaluri Vaishnavi/Downloads/cvrsfesp/cvavish"

Copies results/paper/tables/*.tex to <paper>/tables and the PDF figures of results/paper/figures to <paper>/figures, runs tools/paper_numbers.py to write <paper>/numbers.tex, and writes
<paper>/baseline_text.tex (the prose of the baseline subsection, from results/taskP/baselines.json) when that file exists.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.paper, "tables"), exist_ok=True)
    os.makedirs(os.path.join(a.paper, "figures"), exist_ok=True)
    for f in glob.glob("results/paper/tables/*.tex"):
        shutil.copy2(f, os.path.join(a.paper, "tables"))
    for f in glob.glob("results/paper/figures/*.pdf"):
        shutil.copy2(f, os.path.join(a.paper, "figures"))
    subprocess.check_call([sys.executable, "-m", "tools.paper_numbers", "--out", "results/paper", "--tex", os.path.join(a.paper, "numbers.tex")])
    # prose for the baseline subsection, only from a real results file
    bp = "results/taskP/baselines.json"
    txt = os.path.join(a.paper, "baseline_text.tex")
    if os.path.exists(bp):
        B = json.load(open(bp))
        ms = B["models"]
        ours = next((m for m in ms if m["label"].startswith("Ours")), None)
        parts = []
        if ours:
            parts.append(f"The proposed model scores {ours['mIoU']:.4f} mIoU and {ours['mcIoU']:.4f} mcIoU with {ours['params_millions']:.1f} million parameters at {ours['ms_per_frame']:.0f}\\,ms per frame.")
        for m in ms:
            if m is ours:
                continue
            d = m.get("difference_from_ours_mIoU_points")
            ci = m.get("difference_from_ours_ci95_points")
            parts.append(f"{m['label']} scores {m['mIoU']:.4f} mIoU and {m['mcIoU']:.4f} mcIoU with {m['params_millions']:.1f} million parameters at {m['ms_per_frame']:.0f}\\,ms per frame"
                         + (f", {d:+.2f} mIoU points from the proposed model (interval {ci[0]:+.2f} to {ci[1]:+.2f})." if d is not None else "."))
        miss = [u["label"] for u in B.get("unavailable", [])]
        if miss:
            parts.append("Not finished, and therefore absent: " + "; ".join(miss) + ".")
        parts.append("Pretraining differs by architecture, so the comparison mixes architecture and initialisation; with four test sequences these differences are indicative, not conclusive.")
        open(txt, "w", encoding="utf-8").write(" ".join(parts).replace("%", "\\%") + "\n")
    elif os.path.exists(txt):
        os.remove(txt)
    print("synced paper assets to", a.paper)


if __name__ == "__main__":
    main()
