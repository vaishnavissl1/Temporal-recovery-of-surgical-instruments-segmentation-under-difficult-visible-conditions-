#!/usr/bin/env python3
"""
Publish the paper's plots on the website, unchanged.

    python -m tools.export_web_figures            # writes web/public/figures/*.png and web/data/figures.json

Every figure is copied byte for byte from results/paper/figures (drawn by tools/paper_figures.py and
tools/paper_figures_p1.py from the result files; none contains a typed number) and its title and caption come from
results/paper/figures/_registry.json, the same text the paper uses. Only the figures that are plots are published:
none of them shows an EndoVis frame, and the frame-image figure of the paper (a sample-frame panel) is not in this
folder at all because of the frame-image licence. A SHA-256 of every published file is written beside it so that
tools/submission_check.py can confirm the website serves exactly these files and nothing else in the image class.
"""
import hashlib
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "paper", "figures")
OUT_IMG = os.path.join(ROOT, "web", "public", "figures")
OUT_JSON = os.path.join(ROOT, "web", "data", "figures.json")

# display order and grouping (a presentation choice; the text is the paper's)
GROUPS = [
    ("Segmentation baselines", ["fig03_per_class_baselines", "fig04_per_sequence_baselines"]),
    ("Reliability gate and failure detection", ["fig01_risk_coverage_test", "fig02_roc_test", "fig01_risk_coverage_oof", "fig02_roc_oof",
                                               "fig09_indicator_importance", "fig06_reliability"]),
    ("Temporal evidence", ["fig05_consecutive_frame_iou", "fig10_aligner_comparison", "fig11_memory_starvation"]),
    ("Correction and cost", ["fig07_action_change_distributions", "fig08_accuracy_vs_cost"]),
    ("Metric definition", ["fig12_metric_ablation"]),
]


def main():
    reg = json.load(open(os.path.join(SRC, "_registry.json"), encoding="utf-8"))
    listed = [f for _, ids in GROUPS for f in ids]
    missing = [f for f in reg if f not in listed]
    if missing:
        raise SystemExit(f"figures in the registry that no group lists: {missing}")
    os.makedirs(OUT_IMG, exist_ok=True)
    for old in os.listdir(OUT_IMG):
        os.remove(os.path.join(OUT_IMG, old))
    out = []
    for g, ids in GROUPS:
        items = []
        for fid in ids:
            png = os.path.join(SRC, fid + ".png")
            shutil.copyfile(png, os.path.join(OUT_IMG, fid + ".png"))
            r = reg[fid]
            items.append(dict(id=fid, file=f"/figures/{fid}.png", title=r["title"], caption=r["caption"],
                              sources=[s for s in r["sources"] if s.startswith("results/")],
                              sha256=hashlib.sha256(open(png, "rb").read()).hexdigest()))
        out.append(dict(group=g, figures=items))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print(f"published {len(listed)} figures in {len(GROUPS)} groups -> {OUT_IMG}")


if __name__ == "__main__":
    main()
