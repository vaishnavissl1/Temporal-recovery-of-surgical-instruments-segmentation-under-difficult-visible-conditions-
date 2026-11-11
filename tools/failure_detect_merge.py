#!/usr/bin/env python3
"""
TASK C -- merge the measured latency into the failure-detection table (reads results/taskC/failure_detection.json and latency.json; computes nothing new).

    python -m tools.failure_detect_merge --dir results/taskC

Writes results/taskC/failure_detection_cost.json and .md: for every detector AUROC, AURC (with their sequence-clustered 95% intervals), forward passes, segmenters trained and
ms/frame, for the out-of-fold and the test groups. ms/frame is MEASURED for every row except the disparity control, which is COMPOSED: the measured gate latency plus the measured
mean stereo-matching time of Task 6 (results/task6/final_results.json, components_ms.stereo_evidence_cpu_ms_mean), because the disparity indicators need an SGBM run on the CPU.
"""
from __future__ import annotations

import argparse
import json
import os

LAT = {
    "max-softmax": "max-softmax", "mean entropy": "mean entropy", "Soft Dice Confidence": "Soft Dice Confidence",
    "MC stochastic depth (10 passes)": "MC stochastic depth (10 passes)", "TTA entropy (6 passes)": "TTA entropy (6 passes)",
    "TTA pairwise Dice (6 passes)": "TTA pairwise Dice (6 passes)", "4-model ensemble pairwise Dice": "4-model ensemble pairwise Dice",
    "gate: confidence only": "gate", "gate: + temporal (9 indicators)": "gate", "gate (GBM): 4 original indicators": "gate",
    "gate: production (MLP, 9 indicators)": "gate", "gate (GBM): 4 + 6 cross-view indicators": "xv", "gate (GBM): 6 cross-view indicators alone": "xv",
    "gate (GBM): 4 + 3 disparity indicators (control)": "disp",
}
KEY = {"gate": "gate (segmentation pass + indicators + estimator)", "xv": "gate + cross-view (two passes + six indicators)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/taskC")
    ap.add_argument("--task6", default="results/task6/final_results.json")
    a = ap.parse_args()
    R = json.load(open(os.path.join(a.dir, "failure_detection.json")))
    L = json.load(open(os.path.join(a.dir, "latency.json")))
    st = json.load(open(a.task6))["latency_ms"]["components_ms"]["stereo_evidence_cpu_ms_mean"]
    out = {"hardware": L["hardware"], "batch": L["batch"], "latency_method": L["method"], "groups": {}}
    md = []
    for g, title in (("oof", "Out of fold (15 training sequences; each frame scored by the cross-fitting model that excluded its sequence)"),
                     ("test", "Official test (4 sequences)")):
        G = R["groups"][g]
        rows = {}
        for name, m in G["detectors"].items():
            k = LAT[name]
            if k == "disp":
                ms, how = L["detectors"][KEY["gate"]]["mean_ms"] + st, "composed: measured gate + measured SGBM stereo matching (Task 6)"
            elif k in KEY:
                ms, how = L["detectors"][KEY[k]]["mean_ms"], "measured"
            else:
                ms, how = L["detectors"][k]["mean_ms"], "measured"
            rows[name] = dict(auroc=m["auroc"], auroc_ci95=m["auroc_ci95"], aurc=m["aurc"], aurc_ci95=m["aurc_ci95"], forward_passes=m["forward_passes"],
                              segmenters_trained=m["segmenters_trained"], ms_per_frame=ms, ms_basis=how, auroc_instr_label=m["auroc_instr_label"], note=m["note"])
        any_ = next(iter(G["detectors"].values()))
        out["groups"][g] = dict(n_frames=G["n_with_risk"], bad_rate=G["bad_rate"], aurc_floor=any_["aurc_oracle"], aurc_random=any_["aurc_random"], detectors=rows)
        md += [f"### {title}", "", f"{G['n_with_risk']} frames, bad-frame rate {100*G['bad_rate']:.1f}% (official IoU < 0.5). AURC floor {any_['aurc_oracle']:.4f}, random ranking {any_['aurc_random']:.4f}.", "",
               "| Detector | AUROC (95%) | AURC (95%) | passes | segmenters trained | ms/frame |", "|---|---|---|---|---|---|"]
        for n, r in rows.items():
            md.append(f"| {n}{' (test only)' if 'ensemble' in n else ''} | {r['auroc']:.3f} ({r['auroc_ci95'][0]:.3f}-{r['auroc_ci95'][1]:.3f}) | {r['aurc']:.4f} ({r['aurc_ci95'][0]:.4f}-{r['aurc_ci95'][1]:.4f}) | "
                      f"{r['forward_passes']} | {r['segmenters_trained']} | {r['ms_per_frame']:.0f}{'*' if r['ms_basis'] != 'measured' else ''} |")
        md.append("")
    md += ["`*` composed from measured components, not timed end to end. The intervals resample whole sequences (15 out of fold, 4 on test), so they are wide and overlap heavily.",
           f"Hardware: {L['hardware']}, batch {L['batch']}. {L['method']}."]
    json.dump(out, open(os.path.join(a.dir, "failure_detection_cost.json"), "w"), indent=1)
    open(os.path.join(a.dir, "failure_detection_cost.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
