#!/usr/bin/env python3
"""
One finding, two symptoms: the production gate's ABSOLUTE scores are optimistic on the test sequences while its RANKING is usable.

    python -m tools.gate_calibration_offset --out results/paper/gate_calibration_offset.json

Reads only stored outputs. Reports (a) mean gate score against mean achieved instrument IoU, out of fold (2235 training frames) and on test (997 frames); (b) the conformal budget (0.05) against the
risk met on test (results/web_extract/stage2_gate.json); (c) the gate's AUROC on the official-IoU risk and on its own instrument-IoU label (results/taskC/failure_detection.json).
It also states, in the output, what is and is not established: the two symptoms are consistent with a single cause (the gate was fitted on out-of-fold training sequences whose achieved IoU is higher than
the test sequences'), which has NOT been tested; a calibration shift of this kind does not affect the ranking much, and the ranking is what the threshold relies on only approximately.
"""
import argparse
import csv
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run as a script or as -m; the IDE's interpreter does not know this path, which is its "unresolved import" warning

import tools.paper_figures as PF  # noqa: E402

GATE = "gate: production (MLP, 9 indicators)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/paper/gate_calibration_offset.json")
    a = ap.parse_args()
    ns = types.SimpleNamespace(work="work_v2", scores="work_v2/failure_detection", taskC="results/taskC", taskD="results/taskD", xv_dir="results/taskD")
    D = PF.detector_frames(ns)
    out = {"description": __doc__, "groups": {}}
    for g in ("oof", "test"):
        risk, det = D[g]
        s = det[GATE]
        y = np.load(f"results/taskD/gate_scores_{g}.npz")["r_star"]          # achieved instrument IoU (the gate's regression target); mean is order-independent
        assert len(s) == len(y)
        out["groups"][g] = dict(n_frames=int(len(s)), mean_gate_score=float(s.mean()), mean_achieved_instrument_iou=float(y.mean()), offset_score_minus_achieved=float(s.mean() - y.mean()))
    t = [r for r in csv.DictReader(open("results/taskC/per_frame_scores.csv", encoding="utf-8")) if r["group"] == "test"]
    ts = np.array([float(r["gate: production (MLP; 9 indicators)"]) for r in t])
    ty = np.array([float(r["instrument_iou_risk_label"]) for r in t])
    assert abs(ts.mean() - out["groups"]["test"]["mean_gate_score"]) < 1e-3 and abs(ty.mean() - out["groups"]["test"]["mean_achieved_instrument_iou"]) < 1e-3
    edges = np.quantile(ts, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(edges, ts, side="right") - 1, 0, 9)
    out["test_decile_offsets_score_minus_achieved"] = [float(ts[idx == b].mean() - ty[idx == b].mean()) for b in range(10)]
    S2 = json.load(open("results/web_extract/stage2_gate.json"))
    out["conformal"] = dict(budget_alpha=S2["alpha"], risk_met_on_selection_out_of_fold=S2["empirical_risk_threshold_selection"], risk_on_test=S2["risk_bad_accepted_test"])
    F = json.load(open("results/taskC/failure_detection.json"))["groups"]
    out["ranking"] = {g: dict(auroc_official_risk=F[g]["detectors"][GATE]["auroc"], auroc_official_risk_ci95=F[g]["detectors"][GATE]["auroc_ci95"], auroc_own_label=F[g]["detectors"][GATE]["auroc_instr_label"])
                      for g in ("oof", "test")}
    out["interpretation"] = ("Ranking is moderate and absolute values are not trustworthy: on test the mean score exceeds the mean achieved instrument IoU, out of fold it does not, and the conformal budget met out "
                             "of fold is missed on test. The two symptoms are CONSISTENT with one cause (the test sequences are harder than the out-of-fold training sequences the gate and the threshold were fitted on); "
                             "that cause was not tested here.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for g, v in out["groups"].items():
        print(g, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
    print("decile offsets (test):", [round(x, 3) for x in out["test_decile_offsets_score_minus_achieved"]])
    print("conformal:", out["conformal"])
    print("ranking:", out["ranking"])


if __name__ == "__main__":
    main()
