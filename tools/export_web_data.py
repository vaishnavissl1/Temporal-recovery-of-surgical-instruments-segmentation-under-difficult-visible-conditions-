#!/usr/bin/env python3
"""Collect the aggregate numbers for the results website into web/data/results.json.

Reads only files under results/ (aggregates). No dataset images, no per-patient frames, no weights.

    python -m tools.export_web_data --results results --out web/data/results.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLASS_LABELS = {  # display names for the 12 classes (labels.json naming)
    "instrument-shaft": "Instrument shaft", "instrument-clasper": "Instrument clasper",
    "instrument-wrist": "Instrument wrist", "kidney-parenchyma": "Kidney parenchyma",
    "covered-kidney": "Covered kidney", "thread": "Thread", "clamps": "Clamps",
    "suturing-needle": "Suturing needle", "suction-instrument": "Suction instrument",
    "intestine": "Intestine", "ultrasound-probe": "Ultrasound probe",
}


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stereo_pairs(report_path):
    """Per-pair vertical error table from STEREO_REPORT.md (calibration only vs + self-calibration)."""
    rows = []
    for line in open(report_path, encoding="utf-8"):
        m = re.match(r"\|\s*((?:train|test)/seq_\d+) frame\d+\.png\s*\|\s*([\d.]+) / [\d.]+ / [\d.]+\s*\|\s*([\d.]+) / [\d.]+ / [\d.]+\s*\|", line)
        if m and not any(r["seq"] == m.group(1) for r in rows):
            rows.append({"seq": m.group(1), "calib_only": float(m.group(2)), "self_calibrated": float(m.group(3))})
    return rows


def policy_rows(block):
    out = []
    for name, v in block.items():
        out.append({
            "policy": name,
            "activation_pct": v["activation_pct_of_all"],
            "mean_delta_all": v["mean_delta_all_frames"],
            "mean_delta_when_corrected": v["mean_delta_when_corrected"],
            "harmful_pct_of_corrected": v["harmful_rate_of_corrected"],  # already in percent in selector_results.json
            "harmful_pct_of_all": v["harmful_pct_of_all"],
            "improved_pct_of_corrected": v["improved_rate_of_corrected"],  # already in percent
            "cpu_ms": v["mean_correction_cpu_ms_per_frame"],
            "action_share": v.get("action_share", {}),
            "by_difficulty_quartile": v.get("by_difficulty_quartile", []),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="web/data/results.json")
    a = ap.parse_args()
    R = a.results
    t6 = load(os.path.join(R, "task6", "final_results.json"))
    t5 = load(os.path.join(R, "task5", "selector_results.json"))
    t4o = load(os.path.join(R, "task4", "gate_comparison_gbm.json"))
    t4t = load(os.path.join(R, "task4", "gate_comparison_test.json"))
    t2 = load(os.path.join(R, "task2", "stereo_check.json"))

    models = []
    for k, m in t6["models"].items():
        lat = t6["latency_ms"].get(k, {})
        models.append({
            "id": k, "description": t6["descriptions"][k], "mIoU": m["mIoU"], "boundary_f1": m["boundary_f1_instruments"],
            "cldice": m["cldice"], "activation_pct": m["activation_pct"], "harmful_pct_of_all": m["harmful_pct_of_all"],
            "harmful_pct_of_corrected": m["harmful_pct_of_corrected"], "improved_pct_of_corrected": m["improved_pct_of_corrected"],
            "mean_delta_vs_M1": m["mean_delta_vs_M1_all"], "aurc_gate": m["aurc_gate"],
            "latency_mean_ms": lat.get("mean_ms"), "latency_p95_ms": lat.get("p95_ms"),
            "per_sequence": m["per_sequence"], "by_difficulty_quartile": m["by_difficulty_quartile"],
            "per_class": {CLASS_LABELS.get(c, c): v["iou"] for c, v in m["per_class"].items()},
        })

    def gate(src):
        return [{"set": k, "auc": v["auc"], "aurc": v["aurc"]} for k, v in src["sets"].items()]

    def paired(src):
        return [{"pair": k, "auc_diff": v["auc"]["mean"], "auc_lo": v["auc"]["ci95"][0], "auc_hi": v["auc"]["ci95"][1],
                 "verdict": v["verdict"]} for k, v in src.items()]

    out = {
        "meta": {"n_test_frames": t6["n_frames"], "n_test_sequences": t6["n_sequences"], "n_routed": t6["n_routed"],
                 "harm_margin": t6["harm_margin"], "boundary_tol_px": t6["boundary_tol_px"]},
        "crosscheck": t6["crosscheck_official_run_py"],
        "gate_detection": t6["gate"],
        "headline_M4_vs_M6": t6["headline_M4_vs_M6"],
        "latency": t6["latency_ms"],
        "models": models,
        "stereo": {
            "raw_dy_mean_px": t2["raw_dy_mean_over_pairs"],
            "pairs": stereo_pairs(os.path.join(R, "task2", "STEREO_REPORT.md")),
            "self_calibration": [{"seq": s, "max_abs_px": v["max_abs_px"], "applied": v["applied"]}
                                 for s, v in t2["vertical_corrections"].items()],
            "lr_consistent_pct": {"calibration_only": t2["eval_calibration_only"]["overall"]["lr_pct"],
                                  "refined": t2["eval_refined"]["overall"]["lr_pct"]},
            "instrument_lr_pct": {"calibration_only": t2["eval_calibration_only"]["instrument"]["lr_pct"],
                                  "refined": t2["eval_refined"]["instrument"]["lr_pct"]},
            "gate_oof": gate({"sets": t4o["cv"]}), "gate_test": gate(t4t),
            "paired_oof": paired(t4o["cv_paired"]), "paired_test": paired(t4t["paired"]),
            "leave_one_out": [{"indicator": k, "auc_drop": v["auc_drop"]} for k, v in t4o["loo"].items()],
        },
        "selector": {
            "tau": t5["tau"], "margin": t5["margin"],
            "oof": policy_rows(t5["oof"]), "test": policy_rows(t5["test"]),
            "n_oof": t5["n_oof"], "n_oof_routed": t5["n_oof_routed"], "n_test": t5["n_test"], "n_test_routed": t5["n_test_routed"],
            "prediction_quality_oof": t5["prediction_quality_oof"], "margin_sweep_oof": t5["margin_sweep_oof"],
            "stop_check_oof": t5["stop_check_oof"], "stop_check_test": t5["stop_check_test"],
        },
        # Task 3 numbers are printed by tools.measure_recovery; copied from results/task3*/..._log.txt
        "recovery": {
            "source": ["results/task3/measure_recovery_test_log.txt", "results/task3_oof/measure_recovery_oof_log.txt"],
            "test_always": {"routed": 880, "frames": 997, "instrument_before": 0.6105, "instrument_after": 0.6140, "ci": [0.0002, 0.0081],
                            "official_before": 0.6397, "official_after": 0.6397, "official_ci": [-0.0002, 0.0002]},
            "oof_always": {"routed": 1741, "frames": 2235, "instrument_before": 0.6597, "instrument_after": 0.6627, "ci": [0.0010, 0.0055],
                           "official_before": 0.6772, "official_after": 0.6774, "official_ci": [0.0001, 0.0004]},
            "oof_selective": {"routed": 623, "frames": 2235, "instrument_before": 0.5289, "instrument_after": 0.5374, "ci": [0.0028, 0.0161],
                              "official_before": 0.6227, "official_after": 0.6234, "official_ci": [0.0003, 0.0011]},
        },
        "difficulty": {"oof_spearman": -0.029, "test_spearman": 0.081, "verdict": "not validated"},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}: {len(models)} models, {len(out['stereo']['pairs'])} stereo pairs, "
          f"{len(out['selector']['test'])} test policies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
