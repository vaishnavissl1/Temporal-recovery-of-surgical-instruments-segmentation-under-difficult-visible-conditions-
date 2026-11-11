#!/usr/bin/env python3
"""Collect the numbers for the results website into web/data/results.json.

Two stages, so that every displayed number traces to a JSON file under results/:

  stage 1  (extract)  turns the few sources that are not JSON under results/ into JSON files under
                      results/web_extract/, each carrying a "source" field:
                        per_frame_test.json   from results/task5/choices_test.npz + work_v2/actions/test__*.npz
                        stage2_gate.json      from work_v2/stage2/stage2_report.json (conformal risk target and value)
                        task3_recovery.json   parsed from results/task3*/measure_recovery_*_log.txt
  stage 2  (build)    reads only JSON files under results/ and writes web/data/results.json, plus a "_sources"
                      map {block: [files]} that tools/verify_web_data.py uses to check every numeric field.

No dataset images, no per-patient frames, no weights.

    python -m tools.export_web_data --results results --out web/data/results.json [--actions work_v2/actions --work work_v2]
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
ALIGNERS = ("identity", "farneback", "region")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------- stage 1: extract
def extract_per_frame(results, actions, out):
    import numpy as np

    choices_path = os.path.join(results, "task5", "choices_test.npz")
    c = np.load(choices_path, allow_pickle=True)
    frames = {}
    for seq in sorted(set(c["seq"].tolist())):
        m = c["seq"] == seq
        a = np.load(os.path.join(actions, f"test__{seq.replace('/', '_')}.npz"), allow_pickle=True)
        t = c["t"][m]
        assert np.array_equal(t, a["t"][: len(t)]), "frame order mismatch"
        choice = c["choice"][m].tolist()
        gains = {k: np.nan_to_num(a[f"d_{k}"][: len(t)]) for k in ("A1", "A2", "A3", "FUSE")}
        chosen = [float(gains[ch][i]) if ch in gains else 0.0 for i, ch in enumerate(choice)]
        pred = {k: np.nan_to_num(c[f"pred_{k}"][m]) for k in ("A1", "A2", "A3", "FUSE")}
        pred_chosen = [float(pred[ch][i]) if ch in pred else 0.0 for i, ch in enumerate(choice)]
        r4 = lambda v: [round(float(x), 4) for x in v]
        frames[seq] = {
            "n": int(len(t)), "t": t.tolist(), "r": r4(c["r"][m]), "routed": c["routed"][m].astype(int).tolist(),
            "choice": choice, "before": r4(c["instr_before"][m]), "gain": r4(chosen), "pred_gain": r4(pred_chosen),
            "difficulty": r4(c["difficulty"][m]), "gain_by": {k: r4(v) for k, v in gains.items()},
        }
    json.dump({"source": "results/task5/choices_test.npz + work_v2/actions/test__*.npz (identity aligner; "
                         "the system of record). Values rounded to 4 decimals.", "frames": frames},
              open(out, "w", encoding="utf-8"))


def extract_stage2(work, out):
    d = load(os.path.join(work, "stage2", "stage2_report.json"))
    json.dump({"source": "work_v2/stage2/stage2_report.json",
               "alpha": d["args"]["alpha"],
               "risk_bad_accepted_test": d["test_self_predicted_history"]["risk_bad_accepted"],
               "empirical_risk_threshold_selection": d["threshold"]["empirical_risk"]}, open(out, "w"), indent=1)


def extract_stereo_pairs(results, out):
    rep = os.path.join(results, "task2", "STEREO_REPORT.md")
    json.dump({"source": "results/task2/STEREO_REPORT.md (per-pair table: first row of each sequence)",
               "pairs": stereo_pairs(rep)}, open(out, "w"), indent=1)


def parse_recovery_log(path, mode):
    """Numbers printed by tools.measure_recovery for one mode ('SELECTIVE' or 'ALWAYS')."""
    txt = open(path, encoding="utf-8").read()
    sec = txt.split(f"--- {mode} ---")[1]
    nxt = re.search(r"\n--- [A-Z]+ ---", sec)
    sec = sec[: nxt.start()] if nxt else sec
    fr = re.search(r"frames=(\d+) routed=(\d+)", sec)
    pat = r"{k} IoU\s+n=\s*(\d+)\s+before ([\d.]+) -> after ([\d.]+)\s+mean change ([+-][\d.]+) \(95% CI ([+-][\d.]+)\.\.([+-][\d.]+)\)"
    ins = re.search(pat.format(k="instrument"), sec)
    off = re.search(pat.format(k="official"), sec)
    f = float
    return {"frames": int(fr.group(1)), "routed": int(fr.group(2)),
            "instrument_before": f(ins.group(2)), "instrument_after": f(ins.group(3)), "ci": [f(ins.group(5)), f(ins.group(6))],
            "official_before": f(off.group(2)), "official_after": f(off.group(3)), "official_ci": [f(off.group(5)), f(off.group(6))]}


def extract_task3(results, out):
    t = os.path.join(results, "task3", "measure_recovery_test_log.txt")
    o = os.path.join(results, "task3_oof", "measure_recovery_oof_log.txt")
    json.dump({"source": ["results/task3/measure_recovery_test_log.txt", "results/task3_oof/measure_recovery_oof_log.txt"],
               "test_always": parse_recovery_log(t, "ALWAYS"), "oof_always": parse_recovery_log(o, "ALWAYS"),
               "oof_selective": parse_recovery_log(o, "SELECTIVE"),
               "test_selective": parse_recovery_log(t, "SELECTIVE")}, open(out, "w"), indent=1)


# --------------------------------------------------------------------------------------- stage 2: build
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
            "mean_delta_flagged": v["mean_delta_routed"],
            "mean_delta_when_corrected": v["mean_delta_when_corrected"],
            "mean_delta_official_all": v["mean_delta_official_all_frames"],
            "harmful_pct_of_corrected": v["harmful_rate_of_corrected"],  # already in percent in selector_results.json
            "harmful_pct_of_all": v["harmful_pct_of_all"],
            "improved_pct_of_corrected": v["improved_rate_of_corrected"],  # already in percent
            "cpu_ms": v["mean_correction_cpu_ms_per_frame"],
            "action_share": v.get("action_share", {}),
            "by_difficulty_quartile": v.get("by_difficulty_quartile", []),
        })
    return out


def build(R, out_path):
    src = {}
    t6p, t5p = "task6/final_results.json", "task5/selector_results.json"
    t6, t5 = load(os.path.join(R, t6p)), load(os.path.join(R, t5p))
    t4o = load(os.path.join(R, "task4", "gate_comparison_gbm.json"))
    t4t = load(os.path.join(R, "task4", "gate_comparison_test.json"))
    t4d = load(os.path.join(R, "task4", "difficulty_report.json"))
    t2 = load(os.path.join(R, "task2", "stereo_check.json"))
    pf = load(os.path.join(R, "web_extract", "per_frame_test.json"))
    s2 = load(os.path.join(R, "web_extract", "stage2_gate.json"))
    t3 = load(os.path.join(R, "web_extract", "task3_recovery.json"))
    cmp_ = load(os.path.join(R, "taskA", "aligner_comparison.json"))
    ptab = load(os.path.join(R, "taskB", "policy_table.json"))
    gaps = {a: load(os.path.join(R, "taskB", f"metric_gap_{a}.json")) for a in ("A3", "A2", "FUSE")}
    sel_al = {a: load(os.path.join(R, "taskA", a, "selector_results.json")) for a in ALIGNERS}
    fin_al = {"identity": t6, **{a: load(os.path.join(R, "taskA", a, "final", "final_results.json")) for a in ("farneback", "region")}}

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
            "class_frames": {CLASS_LABELS.get(c, c): v["n_frames"] for c, v in m["per_class"].items()},
        })

    gate = lambda s: [{"set": k, "auc": v["auc"], "aurc": v["aurc"]} for k, v in s["sets"].items()]
    paired = lambda s: [{"pair": k, "auc_diff": v["auc"]["mean"], "auc_lo": v["auc"]["ci95"][0], "auc_hi": v["auc"]["ci95"][1],
                         "verdict": v["verdict"]} for k, v in s.items()]

    def al_block(group):
        G = cmp_["groups"][group]
        return {a: {k: {x: v[x] for x in ("n_flagged", "mean_on_flagged", "mean_all_frames", "pct_improved_of_flagged",
                                           "pct_worse_of_flagged", "pct_unchanged_of_flagged", "pct_exactly_zero_of_flagged")}
                     for k, v in G["policies"][a].items() if isinstance(v, dict)} for a in ALIGNERS}

    out = {
        "frames": pf["frames"],
        "meta": {"n_test_frames": t6["n_frames"], "n_test_sequences": t6["n_sequences"], "n_routed": t6["n_routed"],
                 "n_train_sequences": cmp_["groups"]["oof"]["n_sequences"], "n_oof_frames": t5["n_oof"],
                 "harm_margin": t6["harm_margin"], "boundary_tol_px": t6["boundary_tol_px"]},
        "crosscheck": t6["crosscheck_official_run_py"],
        "gate_detection": t6["gate"],
        "headline_M4_vs_M6": t6["headline_M4_vs_M6"],
        "latency": t6["latency_ms"],
        "models": models,
        "stereo": {
            "raw_dy_mean_px": t2["raw_dy_mean_over_pairs"],
            "pairs": load(os.path.join(R, "web_extract", "stereo_pairs.json"))["pairs"],
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
        "recovery": {k: v for k, v in t3.items() if k != "source"},
        "difficulty": {"oof_spearman": t4d["oof_train_sequences"]["vs_instrument_error"]["spearman"],
                       "test_spearman": t4d["official_test_sequences"]["vs_instrument_error"]["spearman"]},
        "conformal": {"alpha": s2["alpha"], "risk_bad_accepted_test": s2["risk_bad_accepted_test"]},
        "stereo_matching": {k: {"instrument_lr_pct": t2[e]["instrument"]["lr_pct"], "tissue_lr_pct": t2[e]["tissue"]["lr_pct"],
                              "instrument_valid_pct": t2[e]["instrument"]["valid_pct"], "tissue_valid_pct": t2[e]["tissue"]["valid_pct"]}
                            for k, e in (("calibration_only", "eval_calibration_only"), ("refined", "eval_refined"))},
        "taskA": {
            "oof": al_block("oof"), "test": al_block("test"),
            "diff_oof": cmp_["groups"]["oof"]["paired_difference_vs_reference"],
            "diff_test": cmp_["groups"]["test"]["paired_difference_vs_reference"],
            "n_oof_sequences": cmp_["groups"]["oof"]["n_sequences"], "margin_eval": cmp_["margin_eval"],
            "miou": {a: {k: m["mIoU"] for k, m in fin_al[a]["models"].items()} for a in ALIGNERS},
            "selector_flagged": {a: {g: {p: sel_al[a][g][p]["mean_delta_routed"] for p in ("selector", "always-A3 (routed)", "oracle (routed)")}
                                     for g in ("oof", "test")} for a in ALIGNERS},
            "a2_pearson_oof": {a: sel_al[a]["prediction_quality_oof"]["A2"]["pearson"] for a in ALIGNERS},
        },
        "taskB": {"policy_table": ptab, "metric_gap": gaps},
    }
    out["_sources"] = {
        "frames": ["web_extract/per_frame_test.json"],
        "meta": [t6p, "taskA/aligner_comparison.json", t5p],
        "crosscheck": [t6p], "gate_detection": [t6p], "headline_M4_vs_M6": [t6p], "latency": [t6p], "models": [t6p],
        "stereo": ["task2/stereo_check.json", "task4/gate_comparison_gbm.json", "task4/gate_comparison_test.json",
                   "web_extract/stereo_pairs.json"],
        "selector": [t5p], "recovery": ["web_extract/task3_recovery.json"], "difficulty": ["task4/difficulty_report.json"],
        "conformal": ["web_extract/stage2_gate.json"], "stereo_matching": ["task2/stereo_check.json"],
        "taskA": ["taskA/aligner_comparison.json", "taskA/identity/selector_results.json", "taskA/farneback/selector_results.json",
                  "taskA/region/selector_results.json", t6p, "taskA/farneback/final/final_results.json", "taskA/region/final/final_results.json"],
        "taskB": ["taskB/policy_table.json", "taskB/metric_gap_A3.json", "taskB/metric_gap_A2.json", "taskB/metric_gap_FUSE.json"],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", default="work_v2/actions")
    ap.add_argument("--work", default="work_v2")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="web/data/results.json")
    ap.add_argument("--skip-extract", action="store_true", help="reuse results/web_extract/*.json as they are")
    a = ap.parse_args()
    R = a.results
    ex = os.path.join(R, "web_extract")
    os.makedirs(ex, exist_ok=True)
    if not a.skip_extract:
        extract_per_frame(R, a.actions, os.path.join(ex, "per_frame_test.json"))
        extract_stage2(a.work, os.path.join(ex, "stage2_gate.json"))
        extract_task3(R, os.path.join(ex, "task3_recovery.json"))
        extract_stereo_pairs(R, os.path.join(ex, "stereo_pairs.json"))
        print(f"stage 1: wrote {ex}/per_frame_test.json, stage2_gate.json, task3_recovery.json, stereo_pairs.json")
    out = build(R, a.out)
    print(f"stage 2: wrote {a.out}: {len(out['models'])} models, {len(out['stereo']['pairs'])} stereo pairs, "
          f"{len(out['selector']['test'])} test policies, blocks {sorted(out['_sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
