#!/usr/bin/env python3
"""
TASK P.5 -- LaTeX tables for the paper, generated from the result JSON files; nothing is retyped.

    python -m tools.paper_tables --out results/paper

Writes results/paper/tables/*.tex (a bare `tabular` each, no table wrapper, IEEE two-column safe: \\hline rules, no packages beyond the standard ones) and results/paper/NUMBERS.md,
which lists EVERY number placed in a table with the file and key path it was read from, so each can be checked in one step. A table whose input does not exist yet is skipped and listed.
No generalisation table is produced: extra databases were not approved (RUNLOG.md, NEEDS OWNER).
"""
from __future__ import annotations

import argparse
import json
import os

REG = []                 # (table, label, formatted value, file, key path)
SKIPPED = []


def J(p):
    return json.load(open(p, encoding="utf-8"))


def dig(d, path):
    for k in path:
        d = d[k]
    return d


def N(tab, label, d, path, src, fmt="{:.3f}", scale=1.0):
    v = dig(d, path)
    if v is None:
        return "--"
    s = fmt.format(v * scale)
    REG.append((tab, label, s, src, ".".join(str(p) for p in path)))
    return s


def tex(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def write(out, name, cols, header, rows):
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    lines = ["\\begin{tabular}{" + cols + "}", "\\hline", " & ".join(header) + " \\\\", "\\hline"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    open(os.path.join(out, "tables", name + ".tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("  wrote", name)


def t_failure(out):
    src = "results/taskC/failure_detection_cost.json"
    R = J(src)
    order = ["max-softmax", "mean entropy", "Soft Dice Confidence", "MC stochastic depth (10 passes)", "TTA entropy (6 passes)", "TTA pairwise Dice (6 passes)",
             "4-model ensemble pairwise Dice", "gate: production (MLP, 9 indicators)", "gate (GBM): 4 + 6 cross-view indicators"]
    short = {"MC stochastic depth (10 passes)": "MC stochastic depth (10)", "TTA entropy (6 passes)": "TTA entropy (6)", "TTA pairwise Dice (6 passes)": "TTA pairwise Dice (6)",
             "4-model ensemble pairwise Dice": "4-model ensemble Dice$^{\\dagger}$", "gate: production (MLP, 9 indicators)": "Gate (9 indicators)", "gate (GBM): 4 + 6 cross-view indicators": "Gate + cross-view$^{\\ddagger}$"}
    rows = []
    for n in order:
        r = [tex(short.get(n, n))]
        for g in ("oof", "test"):
            d = R["groups"][g]["detectors"]
            if n in d:
                r += [N("failure", f"{n} {g} AURC", R, ["groups", g, "detectors", n, "aurc"], src, "{:.3f}"), N("failure", f"{n} {g} AUROC", R, ["groups", g, "detectors", n, "auroc"], src, "{:.3f}")]
            else:
                r += ["--", "--"]
        d = R["groups"]["test"]["detectors"][n]
        r += [str(d["forward_passes"]), str(d["segmenters_trained"]), N("failure", f"{n} ms", R, ["groups", "test", "detectors", n, "ms_per_frame"], src, "{:.0f}")]
        rows.append(r)
    write(out, "failure_detection", "lcccccccc", ["Detector", "AURC", "AUROC", "AURC", "AUROC", "Passes", "Models", "ms"],
          [[""] + ["\\multicolumn{2}{c}{Out of fold}", "\\multicolumn{2}{c}{Test}", "", "", ""] ] + rows)


def t_policy(out):
    src = "results/taskB/policy_table.json"
    P = J(src)["test"]
    rows = []
    for n, lab in (("oracle (routed)", "Oracle$^{\\dagger}$"), ("selector", "Selector"), ("always-A3 (routed)", "Always A3"), ("always-A2 (routed)", "Always A2"), ("always-A1 (routed)", "Always A1"),
                   ("always-fuse (routed)", "Always fusion"), ("always-fuse (EVERY frame)", "Fusion, every frame")):
        rows.append([lab, N("policy", f"{n} activation", P, [n, "activation_pct_of_all"], src, "{:.1f}"), N("policy", f"{n} change flagged", P, [n, "mean_on_flagged"], src, "{:+.4f}"),
                     N("policy", f"{n} change all", P, [n, "mean_all_frames"], src, "{:+.4f}"), N("policy", f"{n} official", P, [n, "official_all_frames"], src, "{:+.4f}"),
                     N("policy", f"{n} harmful", P, [n, "harmful_pct_of_corrected"], src, "{:.1f}")])
    write(out, "policies", "lccccc", ["Policy", "Corr. \\%", "$\\Delta$ flagged", "$\\Delta$ all", "$\\Delta$ official", "Harm \\%"], rows)


def t_perclass(out):
    src = "results/taskJ/per_class_and_rigs.json"
    J_ = J(src)
    base = "results/taskP/baselines.json"
    B = J(base) if os.path.exists(base) else None
    names = [n for n in J_["test_per_class"] if J_["test_per_class"][n]["pooled_iou"] is not None]
    cols = ["Class", "Frames", "Ours"] + ([m["label"] for m in B["models"]] if B else [])
    rows = []
    for n in names:
        r = [tex(n.replace("instrument-", "").replace("kidney-parenchyma", "kidney")), str(J_["test_per_class"][n]["n_frames"]), N("perclass", f"{n} ours", J_, ["test_per_class", n, "pooled_iou"], src)]
        if B:
            for m in B["models"]:
                r.append(N("perclass", f"{n} {m['label']}", m, ["per_class", n, "iou"], base))
        rows.append(r)
    r = ["mIoU", "", N("perclass", "mIoU ours", J_, ["test_pooled", "mIoU"], src, "{:.4f}")]
    r2 = ["mcIoU", "", N("perclass", "mcIoU ours", J_, ["test_pooled", "mcIoU"], src, "{:.4f}")]
    if B:
        for m in B["models"]:
            r.append(N("perclass", f"mIoU {m['label']}", m, ["mIoU"], base, "{:.4f}"))
            r2.append(N("perclass", f"mcIoU {m['label']}", m, ["mcIoU"], base, "{:.4f}"))
    write(out, "per_class", "l" + "c" * (len(cols) - 1), cols, rows + [r, r2])


def t_ablation(out):
    src = "results/taskI/metric_ablation.json"
    I = J(src)
    rows = []
    for i, r in enumerate(I["rows"]):
        rows.append([tex(r["name"]), N("ablation", f"row {i} mIoU", I, ["rows", i, "mIoU"], src, "{:.4f}"), N("ablation", f"row {i} change", I, ["rows", i, "delta_points_vs_reference"], src, "{:+.2f}")])
    write(out, "metric_ablation", "lcc", ["Setting (one change from the official protocol)", "mIoU", "Points"], rows)


def t_baselines(out):
    src = "results/taskP/baselines.json"
    if not os.path.exists(src):
        SKIPPED.append(("baseline_comparison", "results/taskP/baselines.json does not exist yet"))
        return
    B = J(src)
    rows = []
    for i, m in enumerate(B["models"]):
        rows.append([tex(m["label"]), N("baselines", f"{m['label']} mIoU", B, ["models", i, "mIoU"], src, "{:.4f}"), N("baselines", f"{m['label']} mcIoU", B, ["models", i, "mcIoU"], src, "{:.4f}"),
                     N("baselines", f"{m['label']} bF1", B, ["models", i, "boundary_f1_instruments"], src), N("baselines", f"{m['label']} clDice shaft", B, ["models", i, "cldice", "shaft"], src),
                     N("baselines", f"{m['label']} clDice thread", B, ["models", i, "cldice", "thread"], src),
                     N("baselines", f"{m['label']} params", B, ["models", i, "params_millions"], src, "{:.1f}"), N("baselines", f"{m['label']} GFLOPs", B, ["models", i, "gflops"], src, "{:.0f}"),
                     N("baselines", f"{m['label']} ms", B, ["models", i, "ms_per_frame"], src, "{:.0f}")])
    write(out, "baseline_comparison", "lcccccccc", ["Model", "mIoU", "mcIoU", "bF1", "clD sh.", "clD thr.", "Par. (M)", "GFLOPs", "ms"], rows)


def t_gate_ablation(out):
    src = "results/taskC/gate_ablation_seeds.json"
    if not os.path.exists(src):
        SKIPPED.append(("gate_ablation", f"{src} does not exist yet"))
        return
    G = J(src)
    names = [("max-softmax baseline", "Max-softmax"), ("gate with max-softmax (9 indicators), refit seed 0", "Gate, 9 indicators"), ("gate without max-softmax (8 indicators), refit seed 0", "Gate, no max-softmax")]
    rows = []
    for lab, ak, uk in (("Official", "auroc_official", "aurc_official"), ("Instrument", "auroc_instrument", "aurc_instrument")):
        for key, nm in names:
            r = [lab if key == names[0][0] else "", nm]
            for g in ("oof", "test"):
                m = ["item1", g, "detectors", key]
                r += [N("gate_ablation", f"{nm} {lab} {g} AUROC", G, m + [ak], src), N("gate_ablation", f"{nm} {lab} {g} AURC", G, m + [uk], src)]
            rows.append(r)
        rows[-1][0] = rows[-1][0]
    write(out, "gate_ablation", "llcccc", ["Label", "Detector", "AUROC oof", "AURC oof", "AUROC test", "AURC test"], rows)


def main():
    import argparse as ap_
    ap = ap_.ArgumentParser(description=__doc__, formatter_class=ap_.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/paper")
    a = ap.parse_args()
    for fn, nm in ((t_failure, "failure_detection"), (t_policy, "policies"), (t_perclass, "per_class"), (t_ablation, "metric_ablation"), (t_baselines, "baseline_comparison"), (t_gate_ablation, "gate_ablation")):
        try:
            fn(a.out)
        except Exception as e:  # noqa: BLE001 -- reported, never hidden
            SKIPPED.append((nm, f"{type(e).__name__}: {e}"))
            print(f"  {nm} FAILED: {type(e).__name__}: {e}")
    lines = ["# Numbers in the paper tables", "",
             "Every number placed in a table by `tools/paper_tables.py`, with the file and key path it was read from (keys joined by dots; list positions are integers). Re-run the tool to regenerate; "
             "the tables and this list come from the same pass, so they cannot disagree.", "", "| table | quantity | value as printed | file | key path |", "|---|---|---|---|---|"]
    for t, lab, v, f, k in REG:
        lines.append(f"| {t} | {lab} | {v} | {f} | `{k}` |")
    if SKIPPED:
        lines += ["", "## Not produced", ""] + [f"- {n}: {why}" for n, why in SKIPPED]
    open(os.path.join(a.out, "NUMBERS.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{len(REG)} numbers registered; skipped: {SKIPPED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
