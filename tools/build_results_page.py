#!/usr/bin/env python3
"""
TASK 7B -- one self-contained HTML results page (no server, no tracking, no network).

    python -m tools.build_results_page --results results --out results/report.html

Reads ONLY files the earlier tools wrote (results/task2..task6). A missing file
shows "not produced"; no number is ever typed in here. Figures are inlined as
base64 PNGs. Dataset frames (patient-derived) are NOT included unless you pass
--include-gallery after approving specific images.
"""
import argparse
import base64
import html
import json
import os

CSS = """:root{--bg:#fff;--fg:#1c1c1c;--mut:#666;--line:#ddd;--acc:#1b6ca8;--bad:#b3261e;--good:#1b7f4b}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#aaa;--line:#333;--acc:#6db3e8;--bad:#f2837a;--good:#6fd39a}}
body{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;margin:0;padding:0 16px 64px}
main{max-width:980px;margin:0 auto}h1{font-size:1.7rem;margin:32px 0 4px}h2{margin-top:40px;border-bottom:1px solid var(--line);padding-bottom:6px}
.mut{color:var(--mut)}table{border-collapse:collapse;width:100%;margin:12px 0;font-size:.92rem;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.box{border:1px solid var(--line);border-left:4px solid var(--acc);padding:10px 14px;margin:14px 0;border-radius:4px}
.bad{color:var(--bad)}.good{color:var(--good)}img{max-width:100%}"""


def load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def esc(x):
    return html.escape(str(x))


def f(x, n=3):
    return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else esc(x))


def table(head, rows):
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def img(path, alt):
    if not os.path.exists(path):
        return f"<p class='mut'>{esc(alt)}: not produced</p>"
    with open(path, "rb") as fh:
        return f"<p><img alt='{esc(alt)}' src='data:image/png;base64,{base64.b64encode(fh.read()).decode()}'></p>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/report.html")
    ap.add_argument("--include-gallery", action="store_true", help="inline best/worst panels (patient-derived frames!)")
    a = ap.parse_args()
    r = a.results
    t2 = load(f"{r}/task2/stereo_check.json"); t3 = load(f"{r}/task3/measure_recovery.json")
    t4g = load(f"{r}/task4/gate_comparison_gbm.json"); t4t = load(f"{r}/task4/gate_comparison_test.json")
    t4d = load(f"{r}/task4/difficulty_report.json"); t5 = load(f"{r}/task5/selector_results.json")
    t6 = load(f"{r}/task6/final_results.json")
    P = []
    P.append("<h1>Stereo-gated selective correction, EndoVis 2018</h1>")
    P.append("<p class='mut'>Every number below was read from a result file written by the project's tools. "
             "Numbers labelled <b>official test</b> are on the 4 official test sequences (997 labelled frames, full 1280x1024). "
             "Our mIoU is <b>not</b> comparable to published numbers (different hardware, protocol, reproducibility).</p>")

    # ---- plain-language summary, derived from the numbers
    S = []
    if t2:
        e0, e1 = t2["epipolar_calibration_only"]["summary"], t2["epipolar_refined"]["summary"]
        S.append(f"Calibrated stereo is usable: vertical misalignment {f(e0['mean_of_means'],2)} px with the shipped calibration, "
                 f"{f(e1['mean_of_means'],2)} px after a label-free per-sequence correction.")
    if t4g:
        d = t4g["cv_paired"].get("extended_stereo vs extended")
        if d:
            S.append(f"Stereo indicators vs temporal ones for failure detection: {esc(d['verdict'])} "
                     f"(AUC difference {f(d['auc']['mean'],3)}, 95% interval {f(d['auc']['ci95'][0],3)} to {f(d['auc']['ci95'][1],3)}).")
    if t4d:
        o = t4d.get("oof_train_sequences", {}).get("vs_instrument_error")
        if o:
            S.append(f"The difficulty index on held-out training frames: Spearman {f(o['spearman'],3)} with segmentation error.")
    if t6:
        h = t6["headline_M4_vs_M6"]
        S.append(f"Harmful corrections (share of all test frames made worse): blind fusion M4 {f(h['harmful_pct_of_all'][0],1)} %, "
                 f"selective M6 {f(h['harmful_pct_of_all'][1],1)} %. Frames corrected: {f(h['activation_pct'][0],1)} % vs {f(h['activation_pct'][1],1)} %. "
                 f"Official mIoU {f(h['mIoU'][0],4)} vs {f(h['mIoU'][1],4)}.")
    P.append("<h2>What was measured, in plain language</h2><div class='box'>" +
             ("<ul>" + "".join(f"<li>{s}</li>" for s in S) + "</ul>" if S else "<p>No results available yet.</p>") + "</div>")

    if t6:
        P.append("<h2>Ablation M1 to M6 (official test)</h2>")
        rows = []
        for M, m in t6["models"].items():
            rows.append([M, esc(t6["descriptions"][M]), f(m["mIoU"], 4), f(m["boundary_f1_instruments"], 3),
                         f"{f(m['cldice']['shaft'],2)}/{f(m['cldice']['thread'],2)}/{f(m['cldice']['needle'],2)}",
                         f(m["activation_pct"], 1), f(m["harmful_pct_of_all"], 1), f(m["harmful_pct_of_corrected"], 1), f(m["aurc_gate"], 4)])
        P.append(table(["", "model", "official mIoU", "boundary F1", "clDice shaft/thread/needle", "corrected %", "harmful % of all", "harmful % of corrected", "AURC"], rows))
        lat = t6.get("latency_ms", {})
        if "M1" in lat:
            P.append(f"<p class='mut'>Latency measured on {esc(lat.get('gpu'))} (components timed on {lat['frames_timed']} frames): " +
                     ", ".join(f"{M} {f(lat[M]['mean_ms'],0)} ms mean / {f(lat[M]['p95_ms'],0)} ms P95" for M in ("M1", "M4", "M6") if M in lat) + ".</p>")
        cc = t6.get("crosscheck_official_run_py")
        if cc:
            P.append(f"<p class='mut'>Evaluator cross-check against the organisers' own run.py: official {f(cc['official_run_py_mean_iou'],6)}, ours {f(cc['our_M1_mIoU'],6)}.</p>")
        P.append(img(f"{r}/task6/fig_risk_coverage.png", "risk-coverage curves, M1/M4/M6"))
        P.append(img(f"{r}/task6/fig_by_difficulty.png", "performance by difficulty quartile"))
        P.append("<h3>Per-class IoU (official test)</h3>")
        cls = list(t6["models"]["M1"]["per_class"].keys())
        P.append(table(["class", "frames"] + list(t6["models"]), [[c, t6["models"]["M1"]["per_class"][c]["n_frames"]] +
                  [f(t6["models"][M]["per_class"][c]["iou"], 3) for M in t6["models"]] for c in cls]))
    if t3:
        P.append("<h2>Does temporal correction help? (Task 3)</h2>")
        rows = []
        for mode in ("selective", "always"):
            s = t3[mode]["instrument_iou"]
            if s.get("n"):
                rows.append([mode, s["n"], f(s["mean_before"]), f(s["mean_after"]), f(s["mean_delta"], 4),
                             f"{f(s['ci95'][0],4)} to {f(s['ci95'][1],4)}", f(s["pct_improved"], 1), f(s["pct_worse"], 1)])
        P.append(table(["mode", "frames", "IoU before", "IoU after", "mean change", "95% CI", "improved %", "worse %"], rows))
    if t4g:
        P.append("<h2>Failure detection: stereo vs temporal indicators (Task 4)</h2>")
        rows = [[k, f(v["auc"]), f(v["aurc"], 4), f(v["pearson"]), f(v["mse"], 4)] for k, v in t4g["cv"].items()]
        P.append(table(["indicator set", "AUC", "AURC (lower better)", "Pearson", "MSE"], rows))
        P.append("<p class='mut'>Held-out training sequences, cross-fitted. Differences tested by a paired bootstrap over sequences.</p>")
        rows = [[k, f(v["auc"]["mean"], 3), f"{f(v['auc']['ci95'][0],3)} to {f(v['auc']['ci95'][1],3)}", esc(v["verdict"])] for k, v in t4g["cv_paired"].items()]
        P.append(table(["comparison", "AUC diff", "95% interval", "verdict"], rows))
        if t4t:
            P.append("<p class='mut'>Official test sequences:</p>" + table(["indicator set", "AUC", "AURC", "Pearson"],
                     [[k, f(v["auc"]), f(v["aurc"], 4), f(v["pearson"])] for k, v in t4t["sets"].items()]))
    if t5:
        P.append("<h2>Selector (Task 5)</h2>")
        for tag, title in (("oof", "held-out training sequences"), ("test", "official test")):
            if tag in t5:
                rows = [[k, f(v["activation_pct_of_all"], 1), f(v["mean_delta_all_frames"], 4), f(v["harmful_rate_of_corrected"], 1),
                         f(v["mean_correction_cpu_ms_per_frame"], 1)] for k, v in t5[tag].items()]
                P.append(f"<h3>{esc(title)}</h3>" + table(["policy", "corrected %", "mean IoU change (all frames)", "harmful % of corrected", "correction CPU ms"], rows))
    if t2:
        P.append("<h2>Stereo geometry (Task 2)</h2>")
        o, rr = t2["eval_calibration_only"], t2["eval_refined"]
        P.append(table(["geometry", "valid disparity %", "LR-consistent %", "instrument px LR %", "tissue px LR %"],
                       [["calibration only", f(o["overall"]["valid_pct"], 1), f(o["overall"]["lr_pct"], 1), f(o["instrument"]["lr_pct"], 1), f(o["tissue"]["lr_pct"], 1)],
                        ["+ self-calibration", f(rr["overall"]["valid_pct"], 1), f(rr["overall"]["lr_pct"], 1), f(rr["instrument"]["lr_pct"], 1), f(rr["tissue"]["lr_pct"], 1)]]))
    if a.include_gallery:
        P.append("<h2>Best saves and worst damages</h2>")
        for sub in ("best", "worst"):
            d = f"{r}/task3/{sub}"
            if os.path.isdir(d):
                for fn in sorted(os.listdir(d))[:6]:
                    P.append(img(os.path.join(d, fn), fn))
    else:
        P.append("<h2>Qualitative gallery</h2><p class='mut'>Not included: the frames are patient-derived. Re-run with --include-gallery after approving specific images.</p>")
    P.append("<h2>Limitations</h2><ul><li>Identity verification cannot be supervised on semantic masks; it is a consistency check only.</li>"
             "<li>Our mIoU is not comparable to published numbers (training budget, hardware, protocol).</li>"
             "<li>The test set has no suction-instrument frames, and needle, thread and clamps appear in one test sequence only, so their per-class scores are high-variance.</li>"
             "<li>Temporal memory holds uncorrected predictions when generating action data; a deployed system would store corrected masks.</li></ul>")
    doc = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Stereo-gated correction</title><style>{CSS}</style></head><body><main>{''.join(P)}</main></body></html>"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {a.out} ({len(doc)/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
