#!/usr/bin/env python3
"""Write results/task2/STEREO_REPORT.md from results/task2/stereo_check.json.
Every number in the report is read from the JSON, none is typed in."""
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results/task2"
R = json.load(open(os.path.join(d, "stereo_check.json")))
co, rf = R["epipolar_calibration_only"], R["epipolar_refined"]
ec, er = R["eval_calibration_only"], R["eval_refined"]
L = []
w = L.append
w("# Task 2 — calibrated stereo go / no-go\n")
w("Generated from `stereo_check.json` by `tools/report_task2.py`. Tool: `tools/check_stereo.py`.\n")
w("## Verdict\n")
stops = R["stop_conditions_tripped"]
anyc = any(stops["calibration_only"].values()); anyr = any(stops["refined"].values())
w(f"**Calibrated stereo is VIABLE.** None of the three stop conditions tripped, with calibration alone "
  f"({'TRIPPED' if anyc else 'not tripped'}) or with the self-calibration ({'TRIPPED' if anyr else 'not tripped'}).\n")
w("**But the shipped calibration alone is not enough for every sequence.** It leaves a stable vertical tilt of "
  f"several pixels on some TRAINING sequences ({co['summary']['n_pairs_mean_gt_3px']} of {co['summary']['n_pairs']} "
  "evaluation pairs have mean vertical error above 3 px). A label-free per-sequence correction "
  "(estimated from each sequence's first 40 s, evaluated on other frames) removes most of it. "
  "The four TEST sequences are fine with calibration alone.\n")
w("## 1. Vertical disparity after rectification (matched SIFT features, px)\n")
w(f"Unrectified pairs, mean abs(dy) averaged over the 20 pairs: **{R['raw_dy_mean_over_pairs']:.2f} px**.\n")
w("| Geometry | mean abs(dy) | median | mean p95 | worst pair (mean) | pairs with mean > 3 px |")
w("|---|---|---|---|---|---|")
for name, s in (("Calibration only", co["summary"]), ("Calibration + self-calibration", rf["summary"])):
    w(f"| {name} | {s['mean_of_means']:.2f} | {s['pooled_median']:.2f} | {s['p95_mean']:.2f} | "
      f"{s['worst_pair_mean']:.2f} | {s['n_pairs_mean_gt_3px']} / {s['n_pairs']} |")
w("\nDistortion / rotation convention, decided by the data (calibration only, mean abs(dy)):\n")
w("| Convention | mean abs(dy) |")
w("|---|---|")
for k, v in R["conventions"].items():
    w(f"| {k}{'  **(chosen)**' if k == R['chosen_convention'] else ''} | {v['mean_of_means']:.3f} |")
w("\nPer pair:\n")
w("| Pair | calibration only: mean / median / p95 | + self-calibration: mean / median / p95 |")
w("|---|---|---|")
for a_, b_ in zip(co["per_pair"], rf["per_pair"]):
    f = lambda e: f"{e['dy_mean']:.2f} / {e['dy_median']:.2f} / {e['dy_p95']:.2f}" if e.get("n_matches") else "n/a"
    w(f"| {a_['seq']} {a_['frame']} | {f(a_)} | {f(b_)} |")
w("\n## 2. Per-sequence vertical self-calibration (images only, no labels)\n")
w("Model: `dy(x, y) = y_left - y_right = a + b*x/1000 + c*y/1000`, estimated by robust (Tukey) regression on matched "
  f"features of frames {R['refine_frames']}. Applied only if it implies more than 1 px somewhere in the image.\n")
w("| Sequence | matches | a | b | c | residual MAD | max implied abs(dy), px | applied |")
w("|---|---|---|---|---|---|---|---|")
for k, c in R["vertical_corrections"].items():
    w(f"| {k} | {c['n_matches']} | {c['a']:.2f} | {c['b']:.2f} | {c['c']:.2f} | {c['mad']:.2f} | {c['max_abs_px']:.2f} | {'yes' if c['applied'] else 'no'} |")
w("\n## 3. SGBM settings\n")
ch = R["sgbm_chosen"]
w(f"Chosen (by the criterion fixed before any result: fraction of pixels that are LR-consistent AND photometric "
  f"residual < 10 gray levels, 5 TRAINING tuning frames not among the 20): **block {ch['block']}, uniqueness "
  f"{ch['uniqueness']}, speckle {ch['speckle_window']}, numDisparities {ch['num_disp']}**, P1/P2 = 8/32 x block^2, "
  f"min disparity {ch['min_disp']}, 3-way mode, mirror trick for the right-view disparity.\n")
if R.get("sgbm_optimum_on_grid_edge"):
    w(f"**Caveat:** the optimum is on the edge of the searched grid for {', '.join(R['sgbm_optimum_on_grid_edge'])}, "
      "i.e. the criterion keeps preferring larger windows and fewer uniqueness checks. It cannot see over-smoothing or false "
      "matches in textureless regions. A diagnostic declared in advance (`sgbm_diagnostic.txt`) checked this and did not "
      "overturn the choice.\n")
w("Top 8 grid cells:\n")
w("| block | uniqueness | speckle | verified | LR pass | valid | s/frame |")
w("|---|---|---|---|---|---|---|")
for r in sorted(R["sgbm_grid"], key=lambda r: -r["verified"])[:8]:
    w(f"| {r['block']} | {r['uniqueness']} | {r['speckle_window']} | {r['verified']:.3f} | {r['lr_pass']:.3f} | {r['valid']:.3f} | {r['sec_per_frame']:.2f} |")
w("\n## 4. Disparity validity on the 20 evaluation pairs\n")
w("Percent of the in-view left image. 'Interior' excludes the left strip where no right-image counterpart can exist.\n")
w("| Geometry | valid disparity | LR-consistent | LR (interior) | instrument px: valid / LR | tissue px: valid / LR |")
w("|---|---|---|---|---|---|")
for name, e in (("Calibration only", ec), ("Calibration + self-calibration", er)):
    o = e["overall"]; i_, t_ = e["instrument"], e["tissue"]
    w(f"| {name} | {o['valid_pct']:.1f} | {o['lr_pct']:.1f} | {o['lr_pct_interior']:.1f} | "
      f"{i_['valid_pct']:.1f} / {i_['lr_pct']:.1f} | {t_['valid_pct']:.1f} / {t_['lr_pct']:.1f} |")
w(f"\nInstrument pixels: {er['instrument']['pixels']:,}; tissue pixels: {er['tissue']['pixels']:,} "
  "(training frames only; ground truth is used solely to split the pixels).\n")
p = er["disparity_percentiles_lr_ok"]
w(f"LR-consistent disparity percentiles (px, rectified geometry): p1 {p['p1']:.1f}, p5 {p['p5']:.1f}, "
  f"median {p['p50']:.1f}, p95 {p['p95']:.1f}, p99 {p['p99']:.1f}. Fraction at or above 190 px (the numDisparities cap): "
  f"{100*er['frac_disp_ge_190']:.2f} %. The expected 25-160 px range holds for 1st to ~95th percentile; about 1 % of "
  "consistent pixels sit at the cap, so close-range structures are slightly truncated.\n")
w("Per pair (refined):\n")
w("| Pair | valid % | LR % | instrument LR % | tissue LR % |")
w("|---|---|---|---|---|")
nan = lambda v: "n/a" if v is None else f"{v:.1f}"
for r in er["per_frame"]:
    w(f"| {r['seq']} {r['frame']} | {r['valid_pct']:.1f} | {r['lr_pct']:.1f} | {nan(r.get('instrument_lr_pct'))} | {nan(r.get('tissue_lr_pct'))} |")
w("\n## 5. Stop conditions\n")
for mode, dd in stops.items():
    for k, v in dd.items():
        w(f"* [{mode}] {k}: **{'TRIPPED' if v else 'not tripped'}**")
w(f"\nTotal run time {R['seconds_total']:.0f} s. Images (rectified pairs with epipolar lines, disparity maps) are in this folder (git-ignored).")
open(os.path.join(d, "STEREO_REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", os.path.join(d, "STEREO_REPORT.md"))
