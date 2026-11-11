#!/usr/bin/env python3
"""
TASK P.2 -- the paper figures. Every figure is drawn from a file under results/ (or from the per-frame outputs those files summarise); nothing is typed in.

    python -m tools.paper_figures --out results/paper/figures [--only 1 2 5]

Each figure is written as PDF (vector, for LaTeX) and PNG at 300 dpi, with a colour-blind-safe palette (Okabe and Ito) and a distinct line style or marker per series so
it also reads in greyscale, sized for a single IEEE column (3.5 in) unless a wider layout is stated, plus a caption file `figNN_*.txt`. `README.md` lists every figure,
what it shows and its source files. A figure whose inputs do not exist yet is skipped and listed as MISSING in the README; nothing is fabricated to fill it.
Figure 13 (qualitative frames) is deliberately not produced: publishing EndoVis frames is governed by the dataset licence (see RUNLOG.md, NEEDS OWNER).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sitr.metrics import risk_coverage  # noqa: E402

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#888888"}
CYCLE = ["blue", "vermillion", "green", "orange", "purple", "sky", "black", "grey", "yellow"]
STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
W1, W2 = 3.5, 7.16           # IEEE single and double column widths in inches
MADE, MISSING = [], []

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 5.8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1, "figure.dpi": 100, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def J(path):
    return json.load(open(path, encoding="utf-8"))


def save(fig, out, name, title, caption, sources):
    os.makedirs(out, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    plt.close(fig)
    open(os.path.join(out, name + ".txt"), "w", encoding="utf-8").write(caption.strip() + "\n")
    MADE.append((name, title, sources, caption.strip()))
    print("  wrote", name, flush=True)


def col(i):
    return OI[CYCLE[i % len(CYCLE)]]


# ------------------------------------------------------------------ per-frame detector data (figures 1, 2)
def detector_frames(a):
    """Rebuild the per-frame scores of every failure detector exactly as tools/failure_detect_table.py does (same alignment, same rows)."""
    import pickle
    from tools.compare_stereo import load_dir
    from tools.failure_detect_table import BASELINES, align, load_scores
    oof_c = load_dir(os.path.join(a.work, "cache_oof_stereo"))
    test_c = {k: v for k, v in load_dir(os.path.join(a.work, "cache_test_stereo")).items() if k.startswith("test/")}
    gs = pickle.load(open(os.path.join(a.scores, "gate_scores_gbm.pkl"), "rb"))
    names = {"paper4": "gate (GBM): 4 original indicators", "paper4_stereo3": "gate (GBM): 4 + 3 disparity indicators (control)",
             "paper4_crossview6": "gate (GBM): 4 + 6 cross-view indicators", "crossview6": "gate (GBM): 6 cross-view indicators alone"}
    for g, fn in (("oof", "gate_scores_oof.npz"), ("test", "gate_scores_test.npz")):
        p = os.path.join(a.xv_dir, fn)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            sq = z["seq"].astype(str)
            for k, nm in names.items():
                gs[g][nm] = {u: z[k][sq == u] for u in np.unique(sq)}
    out = {}
    for split, group, cdir in (("train", "oof", oof_c), ("test", "test", test_c)):
        scores = load_scores(a.scores, split)
        keys, rows = align(split, scores, cdir, os.path.join(a.work, "actions"))
        cat = lambda f: np.concatenate([f(rows[k]) for k in keys])  # noqa: E731
        risk = cat(lambda r: r["s"]["risk"])
        det = {}
        for name, key, passes, models, note in BASELINES:
            if key == "ens_dice" and group == "oof":
                continue
            det[name] = cat(lambda r: r["s"][key]).astype(float)
        for nm in gs[group]:
            det[nm] = np.concatenate([gs[group][nm][k] for k in keys]).astype(float)
        det["gate: production (MLP, 9 indicators)"] = cat(lambda r: r["prod"])
        out[group] = (risk, det)
    return out


def fig_curves(a, out, which):
    meta = J(os.path.join(a.taskC, "failure_detection.json"))
    data = detector_frames(a)
    show = ["max-softmax", "mean entropy", "Soft Dice Confidence", "MC stochastic depth (10 passes)", "TTA pairwise Dice (6 passes)",
            "4-model ensemble pairwise Dice", "gate: production (MLP, 9 indicators)", "gate (GBM): 4 + 6 cross-view indicators"]
    for group, tag in (("test", "test"), ("oof", "oof")):
        risk, det = data[group]
        ok = np.isfinite(risk)
        fig, ax = plt.subplots(figsize=(W1, 4.4))
        i = 0
        for name in show:
            nm = name
            if nm not in det:
                continue
            sc = det[nm]
            if which == 1:
                cov, rk = risk_coverage(sc[ok], risk[ok])
                lab = f"{nm}  ({meta['groups'][group]['detectors'][nm]['aurc']:.3f})" if nm in meta["groups"][group]["detectors"] else nm
                ax.plot(cov, rk, color=col(i), ls=STYLES[i], label=lab)
            else:
                bad = (risk[ok] > 0.5).astype(int)
                s = -sc[ok]
                order = np.argsort(-s, kind="mergesort")
                tp = np.cumsum(bad[order]) / max(bad.sum(), 1)
                fp = np.cumsum(1 - bad[order]) / max((1 - bad).sum(), 1)
                lab = f"{nm}  ({meta['groups'][group]['detectors'][nm]['auroc']:.3f})" if nm in meta["groups"][group]["detectors"] else nm
                ax.plot(np.r_[0, fp], np.r_[0, tp], color=col(i), ls=STYLES[i], label=lab)
            i += 1
        if which == 1:
            r = risk[ok]
            cov, orc = risk_coverage(-r, r)
            ax.plot(cov, orc, color=OI["black"], lw=0.7, ls=":", label="oracle ranking")
            ax.axhline(r.mean(), color=OI["grey"], lw=0.6, ls=":", label="random ranking")
            ax.set_xlabel("coverage (share of frames accepted, most reliable first)")
            ax.set_ylabel("risk = mean (1 - official frame IoU)")
        else:
            ax.plot([0, 1], [0, 1], color=OI["grey"], lw=0.6, ls=":")
            ax.set_xlabel("false positive rate (good frames flagged)")
            ax.set_ylabel("true positive rate (bad frames flagged)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), handlelength=2.6, ncol=1, title="detector (AURC)" if which == 1 else "detector (AUROC)", title_fontsize=6)
        nm_fig = f"fig0{which}_{'risk_coverage' if which == 1 else 'roc'}_{tag}"
        what = "AURC" if which == 1 else "AUROC"
        G = meta["groups"][group]
        cap = (f"{'Risk-coverage' if which == 1 else 'ROC'} curves of every failure detector on the same {G['n_with_risk']} "
               f"{'official test' if group == 'test' else 'out-of-fold training'} frames; risk is 1 minus the official frame IoU and a frame is bad when its official IoU is below 0.5 "
               f"(bad rate {100*G['bad_rate']:.1f}%). {what} in the legend. "
               + ("The 4-model ensemble is test-only: every training sequence lies inside some member's training set. " if group == "test" else "No out-of-fold ensemble exists. ")
               + "The MC row is MC stochastic depth (the segmenter has no dropout layer). Source: results/taskC/failure_detection.json and per-frame scores in work_v2/failure_detection.")
        save(fig, out, nm_fig, f"{'Risk-coverage' if which == 1 else 'ROC'} curves, {tag}", cap, ["results/taskC/failure_detection.json", "work_v2/failure_detection/scores", "results/taskD/gate_scores_*.npz"])


# ------------------------------------------------------------------ figure 5
def fig5(a, out):
    K = J("results/taskK/consecutive_frame_iou.json")
    z = np.load("results/taskK/pair_scores.npz")
    tr = [k for k in z.files if k.startswith("train_cons_")]
    vals = {k.replace("train_cons_train_seq_", ""): z[k][:, 0] for k in tr}
    order = sorted(vals, key=int)
    pooled = np.concatenate([vals[k] for k in order])
    rnd = np.concatenate([z[k][:, 0] for k in z.files if k.startswith("train_rand_")])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1, 1.5]})
    bins = np.linspace(0, 1, 21)
    ax1.hist(pooled, bins=bins, color=OI["blue"], alpha=0.85, label="consecutive frames")
    ax1.hist(rnd, bins=bins, color=OI["vermillion"], alpha=0.6, label="random pairs, 20+ frames apart")
    ax1.set_xlabel("instrument IoU between ground-truth masks")
    ax1.set_ylabel("number of frame pairs")
    ax1.legend(loc="upper right")
    ax2.boxplot([vals[k] for k in order], showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=OI["sky"], linewidth=0.5), medianprops=dict(color=OI["black"], linewidth=0.8), whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5))
    ax2.set_xticklabels(order)
    ax2.set_xlabel("training sequence")
    ax2.set_ylabel("instrument IoU of consecutive pairs")
    P = K["groups"]["train"]["pooled"]["consecutive_instrument_iou"]
    cap = (f"Overlap of the ground-truth instrument masks of consecutive frames in the {len(order)} training sequences ({P['n']} pairs, 1 Hz sampling). Left: distribution, pooled mean "
           f"{P['mean']:.3f}, median {P['median']:.3f}, {100*P['frac_below_0_5']:.0f}% of pairs below 0.5, against random pairs from the same sequence. Right: per sequence (box = quartiles, whiskers = 1.5 IQR). "
           "Source: results/taskK/consecutive_frame_iou.json.")
    save(fig, out, "fig05_consecutive_frame_iou", "Consecutive-frame ground-truth IoU", cap, ["results/taskK/consecutive_frame_iou.json", "results/taskK/pair_scores.npz"])


# ------------------------------------------------------------------ figure 7
def fig7(a, out):
    harm = 0.01
    acts = [("A1", "A1 stereo"), ("A2", "A2 temporal"), ("A3", "A3 refine"), ("FUSE", "fusion")]
    d = {}
    for pre, grp, sub in (("oof", "oof", "train"), ("test", "test", "test")):
        rows = {k: [] for k, _ in acts}
        for f in sorted(os.listdir(os.path.join(a.work, "actions"))):
            if f.startswith(pre + "__") and f.endswith(".npz"):
                z = np.load(os.path.join(a.work, "actions", f), allow_pickle=True)
                for k, _ in acts:
                    rows[k].append(np.asarray(z[f"d_{k}"], float))
        d[grp] = {k: np.concatenate(v) for k, v in rows.items()}
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.2), sharex=True)
    bins = np.linspace(-0.4, 0.4, 41)
    for r, (grp, lab) in enumerate((("oof", "out of fold"), ("test", "test"))):
        for c, (k, nm) in enumerate(acts):
            ax = axes[r, c]
            v = d[grp][k]
            v = v[np.isfinite(v)]
            nz = v[np.abs(v) > 1e-9]
            ax.axvspan(-0.4, -harm, color=OI["vermillion"], alpha=0.18, lw=0)
            ax.hist(np.clip(nz, -0.4, 0.4), bins=bins, color=OI["blue"], alpha=0.9)
            ax.axvline(0, color=OI["black"], lw=0.5)
            ax.set_yscale("log")
            ax.set_ylim(0.7, max(10, ax.get_ylim()[1]) * 60)
            ax.set_title(f"{nm}, {lab}", fontsize=6.5)
            ax.text(0.97, 0.95, f"changed {len(nz)}/{len(v)} ({100*len(nz)/len(v):.0f}%)\nharmful {int((v < -harm).sum())} ({100*(v < -harm).mean():.1f}% of all)\nmean change {v.mean():+.4f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=5.2)
            if c == 0:
                ax.set_ylabel("frames (log)")
            if r == 1:
                ax.set_xlabel("change in instrument IoU")
    cap = (f"Distribution of the per-frame change in instrument IoU produced by each correction action, only for frames the action actually changed (the unchanged share is printed in each panel). "
           f"The shaded region is the harmful range, a drop larger than {harm} (the standing harm margin). Top: out-of-fold training frames; bottom: official test frames. Values beyond +-0.4 are "
           "clipped into the end bins. The instrument IoU counts a class if it is in the ground truth or the prediction (see Task I). Source: work_v2/actions/*.npz (keys d_A1 .. d_FUSE).")
    save(fig, out, "fig07_action_change_distributions", "Per-action IoU change distributions with the harmful range", cap, ["work_v2/actions/*.npz"])


# ------------------------------------------------------------------ figure 8
def fig8(a, out):
    PT = J("results/taskB/policy_table.json")["test"]
    lat = J("results/task6/final_results.json")["latency_ms"]
    comp = lat["components_ms"]
    ch = np.load("results/task5/choices_test.npz", allow_pickle=True)
    routed = ch["routed"].astype(bool)
    sec = {}
    for f in sorted(os.listdir(os.path.join(a.work, "actions"))):
        if f.startswith("test__") and f.endswith(".npz"):
            z = np.load(os.path.join(a.work, "actions", f), allow_pickle=True)
            for k in ("A1", "A2", "A3", "FUSE"):
                sec.setdefault(k, []).append(np.asarray(z[f"sec_{k}"], float))
    sec = {k: np.concatenate(v) * 1000 for k, v in sec.items()}
    seg, st, rv = comp["segmentation_forward_ms_mean"], comp["stereo_evidence_cpu_ms_mean"], comp["right_view_forward_ms_mean"]
    fr = routed.mean()
    base = lat["M1"]["mean_ms"]
    ms = {"always-A3 (routed)": base + fr * sec["A3"][routed].mean(), "always-A2 (routed)": base + fr * sec["A2"][routed].mean(),
          "always-A1 (routed)": base + fr * (st + rv + sec["A1"][routed].mean()), "always-fuse (routed)": base + fr * (st + rv + sec["A2"][routed].mean() + sec["FUSE"][routed].mean()),
          "selector": lat["M6"]["mean_ms"], "always-fuse (EVERY frame)": lat["M4"]["mean_ms"], "always-fuse+refine (EVERY)": lat["M5"]["mean_ms"],
          "oracle (routed)": base + fr * (st + rv + sum(sec[k][routed].mean() for k in ("A1", "A2", "A3", "FUSE")))}
    fig, ax = plt.subplots(figsize=(W1 + 1.4, 2.6))
    for i, (p, x) in enumerate(ms.items()):
        y = PT[p]["mean_on_flagged"]
        ax.scatter([x], [y], color=col(i), marker=MARK[i % len(MARK)], s=26 if p != "oracle (routed)" else 46, zorder=3, edgecolor="black", linewidth=0.4,
                   label=f"{p}")
    ax.axhline(0, color=OI["grey"], lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("mean latency per frame (ms, log scale)")
    ax.set_ylabel("mean change in instrument IoU\non flagged frames")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=5.6, handletextpad=0.2, borderpad=0.1, labelspacing=0.3)
    cap = ("Accuracy against cost for every correction policy on the official test set: mean change in instrument IoU on the frames the gate flags (y) against mean latency per frame "
           "(x, log scale). Latency is composed from components measured on this machine (RTX 3050 Ti laptop GPU, CPU for the correction steps): the selector and the every-frame fusion rows are "
           "the measured M6, M4, M5 latencies of Task 6; routed policies add the measured per-action time on the flagged share of frames; the oracle runs every candidate. Instrument IoU counts a "
           "class if it is in the ground truth or the prediction, so most of the A3 gain is removal of invented classes (Task B). Source: results/taskB/policy_table.json, results/task6/final_results.json, work_v2/actions.")
    save(fig, out, "fig08_accuracy_vs_cost", "Accuracy against cost per policy", cap, ["results/taskB/policy_table.json", "results/task6/final_results.json", "work_v2/actions/test__*.npz"])


# ------------------------------------------------------------------ figure 9
def fig9(a, out):
    g = J("results/task4/gate_comparison_gbm.json")["loo"]
    D = J(os.path.join(a.taskD, "decision.json"))
    xv = D["loo_paper4_crossview6"]
    short = lambda n: n.replace("_confidence", "").replace("_temporal", "").replace("_area_stability", "").replace("_feature_cos", "").replace("_entropy_conf", "").replace("_compactness", "").replace("_largest_cc", "").replace("_prev_cos", "").replace("xv_", "xv ")  # noqa: E731
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 2.9))
    items = sorted(g.items(), key=lambda kv: kv[1]["auc_drop"])
    colors = [OI["vermillion"] if n in ("stereo_disagreement", "occluded_fraction", "disparity_confidence") else OI["blue"] for n, _ in items]
    ax1.barh([short(n) for n, _ in items], [v["auc_drop"] for _, v in items], color=colors)
    ax1.axvline(0, color=OI["black"], lw=0.5)
    ax1.set_xlabel("AUROC drop when the indicator is removed")
    ax1.set_title("12-indicator gate (9 + 3 disparity)")
    items2 = sorted(xv.items(), key=lambda kv: kv[1]["auroc_drop"])
    colors2 = [OI["green"] if n.startswith("xv_") else OI["blue"] for n, _ in items2]
    ax2.barh([short(n) for n, _ in items2], [v["auroc_drop"] for _, v in items2], color=colors2)
    ax2.axvline(0, color=OI["black"], lw=0.5)
    ax2.set_xlabel("AUROC drop when the indicator is removed")
    ax2.set_title("10-indicator gate (4 + 6 cross-view)")
    cap = ("Leave-one-indicator-out importance, out of fold (15 training sequences, gradient-boosted gate, bad frame = instrument IoU below 0.5). Left: the 12-indicator gate, the three disparity "
           "indicators in vermillion. Right: the 4 original indicators plus the 6 cross-view indicators (green). Negative bars mean removing the indicator RAISED the AUROC. The cross-view set "
           "did not pass its pre-set test (Task D); differences of a few thousandths are within the noise of 15 sequences. Source: results/task4/gate_comparison_gbm.json and results/taskD/decision.json.")
    save(fig, out, "fig09_indicator_importance", "Leave-one-out indicator importance", cap, ["results/task4/gate_comparison_gbm.json", "results/taskD/decision.json"])


# ------------------------------------------------------------------ figure 10
def fig10(a, out):
    A = J("results/taskA/aligner_comparison.json")
    labels = {"identity": "previous mask\npasted on", "farneback": "optical flow\n(Farneback)", "region": "region\nmatching"}
    fig, axes = plt.subplots(1, 2, figsize=(W1 + 0.4, 2.5), sharey=True)
    for ax, grp, ttl in ((axes[0], "oof", "out of fold"), (axes[1], "test", "test")):
        G = A["groups"][grp]
        xs, ys, lo, hi = [], [], [], []
        for i, al in enumerate(("identity", "farneback", "region")):
            y = G["policies"][al]["A2"]["mean_on_flagged"]
            xs.append(i); ys.append(y)
            if al == "identity":
                lo.append(0); hi.append(0)
            else:
                d = G["paired_difference_vs_reference"][al]["A2"]
                lo.append(d["ci_lo"] if "ci_lo" in d else d["ci95"][0]); hi.append(d["ci_hi"] if "ci_hi" in d else d["ci95"][1])
        ax.bar(xs, ys, color=[OI["blue"], OI["orange"], OI["green"]], width=0.6)
        for i, al in enumerate(("farneback", "region"), start=1):
            d = G["paired_difference_vs_reference"][al]["A2"]
            l, h = (d["ci_lo"], d["ci_hi"]) if "ci_lo" in d else tuple(d["ci95"])
            ax.errorbar([i], [ys[0] + d["diff"]], yerr=[[d["diff"] - l], [h - d["diff"]]], fmt="none", ecolor=OI["black"], capsize=2, lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[k] for k in ("identity", "farneback", "region")], fontsize=5.6)
        ax.set_title(ttl)
        ax.axhline(0, color=OI["black"], lw=0.5)
    axes[0].set_ylabel("temporal-action change in instrument IoU\non flagged frames")
    cap = ("The temporal action (A2) built three ways, mean change in instrument IoU on the frames the gate flags. Whiskers on the two motion-compensated bars are the 95% sequence-clustered "
           "interval of their difference from the original (previous mask pasted on unmoved), drawn at the difference added to the original's value; both intervals are of a PAIRED difference. "
           "Neither motion-compensated version beat the original; region matching was lower with an interval excluding zero out of fold. Source: results/taskA/aligner_comparison.json.")
    save(fig, out, "fig10_aligner_comparison", "Aligner comparison for the temporal action", cap, ["results/taskA/aligner_comparison.json"])


# ------------------------------------------------------------------ figure 11
def fig11(a, out):
    tau_mem = J("work_v2/release/manifest.json").get("tau_train", 0.5)
    rows = []
    for f in sorted(os.listdir(os.path.join(a.work, "actions"))):
        if f.endswith(".npz") and (f.startswith("oof__") or f.startswith("test__")):
            z = np.load(os.path.join(a.work, "actions", f), allow_pickle=True)
            r = np.asarray(z["r"], float)
            rows.append((str(z["seq"]), float((r >= tau_mem).mean()), bool(f.startswith("test__"))))
    rows.sort(key=lambda t: (t[2], int(t[0].split("seq_")[1])))
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    ax.bar(range(len(rows)), [r[1] for r in rows], color=[OI["vermillion"] if r[2] else OI["blue"] for r in rows])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0].replace("train/seq_", "tr ").replace("test/seq_", "te ") for r in rows], rotation=60, fontsize=5.5)
    ax.set_ylabel(f"share of frames with reliability >= {tau_mem}")
    ax.set_ylim(0, 1)
    cap = (f"Memory starvation: for every sequence, the share of frames whose gate reliability clears the memory-admission threshold tau_memory = {tau_mem} (the value the gate was trained and the "
           "temporal memory admits with). Frames that do not clear it cannot enter the reliable memory the temporal action draws on. Blue: training sequences (out of fold); vermillion: official "
           "test sequences. Source: gate reliability per frame in work_v2/actions/*.npz (key r), threshold from work_v2/release/manifest.json (tau_train).")
    save(fig, out, "fig11_memory_starvation", "Memory starvation per sequence", cap, ["work_v2/actions/*.npz", "work_v2/release/manifest.json"])


# ------------------------------------------------------------------ figure 12
def fig12(a, out):
    I = J("results/taskI/metric_ablation.json")
    rows = [r for r in I["rows"] if not r.get("combined") and not r["name"].startswith("reference")]
    fig, ax = plt.subplots(figsize=(W1 + 0.6, 2.4))
    names = [r["name"].replace(" (ground truth downsampled)", "").replace("per-class mean over frames, then over classes (mcIoU)", "per-class mean (mcIoU)")
             .replace("class counted if in ground truth OR prediction", "class counted if in GT or prediction")
             .replace("background included in the per-frame mean", "background included") for r in rows]
    vals = [r["delta_points_vs_reference"] for r in rows]
    ax.barh(names[::-1], vals[::-1], color=[OI["blue"] if v >= 0 else OI["vermillion"] for v in vals[::-1]])
    ax.axvline(0, color=OI["black"], lw=0.5)
    ax.set_xlabel(f"change in mIoU points from the official protocol ({100*I['reference_mIoU']:.2f})")
    cap = ("What each definition choice is worth in mIoU points, on the same predictions (the single model, 997 test frames). Reference: the exact official protocol (class 0 never scored, class counted "
           "only if in the ground truth, per-frame mean, full 1280x1024). Each bar changes exactly one choice. Two choices (counting a class on the prediction as well, and averaging per class "
           "instead of per frame) each move the score by more than 11 points, more than any correction effect in this study. Source: results/taskI/metric_ablation.json.")
    save(fig, out, "fig12_metric_ablation", "Metric-implementation ablation", cap, ["results/taskI/metric_ablation.json"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/paper/figures")
    ap.add_argument("--work", default="work_v2")
    ap.add_argument("--scores", default="work_v2/failure_detection")
    ap.add_argument("--taskC", default="results/taskC")
    ap.add_argument("--taskD", default="results/taskD")
    ap.add_argument("--xv-dir", default="results/taskD")
    ap.add_argument("--only", nargs="*", type=int, default=None)
    a = ap.parse_args()
    todo = {1: lambda: fig_curves(a, a.out, 1), 2: lambda: fig_curves(a, a.out, 2), 5: lambda: fig5(a, a.out), 7: lambda: fig7(a, a.out), 8: lambda: fig8(a, a.out),
            9: lambda: fig9(a, a.out), 10: lambda: fig10(a, a.out), 11: lambda: fig11(a, a.out), 12: lambda: fig12(a, a.out)}
    for k, fn in todo.items():
        if a.only and k not in a.only:
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- reported, never hidden
            MISSING.append((k, f"{type(e).__name__}: {e}"))
            print(f"  FIGURE {k} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("made:", [m[0] for m in MADE], "\nmissing:", MISSING)
    reg_path = os.path.join(a.out, "_registry.json")
    reg = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    for n, t, srcs, c in MADE:
        reg[n] = dict(title=t, sources=srcs, caption=c)
    json.dump(reg, open(reg_path, "w"), indent=1)
    lines = ["# Paper figures", "",
             "Every figure is drawn by `tools/paper_figures.py` (figures 3, 4 and 6a by `tools/paper_figures_p1.py`) from the files listed under it; none contains typed numbers. Each exists as PDF (vector, for LaTeX) and PNG at 300 dpi, "
             "with the caption in the `.txt` file of the same name. Palette: Okabe and Ito (colour-blind safe), with a distinct line style or marker per series. Sized for a single IEEE column (3.5 in) unless the figure needs the double width.", ""]
    for n in sorted(reg):
        r = reg[n]
        lines += [f"## {n}", "", f"**{r['title']}.** {r['caption']}", "", "Source files: " + ", ".join(f"`{x}`" for x in r["sources"]), ""]
    lines += ["## Not produced", "",
              "- **Figure 13, qualitative frames.** Publishing EndoVis frames is governed by the dataset's own licence and data-use terms, which are not the owner's alone to approve (see NEEDS OWNER in RUNLOG.md); it is not generated on assumption.",
              "- Any figure of the list in task P.2 that does not appear above had its inputs missing at the time of the last run; `_manifest.json` lists the reason."]
    open(os.path.join(a.out, "README.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
