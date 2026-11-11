#!/usr/bin/env python3
"""
TASK C, stage 2 -- the failure-detection baseline table (CPU only).

    python -m tools.failure_detect_table --work work_v2 --scores work_v2/failure_detection --out results/taskC \
        [--xv-oof work_v2/cache_oof_xv --xv-test work_v2/cache_test_xv]     (added after Task D)

Same frames and the same risk for every detector:  risk = 1 - official frame IoU of the single-pass mask at
1280x1024 (a class counts only if it is in the ground truth).  A frame is BAD when official IoU < 0.5
(0.5 is the repo's standing bad-frame threshold, tools/final_eval.py; it is not tuned here).

Per detector:  AUROC for ranking bad frames (a lower reliability score should mean bad), AURC of the risk-coverage
curve (lower is better; floor = oracle ranking, ceiling = random ranking), both with 95% intervals from a
bootstrap that resamples whole SEQUENCES (2000 draws, seed 0; 15 clusters out of fold, 4 on test, so the test
interval is coarse and is a warning, not a test).  Paired differences to the production gate are given the same way.

Rows
  baselines  : max-softmax, mean entropy, Soft Dice Confidence, MC stochastic depth (10), TTA entropy (6),
               TTA pairwise Dice (6), 4-model ensemble pairwise Dice (TEST ONLY).
  our gate   : confidence only (5 confidence/shape indicators, no temporal), + temporal (the 9 indicators), both
               gradient-boosted, cross-fitted with work_v2/folds.json (same machinery as tools/compare_stereo.py);
               and the PRODUCTION gate (the released MLP, K=3) whose per-frame scores are in work_v2/actions.
               The gate regresses INSTRUMENT IoU, the risk here is the 12-class official IoU: that mismatch is a
               property of the system and is reported, not corrected.
Secondary column: AUROC against the instrument-IoU < 0.5 label, which is the label the earlier 0.84 was computed
against, so the two can be read side by side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.metrics import aurc, aurc_oracle, roc_auc  # noqa: E402
from tools.compare_stereo import cv_run, load_dir, test_run  # noqa: E402

THETA = 0.5
N_BOOT = 2000
ALIGN_TOL = 0.005         # mean absolute difference; a stated choice, see align()
ALIGN_CORR = 0.99
ALIGN_LOG = {}

# name, score file key, forward passes, segmenters trained, note
BASELINES = [
    ("max-softmax", "ms_fg", 1, 1, ""),
    ("mean entropy", "ent_fg", 1, 1, ""),
    ("Soft Dice Confidence", "sdc_fg", 1, 1, ""),
    ("MC stochastic depth (10 passes)", "mc_dice", 10, 1, "the segmenter has no dropout layer; its DropPath (stochastic depth, rate 0.1) is switched on at inference"),
    ("TTA entropy (6 passes)", "tta_ent", 6, 1, "flip x scales 0.75, 1.0, 1.25"),
    ("TTA pairwise Dice (6 passes)", "tta_dice", 6, 1, "flip x scales 0.75, 1.0, 1.25"),
    ("4-model ensemble pairwise Dice", "ens_dice", 4, 4, "TEST ONLY: every training sequence lies inside some member's training set"),
]


def load_scores(scores_dir, split):
    out = {}
    for f in sorted(os.listdir(os.path.join(scores_dir, "scores"))):
        if f.startswith(split + "__"):
            z = np.load(os.path.join(scores_dir, "scores", f), allow_pickle=False)
            out[str(z["seq"])] = {k: z[k] for k in z.files}
    return out


def align(split, scores, caches, actions_dir):
    """One row per frame, in sequence order then frame order. Refuses to continue on any mismatch."""
    rows = {}
    keys = sorted(scores, key=lambda k: (int(k.split("seq_")[1])))
    for k in keys:
        s = scores[k]
        c = caches[k]
        a = np.load(os.path.join(actions_dir, f"{'oof' if split == 'train' else 'test'}__{split}_{k.split('/')[1]}.npz"), allow_pickle=True)
        n = len(s["risk"])
        if not (n == len(c) == len(a["t"])):
            raise SystemExit(f"FRAME COUNT MISMATCH {k}: scores {n}, cache {len(c)}, actions {len(a['t'])}")
        if not np.array_equal(np.asarray(a["t"]), np.asarray(s["t"])):
            raise SystemExit(f"FRAME INDEX MISMATCH {k}")
        # The cache and the action data come from two separate inference runs of the same model, so their instrument IoUs agree only up to numerical
        # noise (fp16 and cuDNN: about 0.1% of pixels differ; on a few frames of some sequences an instrument class flips and the IoU moves by up to 0.3).
        # Measured on all 19 sequences: mean absolute difference <= 0.0022 and correlation >= 0.994, whereas shifting the frame list by one frame gives
        # a correlation of at most 0.82. The check is therefore: mean absolute difference below ALIGN_TOL and correlation above ALIGN_CORR; the observed
        # mean, maximum and correlation are recorded per sequence.
        diff = np.abs(np.asarray(a["instr_before"], float) - c.r_star.astype(float))
        corr = float(np.corrcoef(a["instr_before"], c.r_star)[0, 1]) if len(c) > 2 else 1.0
        if not (float(np.nanmean(diff)) < ALIGN_TOL and corr > ALIGN_CORR):
            raise SystemExit(f"INSTRUMENT-IoU MISMATCH {k}: cache r_star differs from actions instr_before (mean abs {np.nanmean(diff):.2e}, "
                             f"max abs {np.nanmax(diff):.2e}, correlation {corr:.5f}); the frames are not the same, refusing to continue")
        ALIGN_LOG[k] = dict(mean_abs_difference=float(np.nanmean(diff)), max_abs_difference=float(np.nanmax(diff)), correlation=corr)
        rows[k] = dict(s=s, r_star=c.r_star.astype(np.float64), prod=np.asarray(a["r"], np.float64))
    return keys, rows


def gate_scores(oof, test, folds, K=5, tau=0.5):
    """Per-frame gate scores for the two GBM variants, keyed by sequence, in frame order."""
    out = {"oof": {}, "test": {}}
    kw = {"seed": 0}
    for name, setname in (("gate: confidence only", "nontemporal"), ("gate: + temporal (9 indicators)", "extended")):
        rh, rs, sq = cv_run(oof, folds, "gbm", setname, K, tau, THETA, kw)
        out["oof"][name] = {k: rh[sq == k] for k in np.unique(sq)}
        th, ts, tsq = test_run(oof, test, "gbm", setname, K, tau, THETA, kw)
        out["test"][name] = {k: th[tsq == k] for k in np.unique(tsq)}
    return out


def metric_set(score, risk, bad, bad_instr):
    ok = np.isfinite(score) & np.isfinite(risk)
    sc, rk, b, bi = score[ok], risk[ok], bad[ok], bad_instr[ok]
    return dict(auroc=roc_auc(-sc, b), aurc=aurc(sc, rk), auroc_instr_label=roc_auc(-sc, bi), n=int(ok.sum()),
                aurc_oracle=aurc_oracle(rk), aurc_random=float(rk.mean()), bad_rate=float(b.mean()))


def boot(score, risk, bad, bad_instr, seq, ref=None, n_boot=N_BOOT, seed=0):
    """Cluster bootstrap over sequences. Returns CIs for AUROC and AURC, and (if ref given) for the paired
    difference score - ref. Resamples with fewer than one bad and one good frame are skipped (count reported)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seq)
    idx = {s: np.where(seq == s)[0] for s in uniq}
    A, U, DA, DU, skipped = [], [], [], [], 0
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        b = bad[ii]
        if b.min() == b.max():
            skipped += 1
            continue
        A.append(roc_auc(-score[ii], b)); U.append(aurc(score[ii], risk[ii]))
        if ref is not None:
            DA.append(roc_auc(-score[ii], b) - roc_auc(-ref[ii], b))
            DU.append(aurc(score[ii], risk[ii]) - aurc(ref[ii], risk[ii]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = dict(auroc_ci95=ci(A), aurc_ci95=ci(U), n_boot_used=len(A), n_boot_skipped=skipped, n_clusters=int(len(uniq)))
    if ref is not None:
        out["diff_auroc_vs_production_gate"] = dict(mean=float(np.mean(DA)), ci95=ci(DA))
        out["diff_aurc_vs_production_gate"] = dict(mean=float(np.mean(DU)), ci95=ci(DU))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work_v2")
    ap.add_argument("--scores", default="work_v2/failure_detection")
    ap.add_argument("--out", default="results/taskC")
    ap.add_argument("--latency", default=None, help="JSON written by tools/failure_detect_latency.py")
    ap.add_argument("--xv-dir", default=None, help="results/taskD: gate scores of the four Task D sets (gate_scores_oof.npz, gate_scores_test.npz) and decision.json; adds gate rows")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    folds = json.load(open(os.path.join(a.work, "folds.json")))["folds"]
    oof_c = load_dir(os.path.join(a.work, "cache_oof_stereo"))
    test_c = {k: v for k, v in load_dir(os.path.join(a.work, "cache_test_stereo")).items() if k.startswith("test/")}
    gs_path = os.path.join(a.scores, "gate_scores_gbm.pkl")      # the two GBM gates take ~20 min to fit; cached so a re-run only recomputes the table
    if os.path.exists(gs_path):
        import pickle
        gs = pickle.load(open(gs_path, "rb"))
    else:
        import pickle
        gs = gate_scores(oof_c, test_c, folds)
        pickle.dump(gs, open(gs_path, "wb"))
    xv_label = None
    if a.xv_dir:
        dec = json.load(open(os.path.join(a.xv_dir, "decision.json")))["decision"]
        xv_label = f"Task D pre-set test: {'PASSED' if dec['passed'] else 'NOT passed'} (AUROC gain {dec['auroc_gain_full_sample']:+.3f}, 95% {dec['ci95'][0]:+.3f} to {dec['ci95'][1]:+.3f}, label: instrument IoU < 0.5)"
        names = {"paper4": "gate (GBM): 4 original indicators", "paper4_stereo3": "gate (GBM): 4 + 3 disparity indicators (control)",
                 "paper4_crossview6": "gate (GBM): 4 + 6 cross-view indicators", "crossview6": "gate (GBM): 6 cross-view indicators alone"}
        for grp, fn in (("oof", "gate_scores_oof.npz"), ("test", "gate_scores_test.npz")):
            z = np.load(os.path.join(a.xv_dir, fn), allow_pickle=False)
            sq = z["seq"].astype(str)
            for k, nm in names.items():
                gs[grp][nm] = {u: z[k][sq == u] for u in np.unique(sq)}

    R = {"definition": {"risk": "1 - official frame IoU of the single-pass mask at 1280x1024 (class counted only if in the ground truth)",
                        "bad_frame": f"official frame IoU < {THETA} (the repo's standing threshold, not tuned here)",
                        "intervals": f"95% percentile intervals from {N_BOOT} bootstrap draws that resample whole sequences (seed 0)",
                        "direction": "AUROC higher is better; AURC lower is better"},
         "groups": {}}
    per_frame = []
    for split, group, actions_prefix in (("train", "oof", "oof"), ("test", "test", "test")):
        scores = load_scores(a.scores, split)
        caches = oof_c if split == "train" else test_c
        keys, rows = align(split, scores, caches, os.path.join(a.work, "actions"))
        cat = lambda f: np.concatenate([f(rows[k]) for k in keys])  # noqa: E731
        seq = np.concatenate([[k] * len(rows[k]["risk"] if "risk" in rows[k] else rows[k]["s"]["risk"]) for k in keys])
        risk = cat(lambda r: r["s"]["risk"])
        r_star = cat(lambda r: r["r_star"])
        bad, bad_i = (risk > 1 - THETA).astype(int), (r_star < THETA).astype(int)
        det = {}
        for name, key, passes, models, note in BASELINES:
            if key == "ens_dice" and group == "oof":
                continue
            det[name] = dict(score=cat(lambda r: r["s"][key]).astype(np.float64), passes=passes, models=models, note=note)
        for name in gs[group]:
            xvrow = "cross-view" in name
            det[name] = dict(score=np.concatenate([gs[group][name][k] for k in keys]).astype(np.float64), passes=2 if xvrow else 1, models=1,
                             note=("needs the right image segmented by the same model (a second forward pass) and a small trained estimator; " + (xv_label or "")) if xvrow
                             else "needs a trained reliability estimator (a small gradient-boosted model) in addition to the segmenter")
        det["gate: production (MLP, 9 indicators)"] = dict(score=cat(lambda r: r["prod"]), passes=1, models=1,
                                                         note="the released gate; a small trained estimator on top of the segmenter")
        ref = det["gate: production (MLP, 9 indicators)"]["score"]
        G = {"n_frames": int(len(risk)), "n_with_risk": int(np.isfinite(risk).sum()), "n_sequences": len(keys),
             "bad_rate": float(bad[np.isfinite(risk)].mean()), "detectors": {}}
        for name, d in det.items():
            m = metric_set(d["score"], risk, bad, bad_i)
            m.update(boot(d["score"], risk, bad, bad_i, seq, ref=None if name.endswith("production (MLP, 9 indicators)") else ref))
            m.update(forward_passes=d["passes"], segmenters_trained=d["models"], note=d["note"])
            G["detectors"][name] = m
        R["groups"][group] = G
        for i in range(len(risk)):
            per_frame.append([group, seq[i], i, risk[i], r_star[i]] + [det[n]["score"][i] if n in det else np.nan for n in sorted(det)])
        R["groups"][group]["_columns"] = sorted(det)
    R["frame_alignment_check"] = dict(rule=f"mean abs instrument-IoU difference < {ALIGN_TOL} and correlation > {ALIGN_CORR} between the cache and the action data, per sequence (a one-frame shift gives correlation <= 0.82)",
                                      per_sequence=ALIGN_LOG)
    if a.latency and os.path.exists(a.latency):
        R["latency"] = json.load(open(a.latency))
    with open(os.path.join(a.out, "failure_detection.json"), "w") as f:
        json.dump(R, f, indent=1)
    cols = sorted(set(R["groups"]["test"]["_columns"]) | set(R["groups"]["oof"]["_columns"]))
    with open(os.path.join(a.out, "per_frame_scores.csv"), "w") as f:
        f.write("group,sequence,frame,risk,instrument_iou_risk_label," + ",".join(c.replace(",", ";") for c in sorted(R["groups"]["test"]["_columns"])) + "\n")
        for r in per_frame:
            if r[0] == "oof":
                continue
            f.write(",".join(str(x) if not isinstance(x, float) else f"{x:.6g}" for x in r) + "\n")
    # markdown
    lines = []
    for g, title in (("oof", "Out of fold (15 training sequences, each scored by the fold model that excluded it)"),
                     ("test", "Official test (4 sequences, final model)")):
        G = R["groups"][g]
        lines += [f"### {title}", "",
                  f"{G['n_with_risk']} frames with a defined risk; bad-frame rate {100*G['bad_rate']:.1f}%.", "",
                  "| Detector | AUROC | 95% | AURC | 95% | AUROC (instr.-IoU label) | passes | segmenters trained |",
                  "|---|---|---|---|---|---|---|---|"]
        for n, m in G["detectors"].items():
            lines.append(f"| {n} | {m['auroc']:.3f} | {m['auroc_ci95'][0]:.3f} to {m['auroc_ci95'][1]:.3f} | {m['aurc']:.4f} | "
                         f"{m['aurc_ci95'][0]:.4f} to {m['aurc_ci95'][1]:.4f} | {m['auroc_instr_label']:.3f} | {m['forward_passes']} | {m['segmenters_trained']} |")
        any_ = next(iter(G["detectors"].values()))
        lines += ["", f"AURC floor (oracle ranking) {any_['aurc_oracle']:.4f}; random ranking {any_['aurc_random']:.4f}.", ""]
    open(os.path.join(a.out, "failure_detection.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {a.out}/failure_detection.json, .md and per_frame_scores.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
