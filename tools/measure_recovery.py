#!/usr/bin/env python3
"""
TASK 3 -- does correction actually HELP? Measure it; add nothing.

For every frame that the gate ROUTES to temporal recovery, compare the mask
BEFORE correction (Stage 1's S_t) with the mask AFTER (the existing Stages 3-5)
against ground truth. Also run the always-on baseline (the gate ignored: every
frame that has a memory is corrected), which is the LACOSTE-style comparison.

    python -m tools.measure_recovery --release work_v2/release --split test \
        --out results/task3

Reported, for the routed frames and for the always-on set:
  mean IoU change; % improved; % worse (the HARMFUL-CORRECTION RATE);
  mean size of improvements and of damage; per instrument class.
Two IoU definitions are given because they answer different questions:
  instrument IoU  mean over the instrument classes present in GT or prediction
                  (what the existing runner used)
  official IoU    the challenge's own per-frame score (sitr.metrics.official_*)

HONESTY RULES baked in:
  * default split is TEST. On `train` the final Stage-1 model has seen the
    labels, so every IoU is inflated; the tool warns loudly if you ask for it.
  * nothing is tuned here. The margin below only decides what counts as
    "unchanged"; counts with the strict sign (any change) are always printed too.
  * ground truth is read only AFTER the correction has produced its mask.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.align import build_align  # noqa: E402
from sitr.identity import HungarianIdentityVerifier  # noqa: E402
from sitr.metrics import frame_iou, official_class_ious, official_frame_iou  # noqa: E402
from sitr.recovery import TemporalRecovery  # noqa: E402
from sitr.runtime import ReleaseBundle  # noqa: E402
from sitr.utils import keep_awake  # noqa: E402

PALETTE = np.array([(0, 0, 0), (0, 255, 0), (0, 255, 255), (125, 255, 12), (255, 55, 0),
                    (24, 55, 125), (187, 155, 25), (0, 255, 125), (255, 255, 125),
                    (123, 15, 175), (124, 155, 5), (12, 255, 141)] + [(255, 255, 255)] * 20, np.uint8)


class Keeper:
    """Keep the n frames with the largest (or smallest) delta, with their arrays."""

    def __init__(self, n, largest=True):
        self.n, self.sign, self.h, self.k = n, (1 if largest else -1), [], 0

    def offer(self, delta, make):
        item = (self.sign * delta, self.k)
        self.k += 1
        if len(self.h) < self.n:
            heapq.heappush(self.h, (item, make()))
        elif item > self.h[0][0]:
            heapq.heapreplace(self.h, (item, make()))

    def items(self):
        return [v for _, v in sorted(self.h, key=lambda t: -t[0][0])]


def run_mode(bundle, seqs, mode, aligner, gamma, keep_best, keep_worst):
    """One pass over the sequences. mode: 'selective' | 'always'."""
    tau, nc, inst = bundle.tau, bundle.num_classes, bundle.instrument_ids
    recs = []
    for seq in seqs:
        rec = TemporalRecovery(aligner=aligner, tau=tau, gamma=gamma, num_classes=nc, instrument_ids=inst)
        runner = bundle.runner(recovery=rec, identity=HungarianIdentityVerifier(instrument_ids=inst))
        runner.reset()
        t0 = time.time()
        for state in bundle.stream(seq, with_gt=True):
            K = runner.K
            alive = [m for m in runner.memory if m.t >= state.t - K]
            state.cold_start = (len(alive) == 0)                  # the runner's own memory
            if mode == "always":
                state.reliable = False                            # ignore the gate
            routed = bool(state.needs_recovery)
            before = state.mask.copy()
            final, _ = runner.step(state)                         # the correction happens here
            gt = state.ground_truth                               # GT read only AFTER correction
            if gt is None:
                continue
            r = dict(seq=seq.key, t=int(state.t), reliability=float(state.reliability),
                     reliable=bool(state.reliable), cold=bool(state.cold_start),
                     degenerate=bool(state.degenerate), routed=routed,
                     instr_before=frame_iou(before, gt, nc, class_ids=inst),
                     instr_after=frame_iou(final, gt, nc, class_ids=inst),
                     off_before=official_frame_iou(before, gt, nc),
                     off_after=official_frame_iou(final, gt, nc),
                     changed_px=int((before != final).sum()))
            cb, ca = official_class_ious(before, gt, nc), official_class_ious(final, gt, nc)
            r["cls_before"] = {int(c): v for c, v in cb.items() if c in inst}
            r["cls_after"] = {int(c): v for c, v in ca.items() if c in inst}
            recs.append(r)
            if routed and mode == "selective":
                d = r["instr_after"] - r["instr_before"]
                make = lambda: dict(image=state.image.copy(), gt=gt.copy(), before=before,
                                    after=final.copy(), seq=seq.key, t=int(state.t), delta=d,
                                    ib=r["instr_before"], ia=r["instr_after"], rel=r["reliability"])
                keep_best.offer(d, make)
                keep_worst.offer(d, make)
        print(f"    [{mode}] {seq.key}: {sum(1 for r in recs if r['seq']==seq.key)} frames, "
              f"{sum(1 for r in recs if r['seq']==seq.key and r['routed'])} routed, {time.time()-t0:.0f}s", flush=True)
    return recs


def sign_test_p(n_pos, n_neg):
    """Two-sided exact binomial sign test (ties dropped)."""
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    k = min(n_pos, n_neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def summarise(recs, key_b, key_a, margin, label, n_boot=2000, seed=0):
    sel = [r for r in recs if r["routed"] and r[key_b] is not None and r[key_a] is not None]
    if not sel:
        return dict(label=label, n=0)
    b = np.array([r[key_b] for r in sel]); a = np.array([r[key_a] for r in sel]); d = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    imp, wor = d > margin, d < -margin
    return dict(label=label, n=int(len(d)), mean_before=float(b.mean()), mean_after=float(a.mean()),
                mean_delta=float(d.mean()), median_delta=float(np.median(d)),
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                pct_improved=100 * float(imp.mean()), pct_worse=100 * float(wor.mean()),
                pct_unchanged=100 * float((~imp & ~wor).mean()),
                pct_strictly_better=100 * float((d > 0).mean()), pct_strictly_worse=100 * float((d < 0).mean()),
                pct_exactly_zero=100 * float((d == 0).mean()),
                mean_gain_when_improved=float(d[imp].mean()) if imp.any() else 0.0,
                mean_damage_when_worse=float(-d[wor].mean()) if wor.any() else 0.0,
                sign_test_p=sign_test_p(int((d > 0).sum()), int((d < 0).sum())), margin=margin)


def per_class(recs, margin):
    out = {}
    for r in recs:
        if not r["routed"]:
            continue
        for c, vb in r["cls_before"].items():
            va = r["cls_after"].get(c)
            if va is None:
                continue
            out.setdefault(c, []).append(va - vb)
    res = {}
    for c, ds in sorted(out.items()):
        d = np.array(ds)
        res[c] = dict(n_frames=int(len(d)), mean_delta=float(d.mean()),
                      pct_improved=100 * float((d > margin).mean()), pct_worse=100 * float((d < -margin).mean()))
    return res


def save_panels(items, outdir, names, tag):
    import cv2
    os.makedirs(outdir, exist_ok=True)
    for rank, it in enumerate(items):
        tiles = [it["image"][..., ::-1] if it["image"].ndim == 3 else it["image"]]
        for m in (it["gt"], it["before"], it["after"]):
            col = PALETTE[np.clip(m, 0, 31)]
            col[m == 255] = (80, 80, 80)
            tiles.append(col[..., ::-1])
        h = min(t.shape[0] for t in tiles)
        tiles = [np.ascontiguousarray(t[:h]) for t in tiles]  # cv2.putText needs contiguous memory
        for t, lab in zip(tiles, ("image", "ground truth", f"BEFORE {it['ib']:.2f}", f"AFTER {it['ia']:.2f}")):
            cv2.putText(t, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        row = np.hstack(tiles)
        cv2.putText(row, f"{it['seq']} t={it['t']} delta={it['delta']:+.3f} r={it['rel']:.2f}", (6, row.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(outdir, f"{tag}_{rank+1:02d}_{it['seq'].replace('/', '_')}_t{it['t']:03d}.png"), row)


def interpret(s):
    """Plain-language verdict, from the numbers only."""
    if s.get("n", 0) == 0:
        return "no routed frames: nothing to measure"
    lo, hi = s["ci95"]
    if s["pct_worse"] > s["pct_improved"] and hi < 0.0:
        return "DAMAGES: correction makes more routed frames worse than better and the mean change is negative"
    if s["pct_worse"] > s["pct_improved"]:
        return "NET HARMFUL IN COUNT: more routed frames got worse than better (mean CI includes 0)" if lo <= 0 <= hi \
            else "mixed: more frames got worse than better but the mean change is positive"
    if lo <= 0.0 <= hi:
        return "WITHIN NOISE OF ZERO: the mean change's 95% interval includes 0"
    if hi < 0:
        return "DAMAGES on average"
    return ("IMPROVES: more routed frames got better than worse and the mean CI is above 0 "
            "(read the effect size and the unchanged share before calling it useful)"
            if s["pct_improved"] > s["pct_worse"] else "mixed")


def fold_bundles(release, work):
    """Honest bundles for the TRAINING sequences: for each cross-fitting fold, the
    segmenter is that fold's model (it never saw those sequences) and the
    reliability gate is fit on the OTHER folds' out-of-fold caches. Yields
    (fold index, bundle, held-out sequences).

    Caveat, stated not hidden: the routing threshold tau is the release's, chosen
    by conformal control on ALL out-of-fold frames, so it has seen these frames'
    reliability labels. It decides which frames are routed, not their masks."""
    import copy
    import torch
    from sitr.meta_models import SklearnReliability
    from sitr.reliability import StreamConfig
    from sitr.stage2v2 import FAMILIES
    from sitr.models import build_model
    from sitr.utils import load_checkpoint
    from tools.compare_stereo import load_dir
    base = ReleaseBundle.load(release)
    man = base.manifest
    folds = json.load(open(os.path.join(work, "folds.json")))["folds"]
    oof = load_dir(os.path.join(work, "cache_oof"))
    fam_name = man.get("stage2_family", "gbm")
    cfg = StreamConfig(memory_size=int(man["memory_size"]), tau=float(man.get("tau_train", 0.5)),
                       indicators=man["indicator_set"])
    all_seqs = {s.key: s for s in base.sequences("train")}
    for i, held in enumerate(folds):
        ck = load_checkpoint(os.path.join(work, "crossfit", f"fold{i}", "best.pt"), map_location=base.device)
        meta = ck["meta"]
        model = build_model(int(meta["num_classes"]), arch=meta.get("arch", meta.get("args", {}).get("arch")),
                            pretrained=False).to(base.device)
        model.load_state_dict(ck["model"]); model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        kw = {"seed": 0}
        if fam_name == "mlp":
            kw["epochs"] = 300          # the value stage 2 selection used (tools/train_stage2_cv.py default)
        fam = FAMILIES[fam_name](**kw).fit([c for k, c in oof.items() if k not in held], cfg)
        b = copy.copy(base)
        b.model = model
        if fam_name == "gbm":
            b.reliability = SklearnReliability(fam.model, in_dim=len(man["indicator_names"]))
        else:
            b.reliability = fam.model.to(base.device).eval()
        # Same rule run_v2 used for the OOF caches: the calibration temperature of a
        # fold's predictions is the mean over the OTHER folds (never its own).
        temps = [json.load(open(os.path.join(work, "crossfit", f"fold{j}", "calibration.json")))["temperature"]
                 for j in range(len(folds))]
        b.manifest = dict(man)
        b.manifest["temperature"] = float(np.mean([t for j, t in enumerate(temps) if j != i]))
        yield i, b, [all_seqs[k] for k in held]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--oof-work", default=None,
                    help="work_v2 folder: evaluate the TRAINING sequences honestly with per-fold models "
                         "(instead of --split)")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--aligner", default="identity", choices=["identity", "farneback", "region"])
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--margin", type=float, default=0.01,
                    help="IoU change smaller than this counts as 'unchanged' (a stated choice)")
    ap.add_argument("--sequences", type=int, default=None)
    ap.add_argument("--out", default="results/task3")
    ap.add_argument("--n-images", type=int, default=10)
    a = ap.parse_args()
    keep_awake()
    os.makedirs(a.out, exist_ok=True)
    if a.split == "train":
        print("!" * 72 + "\n! WARNING: split=train. The final Stage-1 model was trained on these\n"
              "! sequences, so every IoU below is INFLATED. Use --split test for the truth.\n" + "!" * 72)

    bundle = ReleaseBundle.load(a.release)
    print(f"release: num_classes={bundle.num_classes} tau={bundle.tau:.4f} tau_memory={bundle.tau_memory:.4f} "
          f"K={bundle.memory_size}  instrument_ids={bundle.instrument_ids}")
    kw = {"instrument_ids": bundle.instrument_ids} if a.aligner == "region" else {}
    aligner = build_align(a.aligner, **kw)
    if a.oof_work:
        groups = [(f"fold{i}", b, s) for i, b, s in fold_bundles(a.release, a.oof_work)]
        a.split = "train-OOF"
        print("split=train-OOF: honest evaluation of the training sequences, per-fold models")
    else:
        seqs = bundle.sequences(split=a.split)
        if a.sequences:
            seqs = seqs[:a.sequences]
        groups = [(a.split, bundle, seqs)]
    print(f"{sum(len(g[2]) for g in groups)} sequences, aligner={aligner.name}, gamma={a.gamma}")

    best, worst = Keeper(a.n_images, True), Keeper(a.n_images, False)
    res = {"release": os.path.abspath(a.release), "split": a.split, "aligner": a.aligner,
           "gamma": a.gamma, "tau": bundle.tau, "tau_memory": bundle.tau_memory, "margin": a.margin}
    all_recs = {}
    for mode in ("selective", "always"):
        print(f"\n--- {mode.upper()} ---")
        recs = []
        for gname, gb, gseqs in groups:
            if a.oof_work:
                print(f"  [{gname}] held-out sequences {[s.key for s in gseqs]}")
            recs += run_mode(gb, gseqs, mode, aligner, a.gamma, best, worst)
        all_recs[mode] = recs
        n, nr = len(recs), sum(r["routed"] for r in recs)
        res[mode] = dict(n_frames=n, n_routed=nr, routing_rate=nr / max(n, 1),
                         n_cold_start=sum(r["cold"] and not r["reliable"] for r in recs),
                         instrument_iou=summarise(recs, "instr_before", "instr_after", a.margin, "instrument IoU"),
                         official_iou=summarise(recs, "off_before", "off_after", a.margin, "official IoU"),
                         per_class_official_iou=per_class(recs, a.margin),
                         # overall (all frames) before/after, for context
                         overall_instr_before=float(np.mean([r["instr_before"] for r in recs])),
                         overall_instr_after=float(np.mean([r["instr_after"] for r in recs])))
        print(f"  frames={n} routed={nr} ({100*nr/max(n,1):.1f}%)")
        for k in ("instrument_iou", "official_iou"):
            s = res[mode][k]
            if s.get("n"):
                print(f"  {s['label']:15s} n={s['n']:4d}  before {s['mean_before']:.4f} -> after {s['mean_after']:.4f}  "
                      f"mean change {s['mean_delta']:+.4f} (95% CI {s['ci95'][0]:+.4f}..{s['ci95'][1]:+.4f})\n"
                      f"    improved {s['pct_improved']:.1f}%  WORSE {s['pct_worse']:.1f}%  unchanged {s['pct_unchanged']:.1f}%   "
                      f"[strict sign: better {s['pct_strictly_better']:.1f}% / worse {s['pct_strictly_worse']:.1f}% / exactly 0 {s['pct_exactly_zero']:.1f}%]\n"
                      f"    mean gain when improved {s['mean_gain_when_improved']:+.4f}   mean damage when worse {s['mean_damage_when_worse']:.4f}   "
                      f"sign-test p={s['sign_test_p']:.3g}")
        print(f"  VERDICT ({mode}, instrument IoU): {interpret(res[mode]['instrument_iou'])}")
        res[mode]["verdict"] = interpret(res[mode]["instrument_iou"])

    names = bundle.target_names
    save_panels(best.items(), os.path.join(a.out, "best"), names, "best")
    save_panels(worst.items(), os.path.join(a.out, "worst"), names, "worst")
    with open(os.path.join(a.out, "measure_recovery.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out, "per_frame.json"), "w") as f:
        json.dump(all_recs, f)
    print(f"\nwrote {a.out}/measure_recovery.json, per_frame.json and best/ worst/ image panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
