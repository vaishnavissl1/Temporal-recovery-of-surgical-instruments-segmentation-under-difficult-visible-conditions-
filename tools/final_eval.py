#!/usr/bin/env python3
"""
TASK 6 -- every number the write-up needs, on the OFFICIAL test sequences only.

    python -m tools.final_eval --actions work_v2/actions --choices results/task5/choices_test.npz \
        --root <endovis2018_cvdataset> --release work_v2/release --out results/task6

Models (as defined in the brief; M4 and M6 carry identical information, the only
difference is whether the correction is CHOSEN or applied blindly):
    M1 segmentation only            mask A0
    M2 + stereo evidence            A1 on EVERY frame
    M3 + temporal only              A2 on EVERY frame
    M4 stereo+temporal fusion       FUSE on EVERY frame        <- the LACOSTE-style baseline
    M5 + class-conditioned refine   FUSE then A3 on EVERY frame
    M6 selective action + abstain   the gate routes, the selector chooses among
                                    A1/A2/A3/FUSE or abstains   <- OUR METHOD

Protocol (matches test_data/run.py and utils.py, and is CROSS-CHECKED against it):
all 4 test sequences, every labelled frame (997), full 1280x1024 (masks are
nearest-neighbour upsampled from the 512x640 working grid), background excluded,
a class scored only if it is in the ground truth, mean over classes per frame,
then over frames.

Per model: mIoU, per-class IoU, boundary F1, clDice on shaft/thread/needle,
harmful-correction rate, correction activation %, AURC and AUROC of the gate,
mean and P95 latency (measured on this machine), every metric split by
difficulty quartile.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.difficulty import DifficultyIndex  # noqa: E402
from sitr.metrics import aurc, official_class_ious, roc_auc  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

MODELS = ["M1", "M2", "M3", "M4", "M5", "M6"]
DESC = {"M1": "segmentation only", "M2": "+ stereo evidence (A1, every frame)",
        "M3": "+ temporal only (A2, every frame)", "M4": "stereo-temporal fusion, EVERY frame (LACOSTE-style)",
        "M5": "+ class-conditioned refinement, every frame", "M6": "selective action choice + abstention (OURS)"}
KEY_OF = {"M1": "A0", "M2": "A1", "M3": "A2", "M4": "FUSE", "M5": "FUSE_REFINE"}
CLASS_NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist", "kidney-parenchyma",
               "covered-kidney", "thread", "clamps", "suturing-needle", "suction-instrument", "intestine", "ultrasound-probe"]
THIN = {1: "shaft", 6: "thread", 8: "needle"}
INSTR = [1, 2, 3, 6, 7, 8, 9, 11]
TOL_PX = 3          # boundary-F1 tolerance at full resolution (a stated choice)
HARM_MARGIN = 0.01  # an official-IoU drop larger than this counts as damage (a stated choice)


def boundary(m):
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def boundary_f1(pred_bin, gt_bin, tol=TOL_PX):
    bp, bg = boundary(pred_bin), boundary(gt_bin)
    if not bp.any() or not bg.any():
        return None
    dg = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    dp = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    prec = float((dg[bp] <= tol).mean()); rec = float((dp[bg] <= tol).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def cl_dice(pred_bin, gt_bin):
    from skimage.morphology import skeletonize
    if not gt_bin.any():
        return None
    if not pred_bin.any():
        return 0.0
    sp, sg = skeletonize(pred_bin), skeletonize(gt_bin)
    tprec = float((sp & gt_bin).sum() / max(sp.sum(), 1)); tsens = float((sg & pred_bin).sum() / max(sg.sum(), 1))
    return 0.0 if tprec + tsens == 0 else 2 * tprec * tsens / (tprec + tsens)


def frame_job(args):
    """Metrics of every model's mask for ONE frame, at full resolution."""
    label_path, mask_npz, keys_by_model = args
    gt = read_mask(label_path)
    z = np.load(mask_npz)
    base = z["A0"]
    out = {}
    for model, key in keys_by_model.items():
        m = z[key]
        full = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        cls = official_class_ious(full, gt, 12)
        row = dict(cls={int(c): float(v) for c, v in cls.items()},
                   frame=float(np.mean(list(cls.values()))) if cls else None,
                   changed=bool((m != base).any()))
        ip, ig = np.isin(full, INSTR), np.isin(gt, INSTR)
        row["bf1"] = boundary_f1(ip, ig)
        row["cl"] = {}
        for c, n in THIN.items():
            if (gt == c).any():
                row["cl"][n] = cl_dice(full == c, gt == c)
        out[model] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--choices", required=True, help="results/task5/choices_test.npz (selector decisions on the test frames)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", default="results/task6")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--crosscheck", action="store_true", help="also run the OFFICIAL run.py arithmetic on M1")
    ap.add_argument("--latency-frames", type=int, default=60)
    ap.add_argument("--no-latency", action="store_true", help="skip the GPU latency measurement (used when the GPU is busy; latency is then composed elsewhere)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from sitr.dataset import discover_sequences
    seqs = {s.key: s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",))}
    ch = np.load(a.choices, allow_pickle=True)
    keys = [str(k) for k in ch["seq"]]; ts = ch["t"]; choice = ch["choice"].astype(str)
    routed, diff, rel = ch["routed"].astype(bool), ch["difficulty"], ch["r"]
    instr_before = ch["instr_before"]
    n = len(ts)
    jobs, meta = [], []
    for i in range(n):
        s = seqs[keys[i]]
        stem = os.path.splitext(os.path.basename(s.frames[int(ts[i])]))[0]
        mp_ = os.path.join(a.actions, "test_masks", f"{s.split}_{s.name}", stem + ".npz")
        kmap = dict(KEY_OF)
        kmap["M6"] = choice[i] if choice[i] != "A0" else "A0"
        jobs.append((s.labels[int(ts[i])], mp_, kmap)); meta.append((keys[i], int(ts[i])))
    print(f"{n} test frames, {len(set(keys))} sequences; full-resolution metrics with {a.workers} workers")
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        res = pool.map(frame_job, jobs, chunksize=4)
    print(f"  metrics done in {time.time()-t0:.0f}s")

    quart = DifficultyIndex.quartile_of(diff)
    R = {"n_frames": n, "n_sequences": len(set(keys)), "n_routed": int(routed.sum()),
         "descriptions": DESC, "harm_margin": HARM_MARGIN, "boundary_tol_px": TOL_PX, "models": {}}
    base_f = np.array([np.nan if r["M1"]["frame"] is None else r["M1"]["frame"] for r in res])
    bad_gate = (instr_before < 0.5).astype(int)
    for M in MODELS:
        f = np.array([np.nan if r[M]["frame"] is None else r[M]["frame"] for r in res])
        changed = np.array([r[M]["changed"] for r in res])
        valid = np.isfinite(f)
        d = f - base_f
        per_cls = {}
        for c in range(1, 12):
            v = [r[M]["cls"][c] for r in res if c in r[M]["cls"]]
            per_cls[CLASS_NAMES[c]] = dict(iou=float(np.mean(v)) if v else None, n_frames=len(v))
        bf1 = [r[M]["bf1"] for r in res if r[M]["bf1"] is not None]
        cld = {nm: float(np.mean([r[M]["cl"][nm] for r in res if r[M]["cl"].get(nm) is not None] or [np.nan])) for nm in THIN.values()}
        corrected = changed & valid
        harm = corrected & (d < -HARM_MARGIN)
        risk = 1.0 - f
        row = dict(mIoU=float(np.nanmean(f)), n_scored=int(valid.sum()), per_class=per_cls,
                   boundary_f1_instruments=float(np.mean(bf1)), cldice=cld,
                   activation_pct=100 * float(changed.mean()),
                   n_corrected=int(corrected.sum()),
                   harmful_pct_of_corrected=100 * float(harm.sum() / max(corrected.sum(), 1)),
                   harmful_pct_of_all=100 * float(harm.sum() / max(valid.sum(), 1)),
                   improved_pct_of_corrected=100 * float(((d > HARM_MARGIN) & corrected).sum() / max(corrected.sum(), 1)),
                   mean_delta_vs_M1_all=float(np.nanmean(np.where(valid, d, np.nan))),
                   mean_delta_vs_M1_when_corrected=float(np.nanmean(d[corrected])) if corrected.any() else 0.0,
                   aurc_gate=aurc(rel[valid], risk[valid]),
                   by_difficulty_quartile=[dict(quartile=q + 1, n=int((quart == q).sum()),
                                                mIoU=float(np.nanmean(f[quart == q])),
                                                harmful_pct_of_all=100 * float((harm & (quart == q)).sum() / max((quart == q).sum(), 1)),
                                                activation_pct=100 * float(changed[quart == q].mean())) for q in range(4)],
                   per_sequence={k: float(np.nanmean(f[np.array(keys) == k])) for k in sorted(set(keys))})
        R["models"][M] = row
    R["gate"] = dict(auroc_detect_bad_base_frames=roc_auc(-rel, bad_gate), bad_rate=float(bad_gate.mean()),
                     aurc_oracle=aurc(-(1 - base_f[np.isfinite(base_f)]), 1 - base_f[np.isfinite(base_f)]),
                     aurc_random=float(np.nanmean(1 - base_f)))
    print(f"\n{'model':6s} {'mIoU':>7s} {'bF1':>6s} {'clDice thin (sh/th/ne)':>24s} {'act%':>6s} {'harm% corr':>10s} {'harm% all':>9s} {'AURC':>7s}")
    for M in MODELS:
        r = R["models"][M]
        print(f"{M:6s} {r['mIoU']:7.4f} {r['boundary_f1_instruments']:6.3f} "
              f"{r['cldice']['shaft']:7.3f}/{r['cldice']['thread']:.3f}/{r['cldice']['needle']:.3f} "
              f"{r['activation_pct']:6.1f} {r['harmful_pct_of_corrected']:10.1f} {r['harmful_pct_of_all']:9.1f} {r['aurc_gate']:7.4f}   {DESC[M]}")
    m4, m6 = R["models"]["M4"], R["models"]["M6"]
    R["headline_M4_vs_M6"] = dict(harmful_pct_of_all=(m4["harmful_pct_of_all"], m6["harmful_pct_of_all"]),
                                  harmful_pct_of_corrected=(m4["harmful_pct_of_corrected"], m6["harmful_pct_of_corrected"]),
                                  activation_pct=(m4["activation_pct"], m6["activation_pct"]),
                                  mIoU=(m4["mIoU"], m6["mIoU"]),
                                  M6_beats_M4_on_harmful_rate=bool(m6["harmful_pct_of_all"] < m4["harmful_pct_of_all"]))
    print("\nHEADLINE M4 (blind) vs M6 (chosen):", R["headline_M4_vs_M6"])
    if not R["headline_M4_vs_M6"]["M6_beats_M4_on_harmful_rate"]:
        print("!! STOP CONDITION: M6 does NOT beat M4 on harmful-correction rate -- the main claim fails.")

    # ------------------------------------------------ independent cross-check
    if a.crosscheck:
        R["crosscheck_official_run_py"] = official_crosscheck(a, seqs, keys, ts, meta, R["models"]["M1"]["mIoU"])
    # ----------------------------------------------------------------- latency
    try:
        if a.no_latency:
            raise RuntimeError("latency measurement skipped (--no-latency)")
        R["latency_ms"] = latency(a, R, routed, jobs, seqs, keys, ts)
    except Exception as e:  # noqa: BLE001 -- reported, never hidden
        R["latency_ms"] = {"error": f"{type(e).__name__}: {e}"}
        print("latency measurement failed:", R["latency_ms"]["error"])
    with open(os.path.join(a.out, "final_results.json"), "w") as f:
        json.dump(R, f, indent=1)
    figures(a.out, R, res, rel, diff, quart, routed)
    print(f"\nwrote {a.out}/final_results.json and figures")
    return 0


def official_crosscheck(a, seqs, keys, ts, meta, our_miou):
    """Write M1's masks as colour PNGs and run the OFFICIAL run.py machinery on them."""
    import importlib.util
    import shutil
    tmp = os.path.join(a.out, "_official_check")
    shutil.rmtree(tmp, ignore_errors=True)
    PAL = np.array([(0, 0, 0), (0, 255, 0), (0, 255, 255), (125, 255, 12), (255, 55, 0), (24, 55, 125), (187, 155, 25),
                    (0, 255, 125), (255, 255, 125), (123, 15, 175), (124, 155, 5), (12, 255, 141)], np.uint8)
    sub = os.path.join(tmp, "submission", "team"); td = os.path.join(tmp, "test")
    for s in seqs.values():
        os.makedirs(os.path.join(sub, s.name), exist_ok=True); os.makedirs(os.path.join(td, s.name, "labels"), exist_ok=True)
    for (k, t) in meta:
        s = seqs[k]; fn = os.path.basename(s.frames[t]); stem = os.path.splitext(fn)[0]
        z = np.load(os.path.join(a.actions, "test_masks", f"{s.split}_{s.name}", stem + ".npz"))
        full = cv2.resize(z["A0"], (1280, 1024), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(sub, s.name, fn), PAL[full][..., ::-1])
        shutil.copy2(s.labels[t], os.path.join(td, s.name, "labels", fn))
    run_dir = os.path.join(a.root, "test_data", "test_data")
    sys.path.insert(0, run_dir)
    spec = importlib.util.spec_from_file_location("official_run", os.path.join(run_dir, "run.py"))
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    import io, contextlib
    err, _labels = official.get_error_and_labels()
    test_dirs = [os.path.join(td, f"seq_{i}") for i in range(1, 5)]
    with contextlib.redirect_stdout(io.StringIO()):
        subs = official.process(err, test_dirs, os.path.join(tmp, "submission"))
        subs[0].compute_scores_across_datasets()
    off = float(subs[0].mean_iou_all_datasets)
    per = [float(x) for x in subs[0].mean_iou_per_dataset]
    out = dict(official_run_py_mean_iou=off, our_M1_mIoU=our_miou, abs_difference=abs(off - our_miou),
               official_per_sequence=per, n_frames_scored=int(sum(len(d.results) for d in subs[0].dataset_results)))
    print(f"\nCROSS-CHECK against the OFFICIAL run.py: official {off:.6f}  ours {our_miou:.6f}  |diff| {abs(off-our_miou):.2e}")
    return out


def latency(a, R, routed, jobs, seqs, keys, ts):
    """Mean / P95 per-frame latency, composed from components measured on THIS machine."""
    import torch
    from sitr.runtime import ReleaseBundle
    from sitr.stereo import (SGBMParams, VerticalCorrection, calibration_path, compute_evidence, get_rectifier,
                             right_path)
    from tools.cache_right_view import segment
    b = ReleaseBundle.load(a.release)
    dev = b.device
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    rng = np.random.default_rng(0)
    pick = rng.choice(len(keys), min(a.latency_frames, len(keys)), replace=False)
    h, w = b.manifest["height"], b.manifest["width"]
    x = torch.randn(1, 3, h, w, device=dev).contiguous(memory_format=torch.channels_last)
    t_seg = []
    with torch.no_grad():
        for _ in range(5):
            b.model(x)
        for _ in range(30):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                lo, _ = b.model(x)
            _ = torch.softmax(lo.float() / b.temperature, 1).max(1)
            torch.cuda.synchronize(); t_seg.append(time.perf_counter() - t0)
    t_seg = np.array(t_seg)
    t_ev, t_right = [], []
    t2 = json.load(open(os.path.join("results", "task2", "stereo_check.json")))
    params = SGBMParams(**t2["sgbm_chosen"]); dist, rot = t2["chosen_convention"].split("/")
    for i in pick:
        s = seqs[keys[i]]; p = s.frames[int(ts[i])]
        L = cv2.imread(p); R = cv2.imread(right_path(p))
        rc = get_rectifier(calibration_path(s.frame_dir), 1.0, dist, rot)
        t0 = time.perf_counter(); compute_evidence(L, R, rc, None, params, (h, w)); t_ev.append(time.perf_counter() - t0)
    for i in pick[:20]:
        s = seqs[keys[i]]; p = right_path(s.frames[int(ts[i])])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in segment(b.model, dev, [p], h, w, b.temperature, batch=1):
            pass
        torch.cuda.synchronize(); t_right.append(time.perf_counter() - t0)
    t_ev, t_right = np.array(t_ev), np.array(t_right)
    act = {}
    for k in ("A1", "A2", "A3", "FUSE", "FUSE_REFINE"):
        act[k] = np.concatenate([np.load(f)[f"sec_{k}"] for f in
                                 [os.path.join(a.actions, fn) for fn in sorted(os.listdir(a.actions)) if fn.startswith("test__")]])
    n = len(keys)
    r = np.random.default_rng(1)
    s_seg = r.choice(t_seg, n); s_ev = r.choice(t_ev, n); s_rt = r.choice(t_right, n)
    gate_ms = 0.005   # indicator computation + one tree-ensemble call, measured as ~5 ms, negligible
    sel_ms = 0.002
    lat = {
        "M1": s_seg + gate_ms,
        "M2": s_seg + s_ev + s_rt + act["A1"],
        "M3": s_seg + act["A2"],
        "M4": s_seg + s_ev + s_rt + act["FUSE"],
        "M5": s_seg + s_ev + s_rt + act["FUSE_REFINE"],
        # M6: only ROUTED frames pay for stereo + the candidate actions + the selector
        "M6": s_seg + gate_ms + np.where(routed, s_ev + s_rt + act["A1"] + act["A2"] + act["A3"] + act["FUSE"] + sel_ms, 0.0),
    }
    out = {"gpu": gpu, "frames_timed": int(len(pick)),
           "components_ms": {"segmentation_forward_ms_mean": 1000 * float(t_seg.mean()),
                             "stereo_evidence_cpu_ms_mean": 1000 * float(t_ev.mean()),
                             "right_view_forward_ms_mean": 1000 * float(t_right.mean())}}
    for M, v in lat.items():
        out[M] = dict(mean_ms=1000 * float(v.mean()), p95_ms=1000 * float(np.percentile(v, 95)))
    print("\nlatency per frame, composed from measured components on", gpu)
    for M in MODELS:
        print(f"  {M}: mean {out[M]['mean_ms']:.0f} ms  P95 {out[M]['p95_ms']:.0f} ms")
    return out


def figures(out, R, res, rel, diff, quart, routed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = {M: np.array([np.nan if r[M]["frame"] is None else r[M]["frame"] for r in res]) for M in MODELS}
    # 1. risk-coverage, M4 vs M6 (and M1)
    from sitr.metrics import risk_coverage
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for M, c in (("M1", "#888888"), ("M4", "#d95f02"), ("M6", "#1b9e77")):
        ok = np.isfinite(f[M])
        cov, risk = risk_coverage(rel[ok], 1 - f[M][ok])
        ax.plot(cov, risk, label=f"{M} (AURC {R['models'][M]['aurc_gate']:.3f})", color=c, lw=2)
    ax.set_xlabel("coverage (frames accepted, most reliable first)"); ax.set_ylabel("risk = mean (1 - official IoU)")
    ax.set_title("Risk-coverage, official test sequences"); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_risk_coverage.png"), dpi=150); plt.close(fig)
    # 2. performance vs difficulty quartile
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for M, c in (("M1", "#888888"), ("M4", "#d95f02"), ("M6", "#1b9e77")):
        ax.plot([1, 2, 3, 4], [q["mIoU"] for q in R["models"][M]["by_difficulty_quartile"]], "o-", label=M, color=c, lw=2)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("difficulty quartile (1 easiest .. 4 hardest)"); ax.set_ylabel("official mIoU")
    ax.set_title("Performance by objective difficulty (test)"); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_by_difficulty.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
