#!/usr/bin/env python3
"""
TASK 2 -- is the calibrated stereo geometry usable? Measure, do not eyeball.

  1. rectify 20 pairs sampled across ALL 19 sequences with the REAL calibration
     and measure the vertical disparity of matched SIFT features
  2. report it for CALIBRATION ONLY and, separately, for calibration + a
     label-free per-sequence vertical self-calibration (sitr/stereo.py), which
     was added after step 1 showed the shipped calibration leaves a stable
     3-14 px tilt on some sequences. Both are always reported.
  3. choose SGBM parameters on 5 TRAINING frames that are not among the 20,
     with a criterion fixed BEFORE looking at any result:
        maximise the fraction of pixels that pass the left-right check AND have
        a photometric residual < 10 gray levels
  4. report % valid disparity and % surviving the left-right check, overall and
     for INSTRUMENT vs TISSUE pixels (ground truth is used here only to split
     the pixels; only TRAINING frames are used for that split)
  5. write images and a JSON with every number

    python -m tools.check_stereo --root <endovis2018_cvdataset> --out results/task2

Nothing here fits a model. The only choices made from data are the SGBM
settings, the distortion/rotation convention and the per-sequence correction.
The correction is estimated from frames 0,5,...,35 of each sequence (the first
40 s: causal, so a deployed system could do the same) and evaluated on other frames.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import discover_sequences  # noqa: E402
from sitr.stereo import (SGBMParams, VerticalCorrection, calibration_path,  # noqa: E402
                         disparity_pair, epipolar_errors,
                         estimate_vertical_correction, get_rectifier,
                         lr_consistent, photometric_residual, right_path)

INSTRUMENT_IDS = (1, 2, 3, 6, 7, 8, 9, 11)     # shaft, clasper, wrist, thread, clamps, needle, suction, probe
RESIDUAL_OK = 10.0                              # gray levels; stated choice
STOP_MEAN_DY, STOP_LR, STOP_INSTR = 3.0, 0.40, 0.10   # the task's stop conditions
PALETTE = {(0, 0, 0): 0, (0, 255, 0): 1, (0, 255, 255): 2, (125, 255, 12): 3,
           (255, 55, 0): 4, (24, 55, 125): 5, (187, 155, 25): 6, (0, 255, 125): 7,
           (255, 255, 125): 8, (123, 15, 175): 9, (124, 155, 5): 10, (12, 255, 141): 11}


def read(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im


def read_mask(path):
    """Colour label PNG -> class ids 0..11 by exact RGB lookup (never interpolated)."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = im[..., [2, 1, 0]]
    out = np.full(rgb.shape[:2], 255, np.uint8)
    for c, i in PALETTE.items():
        out[(rgb[..., 0] == c[0]) & (rgb[..., 1] == c[1]) & (rgb[..., 2] == c[2])] = i
    return out


def pick_frames(seqs, n_eval=20):
    """Deterministic: the middle frame of every sequence (19), plus one extra
    training frame at 1/4 of seq_1. Tuning frames come from training sequences
    and are never among the evaluation frames."""
    ev, used = [], set()
    for s in seqs:
        i = len(s) // 2
        ev.append((s, i)); used.add((s.key, i))
    s0 = next(s for s in seqs if s.key == "train/seq_1")
    ev.append((s0, len(s0) // 4)); used.add((s0.key, len(s0) // 4))
    ev = ev[:n_eval]
    tune = []
    for k in ("train/seq_2", "train/seq_6", "train/seq_11", "train/seq_14", "train/seq_16"):
        s = next(x for x in seqs if x.key == k)
        i = len(s) // 10
        assert (s.key, i) not in used
        tune.append((s, i))
    return ev, tune


def load_pair(s, i, alpha, dist="opencv", rot="as_is", corr=None):
    L = read(s.frames[i]); R = read(right_path(s.frames[i]))
    rc = get_rectifier(calibration_path(s.frame_dir), alpha, dist, rot)
    Lr, Rr = rc.rectify(L, R)
    if corr is not None:
        Rr = corr.apply(Rr)
    return L, R, Lr, Rr, rc


def epi_summary(per):
    ok = [e for e in per if e.get("n_matches")]
    return dict(mean_of_means=float(np.mean([e["dy_mean"] for e in ok])),
                pooled_median=float(np.median([e["dy_median"] for e in ok])),
                worst_pair_mean=float(np.max([e["dy_mean"] for e in ok])),
                p95_mean=float(np.mean([e["dy_p95"] for e in ok])),
                n_pairs_mean_gt_3px=int(sum(e["dy_mean"] > 3.0 for e in ok)),
                n_pairs=len(ok), matches_total=int(sum(e["n_matches"] for e in ok)))


def evaluate(ev, chosen, alpha, dist, rot, corrs, save_dir=None, tag=""):
    """Disparity validity on the evaluation pairs. corrs=None -> calibration only."""
    rows, instr_acc, tissue_acc, disp_all, saved = [], [], [], [], 0
    for s, i in ev:
        corr = corrs[s.key] if corrs is not None else None
        L, R, Lr, Rr, rc = load_pair(s, i, alpha, dist, rot, corr)
        t0 = time.time()
        dl, dr = disparity_pair(Lr, Rr, chosen)
        dt = time.time() - t0
        ok = lr_consistent(dl, dr)
        area = rc.valid_left
        interior = area.copy(); interior[:, :int(chosen.num_disp)] = False
        res = photometric_residual(Lr, Rr, dl)
        row = dict(seq=s.key, frame=os.path.basename(s.frames[i]), sgbm_sec=dt,
                   valid_pct=100 * float((np.isfinite(dl) & area).sum() / area.sum()),
                   lr_pct=100 * float((ok & area).sum() / area.sum()),
                   valid_pct_interior=100 * float((np.isfinite(dl) & interior).sum() / interior.sum()),
                   lr_pct_interior=100 * float((ok & interior).sum() / interior.sum()),
                   residual_median=float(np.nanmedian(res[ok])) if ok.any() else None,
                   focal_rect=rc.focal, area_pct=100 * float(area.mean()))
        if s.split == "train":                      # GT only to SPLIT pixels, training frames only
            gt = rc.rectify_labels(read_mask(s.labels[i]))
            known = gt != 255
            instr = np.isin(gt, INSTRUMENT_IDS) & area
            tissue = known & ~np.isin(gt, INSTRUMENT_IDS) & area
            for name, m, acc in (("instrument", instr, instr_acc), ("tissue", tissue, tissue_acc)):
                n = int(m.sum())
                row[f"{name}_px"] = n
                if n:
                    row[f"{name}_valid_pct"] = 100 * float((np.isfinite(dl) & m).sum() / n)
                    row[f"{name}_lr_pct"] = 100 * float((ok & m).sum() / n)
                    acc.append((n, int((np.isfinite(dl) & m).sum()), int((ok & m).sum())))
        disp_all.append(dl[ok & area])
        rows.append(row)
        print(f"    [{tag}] {s.key:11s} {row['frame']}: valid {row['valid_pct']:5.1f}%  LR {row['lr_pct']:5.1f}%  "
              f"instr LR {row.get('instrument_lr_pct', float('nan')):5.1f}%  "
              f"tissue LR {row.get('tissue_lr_pct', float('nan')):5.1f}%  {dt:.1f}s", flush=True)
        if save_dir and saved < 10 and (s.index in (1, 2, 5, 9, 14) or s.split == "test"):
            side = np.hstack([cv2.resize(Lr, (640, 512)), cv2.resize(Rr, (640, 512))])
            for y in range(0, 512, 32):
                cv2.line(side, (0, y), (1279, y), (0, 255, 0), 1)
            dv = np.where(ok, dl, np.nan)
            vis = np.zeros((*dv.shape, 3), np.uint8)
            fin = np.isfinite(dv)
            vis[fin] = cv2.applyColorMap(np.clip(dv[fin] / chosen.num_disp * 255, 0, 255).astype(np.uint8)[:, None],
                                         cv2.COLORMAP_TURBO)[:, 0]
            both = np.hstack([cv2.resize(Lr, (640, 512)), cv2.resize(vis, (640, 512), interpolation=cv2.INTER_NEAREST)])
            base = f"{s.split}_{s.name}_{row['frame'][:-4]}_{tag}"
            cv2.imwrite(os.path.join(save_dir, base + "_pair_lines.png"), side)
            cv2.imwrite(os.path.join(save_dir, base + "_disparity.png"), both)
            saved += 1

    def agg(acc):
        if not acc:
            return None
        n, v, k = (sum(x[j] for x in acc) for j in range(3))
        return dict(pixels=n, valid_pct=100 * v / n, lr_pct=100 * k / n)
    allv = np.concatenate(disp_all)
    return dict(per_frame=rows, instrument=agg(instr_acc), tissue=agg(tissue_acc),
                disparity_percentiles_lr_ok={f"p{q}": float(np.percentile(allv, q)) for q in (1, 5, 50, 95, 99)},
                frac_disp_ge_190=float((allv >= 190).mean()), frac_disp_le_2=float((allv <= 2).mean()),
                overall=dict(valid_pct=float(np.mean([r["valid_pct"] for r in rows])),
                             lr_pct=float(np.mean([r["lr_pct"] for r in rows])),
                             valid_pct_interior=float(np.mean([r["valid_pct_interior"] for r in rows])),
                             lr_pct_interior=float(np.mean([r["lr_pct_interior"] for r in rows])),
                             sgbm_sec_per_frame=float(np.mean([r["sgbm_sec"] for r in rows]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/task2")
    ap.add_argument("--threads", type=int, default=8, help="OpenCV threads; leave cores free for training")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--refine-frames", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 35],
                    help="frame indices used to estimate each sequence's vertical correction")
    a = ap.parse_args()
    cv2.setNumThreads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t00 = time.time()

    seqs = discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("train", "test"))
    seqs.sort(key=lambda s: (s.split != "train", s.index))
    ev, tune = pick_frames(seqs)
    print(f"evaluation pairs: {len(ev)} across {len({s.key for s, _ in ev})} sequences; "
          f"tuning frames: {len(tune)} (training only)")
    for s, i in ev + tune:
        assert i not in a.refine_frames, "estimation frames must be disjoint from evaluation/tuning frames"
    R_ = {"alpha": a.alpha, "refine_frames": a.refine_frames,
          "eval_frames": [f"{s.key}/{os.path.basename(s.frames[i])}" for s, i in ev],
          "tuning_frames": [f"{s.key}/{os.path.basename(s.frames[i])}" for s, i in tune]}

    # ------------------------------------------- 1. calibration-only quality
    print("\n[1] vertical disparity of matched SIFT features (px), CALIBRATION ONLY")
    raw_dy = []
    for s, i in ev:
        e = epipolar_errors(read(s.frames[i]), read(right_path(s.frames[i])), dx_range=(-160.0, 260.0))
        if e.get("n_matches"):
            raw_dy.append(e["dy_mean"])
    R_["raw_dy_mean_over_pairs"] = float(np.mean(raw_dy))
    print(f"    unrectified: mean |dy| averaged over pairs = {R_['raw_dy_mean_over_pairs']:.3f}")
    var_res = {}
    for dist, rot in [("opencv", "as_is"), ("opencv", "inverse"), ("none", "as_is"), ("none", "inverse")]:
        per = []
        for s, i in ev:
            _, _, Lr, Rr, _ = load_pair(s, i, a.alpha, dist, rot)
            e = epipolar_errors(Lr, Rr)
            e["seq"] = s.key; e["frame"] = os.path.basename(s.frames[i])
            per.append(e)
        var_res[f"{dist}/{rot}"] = dict(summary=epi_summary(per), per_pair=per)
        v = var_res[f"{dist}/{rot}"]["summary"]
        print(f"    distortion={dist:6s} rotation={rot:8s}: mean|dy|={v['mean_of_means']:.3f} median={v['pooled_median']:.3f} "
              f"p95={v['p95_mean']:.2f} worst pair={v['worst_pair_mean']:.2f}  pairs with mean>3px: {v['n_pairs_mean_gt_3px']}/{v['n_pairs']}")
    best = min(var_res, key=lambda k: var_res[k]["summary"]["mean_of_means"])
    dist, rot = best.split("/")
    R_["conventions"] = {k: v["summary"] for k, v in var_res.items()}
    R_["chosen_convention"] = best
    R_["epipolar_calibration_only"] = dict(summary=var_res[best]["summary"], per_pair=var_res[best]["per_pair"])
    print(f"    -> convention chosen by the data: distortion={dist}, rotation={rot}")

    # ------------------------------------- 2. per-sequence self-calibration
    print(f"\n[2] per-sequence vertical self-calibration from frames {a.refine_frames} (images only)")
    corrs = {}
    for s in seqs:
        rc = get_rectifier(calibration_path(s.frame_dir), a.alpha, dist, rot)
        corrs[s.key] = estimate_vertical_correction(rc, [s.frames[i] for i in a.refine_frames if i < len(s)])
        c = corrs[s.key]
        print(f"    {s.key:11s} N={c.n_matches:4d}  a={c.a:6.2f} b={c.b:6.2f} c={c.c:6.2f}  MAD={c.mad:.2f}  "
              f"max|dy|={c.max_abs_px:5.2f}px  {'APPLIED' if c.applied else 'identity (below noise)'}")
    R_["vertical_corrections"] = {k: c.to_dict() for k, c in corrs.items()}
    per = []
    for s, i in ev:
        _, _, Lr, Rr, _ = load_pair(s, i, a.alpha, dist, rot, corrs[s.key])
        e = epipolar_errors(Lr, Rr)
        e["seq"] = s.key; e["frame"] = os.path.basename(s.frames[i])
        per.append(e)
    R_["epipolar_refined"] = dict(summary=epi_summary(per), per_pair=per)
    v0, v1 = R_["epipolar_calibration_only"]["summary"], R_["epipolar_refined"]["summary"]
    print(f"    EVALUATION frames (disjoint from the estimation frames):\n"
          f"      calibration only : mean|dy|={v0['mean_of_means']:.3f} median={v0['pooled_median']:.3f} "
          f"p95={v0['p95_mean']:.2f} worst={v0['worst_pair_mean']:.2f} pairs>3px={v0['n_pairs_mean_gt_3px']}\n"
          f"      + self-calibration: mean|dy|={v1['mean_of_means']:.3f} median={v1['pooled_median']:.3f} "
          f"p95={v1['p95_mean']:.2f} worst={v1['worst_pair_mean']:.2f} pairs>3px={v1['n_pairs_mean_gt_3px']}")

    # --------------------------------------------------------- 3. SGBM tuning
    print("\n[3] SGBM parameter search on 5 training frames, geometry = calibration + self-calibration "
          f"(criterion fixed in advance: LR-consistent AND residual < {RESIDUAL_OK:.0f} gray levels)")
    tune_data = [load_pair(s, i, a.alpha, dist, rot, corrs[s.key]) for s, i in tune]
    grid_rows = []
    for block, uniq, speckle in itertools.product((3, 5, 9, 11, 15), (0, 3, 5, 10), ((0, 2), (150, 2))):
        p = SGBMParams(block=block, uniqueness=uniq, speckle_window=speckle[0], speckle_range=speckle[1])
        t0 = time.time(); ver, lrf, vf = [], [], []
        for _, _, Lr, Rr, rc in tune_data:
            dl, dr = disparity_pair(Lr, Rr, p)
            ok = lr_consistent(dl, dr)
            res = photometric_residual(Lr, Rr, dl)
            area = rc.valid_left
            ver.append(float((ok & (res < RESIDUAL_OK) & area).sum() / area.sum()))
            lrf.append(float((ok & area).sum() / area.sum()))
            vf.append(float((np.isfinite(dl) & area).sum() / area.sum()))
        row = dict(block=block, uniqueness=uniq, speckle_window=speckle[0], speckle_range=speckle[1],
                   verified=float(np.mean(ver)), lr_pass=float(np.mean(lrf)), valid=float(np.mean(vf)),
                   sec_per_frame=(time.time() - t0) / len(tune_data))
        grid_rows.append(row)
        print(f"    block={block:2d} uniq={uniq:2d} speckle={speckle[0]:3d}: verified={row['verified']:.3f} "
              f"LR={row['lr_pass']:.3f} valid={row['valid']:.3f}  {row['sec_per_frame']:.1f}s/frame", flush=True)
    top = max(grid_rows, key=lambda r: r["verified"])
    chosen = SGBMParams(block=top["block"], uniqueness=top["uniqueness"],
                        speckle_window=top["speckle_window"], speckle_range=top["speckle_range"])
    edge = []
    if top["block"] in (3, 15): edge.append("block")
    if top["uniqueness"] in (0, 10): edge.append("uniqueness")
    print(f"    -> chosen: block={chosen.block} uniqueness={chosen.uniqueness} speckle={chosen.speckle_window} "
          f"(verified {top['verified']:.3f}){'  [WARNING: optimum on the grid edge for ' + ', '.join(edge) + ']' if edge else ''}")
    R_["sgbm_grid"], R_["sgbm_chosen"], R_["sgbm_optimum_on_grid_edge"] = grid_rows, chosen.__dict__, edge

    # ------------------------------------------------------------ 4. evaluation
    print("\n[4a] disparity validity, CALIBRATION ONLY (same SGBM settings)")
    R_["eval_calibration_only"] = evaluate(ev, chosen, a.alpha, dist, rot, None, a.out, "calibonly")
    print("\n[4b] disparity validity, CALIBRATION + SELF-CALIBRATION")
    R_["eval_refined"] = evaluate(ev, chosen, a.alpha, dist, rot, corrs, a.out, "refined")

    # ------------------------------------------------------- stop conditions
    def stops_for(mode_key, epi):
        e = R_[mode_key]
        return {
            "mean vertical disparity after rectification > 3 px": epi["summary"]["mean_of_means"] > STOP_MEAN_DY,
            "fewer than 40% of pixels survive the LR check": e["overall"]["lr_pct"] / 100 < STOP_LR,
            "instrument pixels almost no valid disparity (<10% LR-consistent)":
                bool(e["instrument"] and e["instrument"]["lr_pct"] / 100 < STOP_INSTR),
        }
    R_["stop_conditions_tripped"] = {"calibration_only": stops_for("eval_calibration_only", R_["epipolar_calibration_only"]),
                                     "refined": stops_for("eval_refined", R_["epipolar_refined"])}
    R_["seconds_total"] = time.time() - t00
    with open(os.path.join(a.out, "stereo_check.json"), "w") as f:
        json.dump(R_, f, indent=1)

    print("\n[5] SUMMARY")
    for name, key, epi in (("calibration only", "eval_calibration_only", "epipolar_calibration_only"),
                           ("calibration + self-calibration", "eval_refined", "epipolar_refined")):
        e, s_ = R_[key], R_[epi]["summary"]
        print(f"  {name}:\n    vertical |dy| mean {s_['mean_of_means']:.2f} / median {s_['pooled_median']:.2f} / p95 {s_['p95_mean']:.2f} px; "
              f"pairs with mean>3px: {s_['n_pairs_mean_gt_3px']}/{s_['n_pairs']}\n"
              f"    valid {e['overall']['valid_pct']:.1f}%, LR-consistent {e['overall']['lr_pct']:.1f}% "
              f"(interior {e['overall']['lr_pct_interior']:.1f}%)\n"
              f"    instrument px LR {e['instrument']['lr_pct']:.1f}% (valid {e['instrument']['valid_pct']:.1f}%), "
              f"tissue px LR {e['tissue']['lr_pct']:.1f}% (valid {e['tissue']['valid_pct']:.1f}%)\n"
              f"    disparity percentiles {e['disparity_percentiles_lr_ok']}  frac>=190: {e['frac_disp_ge_190']:.4f}")
    for mode, d in R_["stop_conditions_tripped"].items():
        for k, v in d.items():
            print(f"  STOP CONDITION [{mode}] {'TRIPPED' if v else 'not tripped'}: {k}")
    print(f"  total {R_['seconds_total']:.0f}s; wrote {a.out}/stereo_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
