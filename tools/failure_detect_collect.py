#!/usr/bin/env python3
"""
TASK C, stage 1 -- per-frame failure-detection scores for every baseline, on the SAME frames.

    python -m tools.failure_detect_collect --work work_v2 --root <endovis2018_cvdataset> --out work_v2/failure_detection

For every labelled frame this runs the segmenter and computes, from the model's own output only
(no ground truth in any score; ground truth is read afterwards, to compute the RISK):

  ms_fg    max-softmax          mean top-class probability over predicted-foreground pixels
  ent_fg   mean entropy         1 - mean normalised predictive entropy over predicted-foreground pixels
  sdc_fg   Soft Dice Confidence per class c present in the prediction: 2*sum(p_c*y_c) / sum(p_c+y_c),
                                p_c = softmax probability map, y_c = the model's OWN hard mask; mean over classes
  mc_dice  MC stochastic-depth  10 stochastic passes, mean pairwise Dice between the discrete masks
  tta_ent  TTA, 6 passes        1 - mean entropy of the averaged probabilities over predicted-foreground pixels
  tta_dice TTA, 6 passes        mean pairwise Dice between the 6 discrete masks
  ens_dice 4-model ensemble     mean pairwise Dice between the 4 models' masks     (TEST ONLY, see below)

All scores are oriented HIGHER = MORE RELIABLE.

Stated choices (each fixed before any number was seen):
  * "predicted foreground" = every predicted pixel whose class is not 0 (background-tissue). This is the same set of
    classes the official metric scores (utils.py:172 never scores class 0), so the score and the risk look at the same
    pixels. The instrument-only variants (the repo's own q convention) are stored with the suffix _inst and are
    reported as secondary, never swapped in for the primary.
  * Softmax is taken at temperature 1 (raw). The repo's calibration temperature is a gate-side choice and the published
    baselines do not use it.
  * An empty predicted foreground gives score 0 (least reliable), matching eq. (10) of the repo's reliability module.
  * Pairwise Dice between two masks: for every class 1..11 present in EITHER mask, 2|A&B|/(|A|+|B|); the mean over those
    classes; two all-background masks give 1.0. The score is the mean over all pairs.
  * The segmenter has no Dropout layer. Its only stochastic layer is stochastic depth (DropPath, rate 0.1, the rate it
    was trained with). "MC-dropout" is therefore MC stochastic depth: DropPath modules are switched to train mode at
    inference, everything else stays in eval mode. This is a deviation from textbook MC-dropout and is stated in the table.
  * TTA = scales {0.75, 1.0, 1.25} x {no flip, horizontal flip}, six passes. Scaling is a bilinear resize of the
    512x640 network input, the logits are resized back bilinearly and un-flipped, then softmax; probabilities are averaged.
  * Honesty constraint: a training sequence is scored only by the cross-fitting model whose fold EXCLUDED it. The
    4-model ensemble (3 fold models + the final model) is valid only on the TEST sequences, because every training
    sequence was inside some member's training set. It is computed on test frames only.

The risk is 1 - official frame IoU of the single-pass mask at full 1280x1024 resolution (mask upsampled
nearest-neighbour; a class counts only if it is in the ground truth). It is the same risk for every detector.
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
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import IMAGENET_MEAN, IMAGENET_STD, discover_sequences  # noqa: E402
from sitr.metrics import official_class_ious  # noqa: E402
from sitr.utils import cap_gpu_memory, get_device, keep_awake, set_seed  # noqa: E402
from tools.cache_right_view import load_model  # noqa: E402
from tools.check_stereo import read_mask  # noqa: E402

NC = 12
H, W = 512, 640
INSTR = [1, 2, 3, 6, 7, 8, 9, 11]
SCALES = (0.75, 1.0, 1.25)
N_MC = 10
LOG_C = float(np.log(NC))
SCORE_KEYS = ("ms_fg", "ent_fg", "sdc_fg", "ms_inst", "ent_inst", "sdc_inst", "mc_dice", "mc_ent_fg",
              "tta_ent", "tta_dice", "ens_dice")


# ------------------------------------------------------------------ image
def load_image(path, device):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    x = ((img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return torch.from_numpy(x)[None].to(device).contiguous(memory_format=torch.channels_last)


# ------------------------------------------------------------------ forward passes
@torch.no_grad()
def probs_of(model, x):
    with torch.autocast("cuda", enabled=x.is_cuda):
        logits, _ = model(x)
    return torch.softmax(logits.float(), 1)                    # [B,12,H,W], temperature 1


def set_stochastic_depth(model, on: bool):
    """Switch only the DropPath modules to train mode (MC stochastic depth); everything else stays eval."""
    n = 0
    for m in model.modules():
        if type(m).__name__ == "DropPath" and m.drop_prob > 0:
            m.train(on)
            n += 1
    return n


@torch.no_grad()
def tta_probs(model, x):
    """Six passes. Returns the list of per-pass probability maps at the network resolution."""
    outs = []
    for s in SCALES:
        xs = x if s == 1.0 else F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        for flip in (False, True):
            xi = torch.flip(xs, dims=[3]) if flip else xs
            with torch.autocast("cuda", enabled=x.is_cuda):
                lg, _ = model(xi.contiguous(memory_format=torch.channels_last))
            lg = lg.float()
            if flip:
                lg = torch.flip(lg, dims=[3])
            if s != 1.0:
                lg = F.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(torch.softmax(lg, 1))
    return outs


# ------------------------------------------------------------------ scores
def fg_mask(pred, ids=None):
    if ids is None:
        return pred != 0
    m = torch.zeros_like(pred, dtype=torch.bool)
    for c in ids:
        m |= pred == c
    return m


def conf_scores(probs, pred, ids=None):
    """probs [12,H,W], pred [H,W]. Returns (max-softmax, 1-entropy, SDC) over the chosen foreground; 0 if empty."""
    fg = fg_mask(pred, ids)
    n = int(fg.sum())
    if n == 0:
        return 0.0, 0.0, 0.0
    top = probs.max(0).values
    ms = float(top[fg].mean())
    ent = -(probs * probs.clamp_min(1e-8).log()).sum(0) / LOG_C
    en = 1.0 - float(ent[fg].mean())
    cls = [c for c in (range(1, NC) if ids is None else ids) if (pred == c).any()]
    d = []
    for c in cls:
        p, y = probs[c], (pred == c).float()
        d.append(float(2.0 * (p * y).sum() / (p.sum() + y.sum()).clamp_min(1e-8)))
    return ms, en, float(np.mean(d)) if d else 0.0


def pairwise_dice(masks):
    """masks [N,H,W] long. Mean over pairs of the mean per-class Dice over classes present in either mask."""
    N = masks.shape[0]
    flat = masks.reshape(N, -1)
    A = torch.stack([(flat == c).float() for c in range(1, NC)], 0)         # [11,N,P]
    area = A.sum(2)                                                         # [11,N]
    inter = torch.bmm(A, A.transpose(1, 2))                                 # [11,N,N]
    vals = []
    for i, j in itertools.combinations(range(N), 2):
        tot = area[:, i] + area[:, j]
        present = tot > 0
        if not bool(present.any()):
            vals.append(1.0)
            continue
        dice = 2.0 * inter[:, i, j][present] / tot[present]
        vals.append(float(dice.mean()))
    return float(np.mean(vals))


def mean_ent_conf(probs, pred, ids=None):
    fg = fg_mask(pred, ids)
    if int(fg.sum()) == 0:
        return 0.0
    ent = -(probs * probs.clamp_min(1e-8).log()).sum(0) / LOG_C
    return 1.0 - float(ent[fg].mean())


# ------------------------------------------------------------------ one frame
@torch.no_grad()
def frame_scores(model, x, ensemble=None):
    out, masks = {}, {}
    p1 = probs_of(model, x)[0]
    base = p1.argmax(0)
    masks["single"] = base
    out["ms_fg"], out["ent_fg"], out["sdc_fg"] = conf_scores(p1, base, None)
    out["ms_inst"], out["ent_inst"], out["sdc_inst"] = conf_scores(p1, base, INSTR)

    # MC stochastic depth, 10 passes
    set_stochastic_depth(model, True)
    mc = [probs_of(model, x)[0] for _ in range(N_MC)]
    set_stochastic_depth(model, False)
    mc_masks = torch.stack([p.argmax(0) for p in mc], 0)
    out["mc_dice"] = pairwise_dice(mc_masks)
    mp = torch.stack(mc, 0).mean(0)
    out["mc_ent_fg"] = mean_ent_conf(mp, mp.argmax(0), None)

    # TTA, 6 passes
    tp = tta_probs(model, x)
    tm = torch.stack([p[0].argmax(0) for p in tp], 0)
    avg = torch.stack([p[0] for p in tp], 0).mean(0)
    masks["tta"] = avg.argmax(0)
    out["tta_ent"] = mean_ent_conf(avg, masks["tta"], None)
    out["tta_dice"] = pairwise_dice(tm)

    out["ens_dice"] = float("nan")
    if ensemble is not None:
        ps = [probs_of(m, x)[0] for m in ensemble]
        em = torch.stack([p.argmax(0) for p in ps], 0)
        out["ens_dice"] = pairwise_dice(em)
        masks["ensemble"] = torch.stack(ps, 0).mean(0).argmax(0)
        masks["_members"] = em
    return out, masks


def official_full_res(mask512, label_path):
    gt = read_mask(label_path)
    full = cv2.resize(mask512.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    v = official_class_ious(full, gt, NC)
    return (float(np.mean(list(v.values()))) if v else float("nan")), len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="+", default=["oof", "test"], choices=["oof", "test"])
    ap.add_argument("--limit-frames", type=int, default=0, help="debug: only the first N frames of each sequence")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these sequence keys, e.g. train/seq_1")
    ap.add_argument("--gpu-cap-gb", type=float, default=3.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    keep_awake()
    set_seed(a.seed)
    device = get_device("auto")
    cap_gpu_memory(a.gpu_cap_gb)
    os.makedirs(os.path.join(a.out, "scores"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    folds = json.load(open(os.path.join(a.work, "folds.json")))["folds"]
    seqs = discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("train", "test"))
    seqs = sorted(seqs, key=lambda s: (s.split != "train", s.index))
    t0, done = time.time(), 0

    def run_seq(s, model, ensemble, save_masks, tag):
        nonlocal done
        path = os.path.join(a.out, "scores", f"{s.split}__{s.name}.npz")
        if os.path.exists(path) and not a.limit_frames:
            print(f"  {s.key}: exists, skipping", flush=True)
            return
        n = len(s.frames) if not a.limit_frames else min(a.limit_frames, len(s.frames))
        rows = {k: [] for k in SCORE_KEYS}
        risk, nscored, mk = [], [], {"single": [], "tta": [], "ensemble": []}
        for t in range(n):
            x = load_image(s.frames[t], device)
            sc, masks = frame_scores(model, x, ensemble)
            for k in SCORE_KEYS:
                rows[k].append(sc[k])
            m1 = masks["single"].cpu().numpy().astype(np.uint8)
            off, ncls = official_full_res(m1, s.labels[t])
            risk.append(1.0 - off if np.isfinite(off) else float("nan"))
            nscored.append(ncls)
            if save_masks:
                for k in mk:
                    if k in masks:
                        mk[k].append(masks[k].cpu().numpy().astype(np.uint8))
            done += 1
            if t % 25 == 0:
                print(f"  {s.key:<12} {t+1}/{n}   {(time.time()-t0)/60:.1f} min total", flush=True)
        payload = {k: np.array(v, np.float32) for k, v in rows.items()}
        payload.update(risk=np.array(risk, np.float64), n_gt_classes=np.array(nscored), t=np.arange(n),
                       seq=np.array(s.key), model=np.array(tag))
        np.savez_compressed(path, **payload)
        if save_masks and not a.limit_frames:
            np.savez_compressed(os.path.join(a.out, "masks", f"{s.split}__{s.name}.npz"),
                                **{k: np.stack(v) for k, v in mk.items() if v})
        print(f"  {s.key:<12} {n} frames  model={tag}  mean risk {np.nanmean(risk):.4f}  "
              f"{(time.time()-t0)/60:.1f} min", flush=True)

    if "oof" in a.groups:
        print("OUT-OF-FOLD training sequences: each scored by the fold model that excluded it")
        for i, held in enumerate(folds):
            model, _ = load_model(os.path.join(a.work, "crossfit", f"fold{i}", "best.pt"), device)
            print(f" fold {i}: {held}  stochastic-depth modules: {set_stochastic_depth(model, False)}")
            for s in seqs:
                if s.key in held and (not a.only or s.key in a.only):
                    run_seq(s, model, None, False, f"fold{i}")
            del model
            torch.cuda.empty_cache()
    if "test" in a.groups:
        print("OFFICIAL TEST sequences: final model, plus the 4-model ensemble (valid on test only)")
        final, _ = load_model(os.path.join(a.work, "final", "best.pt"), device)
        members = [load_model(os.path.join(a.work, "crossfit", f"fold{i}", "best.pt"), device)[0] for i in range(len(folds))]
        ensemble = members + [final]
        for s in seqs:
            if s.split == "test" and (not a.only or s.key in a.only):
                run_seq(s, final, ensemble, True, "final")
    print(f"done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
