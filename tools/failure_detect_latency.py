#!/usr/bin/env python3
"""
TASK C -- ms/frame of every failure detector, measured on this machine at batch 1.

    python -m tools.failure_detect_latency --work work_v2 --root <endovis2018_cvdataset> --release work_v2/release \
        --out results/taskC/latency.json

Each detector is timed end to end for ONE frame: every forward pass it needs plus the computation of its score, with the
input image already on the GPU (decode and resize are identical for every method and are excluded) and the GPU synchronised
before the clock stops. Frames are 40 random test frames (seed 0); each method is timed on every one after warm-up. The
numbers are measured here, not composed from a table; the gate rows add the measured cost of its indicators and estimator on
top of the one segmentation pass, and the cross-view row adds a second segmentation pass (the right image) and the measured
cost of the six indicators. Hardware is recorded in the output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr import crossview as XV  # noqa: E402
from sitr.dataset import discover_sequences  # noqa: E402
from sitr.metrics import instrument_iou  # noqa: E402
from sitr.reliability import cosine, mask_stability  # noqa: E402
from sitr.runtime import ReleaseBundle, _compact_frag  # noqa: E402
from sitr.stereo import VerticalCorrection, get_rectifier, right_path  # noqa: E402
from sitr.utils import cap_gpu_memory, get_device, set_seed  # noqa: E402
from tools.cache_right_view import load_model  # noqa: E402
from tools import failure_detect_collect as C  # noqa: E402


def sync():
    torch.cuda.synchronize()


def timed(fn, frames, warm=2):
    for f in frames[:warm]:
        fn(f)
    out = []
    for f in frames:
        sync()
        t0 = time.perf_counter()
        fn(f)
        sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(out)
    return dict(mean_ms=float(a.mean()), p95_ms=float(np.percentile(a, 95)), n_frames=int(len(a)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", default="results/taskC/latency.json")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    set_seed(0)
    dev = get_device("auto")
    cap_gpu_memory(3.3)
    folds = json.load(open(os.path.join(a.work, "folds.json")))["folds"]
    final, _ = load_model(os.path.join(a.work, "final", "best.pt"), dev)
    members = [load_model(os.path.join(a.work, "crossfit", f"fold{i}", "best.pt"), dev)[0] for i in range(len(folds))] + [final]
    seqs = [s for s in discover_sequences(a.root, require_labels=True, apply_repairs=True, splits=("test",))]
    rng = np.random.default_rng(0)
    picks = [(s, int(t)) for s in seqs for t in range(1, len(s.frames))]
    sel = [picks[i] for i in rng.choice(len(picks), a.n, replace=False)]
    frames = []
    for s, t in sel:
        x = C.load_image(s.frames[t], dev)
        xr = C.load_image(right_path(s.frames[t]), dev)
        prev = C.probs_of(final, C.load_image(s.frames[t - 1], dev))[0].argmax(0).cpu().numpy().astype(np.uint8)
        frames.append((s, t, x, xr, prev))
    bundle = ReleaseBundle.load(a.release)
    ms_ids = bundle.instrument_ids
    ids = [int(c) for c in ms_ids]

    @torch.no_grad()
    def single(f):
        return C.probs_of(final, f[2])[0]

    @torch.no_grad()
    def m_maxsoft(f):
        p = single(f); pred = p.argmax(0); fg = pred != 0
        return float(p.max(0).values[fg].mean()) if int(fg.sum()) else 0.0

    @torch.no_grad()
    def m_entropy(f):
        p = single(f); pred = p.argmax(0); fg = pred != 0
        ent = -(p * p.clamp_min(1e-8).log()).sum(0) / C.LOG_C
        return float(1 - ent[fg].mean()) if int(fg.sum()) else 0.0

    @torch.no_grad()
    def m_sdc(f):
        p = single(f); pred = p.argmax(0)
        return C.conf_scores(p, pred, None)[2]

    @torch.no_grad()
    def m_mc(f):
        C.set_stochastic_depth(final, True)
        try:
            ms = torch.stack([C.probs_of(final, f[2])[0].argmax(0) for _ in range(C.N_MC)], 0)
        finally:
            C.set_stochastic_depth(final, False)
        return C.pairwise_dice(ms)

    @torch.no_grad()
    def m_tta_ent(f):
        tp = C.tta_probs(final, f[2])
        avg = torch.stack([p[0] for p in tp], 0).mean(0)
        return C.mean_ent_conf(avg, avg.argmax(0), None)

    @torch.no_grad()
    def m_tta_dice(f):
        tp = C.tta_probs(final, f[2])
        return C.pairwise_dice(torch.stack([p[0].argmax(0) for p in tp], 0))

    @torch.no_grad()
    def m_ens(f):
        return C.pairwise_dice(torch.stack([C.probs_of(m, f[2])[0].argmax(0) for m in members], 0))

    @torch.no_grad()
    def m_gate(f):
        """One pass with features, the gate's indicators on GPU, the CPU ones (compactness, temporal, feature cosine),
        and the released estimator. Mirrors ReleaseBundle.stream."""
        with torch.autocast("cuda", enabled=True):
            logits, feats = final(f[2])
        T = bundle.temperature
        probs = torch.softmax(logits.float() / T, 1)
        conf, pred = probs.max(1)
        inst = torch.zeros_like(pred, dtype=torch.bool)
        for c in ids:
            inst |= pred == c
        fgm = inst.to(probs.dtype)
        den = fgm.sum((1, 2)).clamp_min(1e-6)
        q = ((conf * fgm).sum((1, 2)) / den).cpu().numpy()
        emb = ((feats.float() * fgm.unsqueeze(1)).sum((2, 3)) / den.unsqueeze(1)).cpu().numpy()[0]
        ent = -(probs * probs.clamp_min(1e-8).log()).sum(1) / C.LOG_C
        ent_b = (1 - (ent * fgm).sum((1, 2)) / den).cpu().numpy()
        top2 = probs.topk(2, 1).values
        mar = (((top2[:, 0] - top2[:, 1]) * fgm).sum((1, 2)) / den).cpu().numpy()
        p = pred[0].cpu().numpy().astype(np.uint8)
        fg = inst[0].cpu().numpy()
        cmp_, frg = _compact_frag(p, fg)
        sm = cv2.resize(p, (320, 256), interpolation=cv2.INTER_NEAREST)
        sp = cv2.resize(f[4], (320, 256), interpolation=cv2.INTER_NEAREST)
        Tt = instrument_iou(sm, sp, bundle.num_classes, ids)
        At = mask_stability(float(fg.sum()), float(fg.sum()))
        Ft = cosine(emb, emb)
        vec = np.array([q[0], Tt, At, Ft, ent_b[0], mar[0], cmp_, frg, cosine(emb, emb)], np.float32)
        return float(bundle.reliability(torch.from_numpy(vec).view(1, -1).to(dev)).item())

    # cross-view cost: right pass + the six indicators on the already-computed masks
    meta_cache = {}

    def rect_of(s):
        key = s.key
        if key not in meta_cache:
            sd = os.path.join(a.work, "stereo", f"{s.split}_{s.name}", "stereo_meta.json")
            m = json.load(open(sd))
            dist, rot = m["convention"]
            corr = VerticalCorrection(**{k: m["correction"][k] for k in ("a", "b", "c", "n_matches", "mad", "max_abs_px", "applied")}) if m.get("correction") else None
            meta_cache[key] = (get_rectifier(m["calibration_file"], m["alpha"], dist, rot), corr)
        return meta_cache[key]

    @torch.no_grad()
    def m_gate_xv(f):
        m_gate(f)
        pl = C.probs_of(final, f[2])[0].argmax(0).cpu().numpy().astype(np.uint8)
        pr = C.probs_of(final, f[3])[0].argmax(0).cpu().numpy().astype(np.uint8)
        rect, corr = rect_of(f[0])
        XV.crossview_indicators(cv2.resize(pl, (320, 256), interpolation=cv2.INTER_NEAREST),
                                cv2.resize(pr, (320, 256), interpolation=cv2.INTER_NEAREST), rect, corr, ids)

    # the six indicators alone (CPU), to show the cost they add beyond the second forward pass
    pre = []
    for f in frames:
        pl = C.probs_of(final, f[2])[0].argmax(0).cpu().numpy().astype(np.uint8)
        pr = C.probs_of(final, f[3])[0].argmax(0).cpu().numpy().astype(np.uint8)
        pre.append((cv2.resize(pl, (320, 256), interpolation=cv2.INTER_NEAREST), cv2.resize(pr, (320, 256), interpolation=cv2.INTER_NEAREST)))
    for i, f in enumerate(frames):
        rect_of(f[0])
    t_xv = []
    for i, f in enumerate(frames):
        rect, corr = rect_of(f[0])
        t0 = time.perf_counter()
        XV.crossview_indicators(pre[i][0], pre[i][1], rect, corr, ids)
        t_xv.append((time.perf_counter() - t0) * 1000.0)

    R = {"hardware": torch.cuda.get_device_name(0), "batch": 1, "input": "512x640, image already on the GPU, fp16 autocast",
         "method": "wall clock with the GPU synchronised before and after, mean and P95 over the frames; measured, not composed",
         "n_frames": len(frames), "frames": [f"{s.key}:{t}" for s, t, *_ in frames], "detectors": {}}
    for name, fn in [("max-softmax", m_maxsoft), ("mean entropy", m_entropy), ("Soft Dice Confidence", m_sdc),
                     ("MC stochastic depth (10 passes)", m_mc), ("TTA entropy (6 passes)", m_tta_ent),
                     ("TTA pairwise Dice (6 passes)", m_tta_dice), ("4-model ensemble pairwise Dice", m_ens),
                     ("gate (segmentation pass + indicators + estimator)", m_gate),
                     ("gate + cross-view (two passes + six indicators)", m_gate_xv)]:
        R["detectors"][name] = timed(fn, frames)
        print(f"  {name:55s} {R['detectors'][name]['mean_ms']:8.1f} ms  (P95 {R['detectors'][name]['p95_ms']:.1f})", flush=True)
    R["crossview_indicators_only_cpu_ms"] = dict(mean_ms=float(np.mean(t_xv)), p95_ms=float(np.percentile(t_xv, 95)))
    R["peak_gpu_gb"] = float(torch.cuda.max_memory_allocated() / 2 ** 30)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
