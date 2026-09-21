#!/usr/bin/env python3
"""
Stage 2 — train the reliability estimator (report Sec. IV-C, Sec. V-D).

The base model stays frozen.  The estimator regresses the segmentation quality
it will achieve, from four observable indicators only:

    r_t = sigma(f_theta([q_t, T_t, A_t, F_t]))        eq. (11)
    L_rel = (1/N) sum (r_t - r*_t)^2,  r*_t = IoU(S_t, Y_t)   eqs. (13), (30)

Training history regime (Sec. V-D):
  * first half  — teacher forcing: "reliable" past frames are those whose TRUE
                  IoU exceeded tau.  These features do not depend on the
                  estimator, so they are computed once and reused.
  * second half — scheduled sampling: a linearly growing fraction of past
                  reliability decisions is taken from the estimator's own
                  output, exposing it to its own error distribution.
  * evaluation  — reported under BOTH regimes, because the gap between them is
                  the exposure-bias mismatch the report asks to measure.

    python -m tools.train_stage2 --cache runs/cache --val-seqs 2 5 9 15 \
        --epochs 300 --memory-size 5 --tau-train 0.5 --out runs/stage2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.align import build_align  # noqa: E402
from sitr.losses import ReliabilityLoss  # noqa: E402
from sitr.metrics import pearson, roc_auc, spearman  # noqa: E402
from sitr.models import ReliabilityMLP, count_parameters  # noqa: E402
from sitr.reliability import (INDICATOR_SETS, SequenceCache,  # noqa: E402
                              StreamConfig, indicator_names, run_sequence,
                              sweep_threshold)
from sitr.utils import get_device, save_checkpoint, set_seed, write_json  # noqa: E402

# resolved at run time from --indicators


def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="output dir of cache_features.py")
    ap.add_argument("--val-seqs", nargs="+", type=int, default=[2, 5, 9, 15])
    ap.add_argument("--indicators", default="extended",
                    choices=sorted(INDICATOR_SETS),
                    help="paper4 = the report's four (eqs. 6-9); extended adds "
                         "entropy, margin, compactness, fragmentation and "
                         "short-term drift, all free from the same forward pass")
    ap.add_argument("--memory-size", type=int, default=5, help="K in eq. (15)")
    ap.add_argument("--tau-train", type=float, default=0.5,
                    help="quality threshold defining a 'reliable' past frame "
                         "while building history (distinct from the deployed "
                         "decision threshold, which is swept afterwards)")
    ap.add_argument("--theta", type=float, default=0.5,
                    help="acceptance quality level used to label a frame as a "
                         "genuine failure when sweeping tau")
    ap.add_argument("--align", default="identity", choices=["identity", "farneback"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--inner-steps", type=int, default=8,
                    help="minibatch updates per epoch over the pooled frames")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="regularises the estimator; worth trying when the "
                         "train/validation MSE gap is large, which it usually "
                         "is because the splits are different procedures")
    ap.add_argument("--sched-max", type=float, default=0.5,
                    help="maximum fraction of self-predicted history reached at "
                         "the end of training")
    ap.add_argument("--empty-both-iou", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/stage2")
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args()


def load_caches(cache_dir: str) -> List[SequenceCache]:
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"no .npz caches in {cache_dir}")
    return [SequenceCache.load(os.path.join(cache_dir, f)) for f in files]


def seq_index(name: str) -> int:
    """Sequence index from a cache key like 'train/seq_12'."""
    import re
    m = re.search(r"seq[_-]?(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def seq_split(name: str) -> str:
    return "test" if name.lower().startswith("test") else "train"


def collect(caches, cfg, predict_fn, sched_prob, rng, align) -> Dict[str, np.ndarray]:
    feats, targets, rhat, cold, degen = [], [], [], [], []
    for c in caches:
        out = run_sequence(c, cfg, predict_fn=predict_fn, sched_prob=sched_prob,
                           rng=rng, align=align)
        feats.append(out["feats"])
        targets.append(out["r_star"])
        rhat.append(out["r_hat"])
        cold.append(out["cold"])
        degen.append(out["degenerate"])
    return {
        "feats": np.concatenate(feats, 0),
        "r_star": np.concatenate(targets, 0),
        "r_hat": np.concatenate(rhat, 0),
        "cold": np.concatenate(cold, 0),
        "degenerate": np.concatenate(degen, 0),
    }


def make_predict_fn(model: torch.nn.Module, device):
    @torch.no_grad()
    def fn(feat_vec):
        x = torch.as_tensor(feat_vec, dtype=torch.float32,
                            device=device).view(1, -1)
        return float(model(x).item())
    return fn


def diagnostics(r_hat: np.ndarray, r_star: np.ndarray, theta: float) -> Dict:
    ok = np.isfinite(r_hat) & np.isfinite(r_star)
    r_hat, r_star = r_hat[ok], r_star[ok]
    if r_hat.size == 0:
        return {}
    bad = (r_star < theta).astype(np.int32)
    return {
        "n": int(r_hat.size),
        "mse": float(np.mean((r_hat - r_star) ** 2)),
        "mae": float(np.mean(np.abs(r_hat - r_star))),
        "bias": float(np.mean(r_hat - r_star)),
        "pearson_r": pearson(r_hat, r_star),
        "spearman_rho": spearman(r_hat, r_star),
        # falsification criterion 5 of the report: does r_t actually track IoU?
        "auc_detect_bad_frames": roc_auc(-r_hat, bad),
        "bad_frame_rate": float(bad.mean()),
        "mean_r_hat": float(r_hat.mean()),
        "mean_r_star": float(r_star.mean()),
    }


def maybe_plot(out_dir: str, val: Dict, sweep: Dict, history: List[Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (plots skipped: {e})")
        return

    ok = np.isfinite(val["r_hat"]) & np.isfinite(val["r_star"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].scatter(val["r_star"][ok], val["r_hat"][ok], s=8, alpha=0.45)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel(r"true quality $r^*_t$ = IoU$(S_t, Y_t)$")
    ax[0].set_ylabel(r"predicted reliability $r_t$")
    ax[0].set_title("Reliability calibration (val)")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)

    g = sweep["grid"]
    ax[1].plot([r["tau"] for r in g], [r["rho"] for r in g], label=r"$\rho$ routed")
    ax[1].plot([r["tau"] for r in g], [r["youden_j"] for r in g], label="Youden J")
    ax[1].axvline(sweep["tau"], color="r", ls="--", lw=1,
                  label=fr"$\tau$={sweep['tau']:.2f}")
    ax[1].set_xlabel(r"threshold $\tau$"); ax[1].legend()
    ax[1].set_title("Threshold sweep")

    ax[2].plot([h["epoch"] for h in history], [h["train_mse"] for h in history],
               label="train MSE")
    ax[2].plot([h["epoch"] for h in history], [h["val_mse"] for h in history],
               label="val MSE (self-predicted history)")
    ax[2].set_xlabel("epoch"); ax[2].set_yscale("log"); ax[2].legend()
    ax[2].set_title("Reliability loss (eq. 30)")

    fig.tight_layout()
    path = os.path.join(out_dir, "stage2_diagnostics.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> int:
    args = build_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    align = build_align(args.align)

    caches = load_caches(args.cache)
    # Test sequences are different procedures; they never enter train or val.
    held_out = [c for c in caches if seq_split(c.name) == "test"]
    caches = [c for c in caches if seq_split(c.name) == "train"]
    val_set = set(args.val_seqs)
    train_c = [c for c in caches if seq_index(c.name) not in val_set]
    val_c = [c for c in caches if seq_index(c.name) in val_set]
    if not train_c or not val_c:
        raise ValueError(f"empty split: sequences={[c.name for c in caches]}, "
                         f"val={sorted(val_set)}")
    if held_out:
        print(f"held-out test caches (not used for fitting): "
              f"{[c.name for c in held_out]}")

    # Two caches for the same sequence index mean a stale file is present; the
    # same frames would land in both train and val and the split would leak.
    seen = {}
    for c in caches:
        i = seq_index(c.name)
        if i in seen:
            raise SystemExit(
                f"sequence index {i} appears twice in {args.cache!r}: "
                f"{seen[i]!r} and {c.name!r}. This is almost always a stale "
                "cache from an earlier run — the same frames would enter both "
                "the train and validation splits. Re-run cache_features.py "
                "with --overwrite, or use an empty --out.")
        seen[i] = c.name
    print(f"train: {[c.name for c in train_c]}")
    print(f"val  : {[c.name for c in val_c]}")
    print(f"frames: train={sum(len(c) for c in train_c)} "
          f"val={sum(len(c) for c in val_c)}  align={align.name}")

    FEATURE_NAMES = indicator_names(args.indicators)
    for c in caches:
        c.require(args.indicators)
    print(f"indicator set '{args.indicators}' ({len(FEATURE_NAMES)}): "
          f"{list(FEATURE_NAMES)}")

    cfg = StreamConfig(memory_size=args.memory_size, tau=args.tau_train,
                       empty_both_iou=args.empty_both_iou,
                       indicators=args.indicators)

    model = ReliabilityMLP(in_dim=len(FEATURE_NAMES), hidden=args.hidden,
                           dropout=args.dropout).to(device)
    print(f"reliability MLP params: {count_parameters(model):,}")
    criterion = ReliabilityLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # Teacher-forced features do not depend on the estimator -> compute once.
    t0 = time.time()
    tf_train = collect(train_c, cfg, None, 0.0, rng, align)
    tf_val = collect(val_c, cfg, None, 0.0, rng, align)
    print(f"teacher-forced features cached in {time.time()-t0:.2f}s")

    Xtf = torch.from_numpy(tf_train["feats"]).to(device)
    Ytf = torch.from_numpy(tf_train["r_star"]).to(device)

    half = max(1, args.epochs // 2)
    history: List[Dict] = []
    best_val = float("inf")

    for epoch in range(args.epochs):
        # scheduled sampling ramp over the second half of training
        if epoch < half:
            sched_prob = 0.0
            X, Y = Xtf, Ytf
        else:
            sched_prob = args.sched_max * (epoch - half + 1) / max(1, args.epochs - half)
            model.eval()
            data = collect(train_c, cfg, make_predict_fn(model, device),
                           sched_prob, rng, align)
            X = torch.from_numpy(data["feats"]).to(device)
            Y = torch.from_numpy(data["r_star"]).to(device)

        model.train()
        n = X.shape[0]
        losses = []
        for _ in range(args.inner_steps):
            idx = torch.randint(0, n, (min(args.batch_size, n),), device=device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(X[idx]), Y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            # inference regime: history comes entirely from the estimator itself
            v = collect(val_c, cfg, make_predict_fn(model, device), 1.0, rng, align)
            d = diagnostics(v["r_hat"], v["r_star"], args.theta)
            row = {"epoch": epoch, "sched_prob": float(sched_prob),
                   "train_mse": float(np.mean(losses)), "val_mse": d.get("mse", 0.0),
                   "val_pearson": d.get("pearson_r", float("nan")),
                   "val_auc": d.get("auc_detect_bad_frames", float("nan"))}
            history.append(row)
            print(f"[{epoch:4d}/{args.epochs}] p_sched={sched_prob:.2f} "
                  f"train MSE={row['train_mse']:.5f}  val MSE={row['val_mse']:.5f}  "
                  f"r={row['val_pearson']:.3f}  AUC={row['val_auc']:.3f}")
            if row["val_mse"] < best_val:
                best_val = row["val_mse"]
                save_checkpoint(os.path.join(args.out, "reliability_best.pt"), model,
                                {"args": vars(args), "epoch": epoch,
                                 "feature_names": list(FEATURE_NAMES),
                                 "diagnostics": d})

    # ------------------------------------------------------------------ final
    # The RELEASE ships reliability_best.pt, so the report must describe THAT
    # model, not whatever the last epoch happened to leave in memory.  Reload
    # it before computing the final diagnostics and the threshold sweep.
    # Skipping this reload is how a release ends up shipping one estimator
    # while its stage2_report.json describes another: tau and the routing rate
    # rho then belong to a model nobody runs, and tools/verify_release.py
    # correctly refuses the release.
    best_path = os.path.join(args.out, "reliability_best.pt")
    if os.path.isfile(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"\nreloaded {os.path.basename(best_path)} "
              f"(epoch {ck['meta'].get('epoch', '?')}) for the final report — "
              "the report now describes the model that ships")
    else:
        print("\n! no reliability_best.pt; reporting the final-epoch model")

    model.eval()
    pred_fn = make_predict_fn(model, device)
    val_self = collect(val_c, cfg, pred_fn, 1.0, rng, align)
    val_teacher = collect(val_c, cfg, pred_fn, 0.0, rng, align)
    train_self = collect(train_c, cfg, pred_fn, 1.0, rng, align)

    d_self = diagnostics(val_self["r_hat"], val_self["r_star"], args.theta)
    d_teacher = diagnostics(val_teacher["r_hat"], val_teacher["r_star"], args.theta)
    d_train = diagnostics(train_self["r_hat"], train_self["r_star"], args.theta)
    sweep = sweep_threshold(val_self["r_hat"], val_self["r_star"], theta=args.theta)

    feats = val_self["feats"]
    feat_corr = {name: pearson(feats[:, i], val_self["r_star"])
                 for i, name in enumerate(FEATURE_NAMES)}

    train_val_gap = (d_train.get("mse", float("nan"))
                     - d_self.get("mse", float("nan")))
    report = {
        "args": vars(args),
        # Which checkpoint every number below describes. run_all.py ships this
        # same file, and tools/verify_release.py replays it -- if these ever
        # disagree again, the routing-rate check will catch it.
        "reported_checkpoint": ("reliability_best.pt" if os.path.isfile(best_path)
                                else "final_epoch"),
        "train_minus_val_mse": train_val_gap,
        "indicator_set": args.indicators,
        "indicator_names": list(FEATURE_NAMES),
        "val_self_predicted_history": d_self,
        "val_teacher_forced_history": d_teacher,
        "train_self_predicted_history": d_train,
        "exposure_bias_mse_gap": (d_self.get("mse", 0.0)
                                  - d_teacher.get("mse", 0.0)),
        "threshold": sweep,
        "indicator_correlation_with_r_star": feat_corr,
        "frac_cold_start_frames": float(val_self["cold"].mean()),
        "frac_degenerate_frames": float(val_self["degenerate"].mean()),
        "history": history,
    }
    write_json(os.path.join(args.out, "stage2_report.json"), report)
    save_checkpoint(os.path.join(args.out, "reliability_last.pt"), model,
                    {"args": vars(args), "feature_names": list(FEATURE_NAMES),
                     "tau": sweep["tau"], "diagnostics": d_self})

    print("\n=== Stage 2 summary (validation, self-predicted history) ===")
    for k in ("mse", "mae", "pearson_r", "spearman_rho", "auc_detect_bad_frames"):
        print(f"  {k:<24} {d_self.get(k, float('nan')):.4f}")
    print(f"  exposure-bias MSE gap    {report['exposure_bias_mse_gap']:+.5f}")
    if d_train.get("mse") and d_self.get("mse"):
        ratio = d_self["mse"] / max(d_train["mse"], 1e-9)
        print(f"  val/train MSE ratio      {ratio:.2f}"
              f"{'   <- overfitting; try --dropout 0.2' if ratio > 3 else ''}")
    print(f"  provisional tau          {sweep['tau']:.2f}  "
          f"(routes rho={[r['rho'] for r in sweep['grid'] if abs(r['tau']-sweep['tau'])<1e-9][0]:.3f} "
          f"of frames to recovery)")
    print("  indicator correlation with r* (which ones carry signal):")
    for k, v in sorted(feat_corr.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:<20} {v:+.3f}")
    print(f"\n  cold-start frames {report['frac_cold_start_frames']:.3f}   "
          f"degenerate (empty mask) frames {report['frac_degenerate_frames']:.3f}")

    if not args.no_plots:
        maybe_plot(args.out, val_self, sweep, history)
    print(f"\nWrote {os.path.join(args.out, 'stage2_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
