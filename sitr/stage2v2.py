"""
Stage 2, version 2: reliability estimation trained on HONEST errors.

What changed from v1, and why each change is needed rather than cosmetic:

1. Cross-fitted training data (the correctness fix).
   v1 trained the estimator on Stage 1's predictions for the very sequences
   Stage 1 was trained on. Stage 1 fits those sequences far better than unseen
   ones, so the estimator learned from unrealistically good examples -- the
   likely cause of the 4-5x validation/training error gap. v2 uses
   out-of-fold (OOF) predictions: every training sequence is predicted by a
   Stage 1 model that never saw it. This is standard stacking practice and
   the same protocol used to train segmentation quality estimators in the
   prediction-quality literature (e.g. MetaSeg, Rottmann et al.).

2. The meta-model family is SELECTED, not assumed.
   Candidates: the v1 MLP, and gradient-boosted trees (the model family
   MetaSeg-style work finds strongest on these tabular per-frame metrics).
   Selection is by leave-fold-out AUC for detecting bad frames under
   self-predicted history, MSE breaking ties. The criterion is fixed in code
   before any number is seen.

3. The routing threshold comes with a stated risk guarantee.
   v1 chose tau by Youden's J, which has no operational meaning. v2 uses
   conformal risk control (Angelopoulos et al., "Conformal Risk Control",
   ICLR 2024): given a risk budget alpha, it picks the smallest tau such that

        E[ 1{ frame accepted as reliable  AND  actually bad } ]  <=  alpha

   i.e. "at most alpha of all frames are bad frames that slip through
   unrepaired". alpha is a REQUIREMENT the user states, not a fitted number;
   the report shows the whole alpha -> (tau, routing rate) trade-off.
   Caveat, stated rather than hidden: the guarantee assumes exchangeable
   frames; video frames are correlated, so the bound is verified empirically
   on the official test sequences rather than assumed.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .metrics import pearson, roc_auc, spearman
from .models import ReliabilityMLP
from .reliability import SequenceCache, StreamConfig, run_sequence


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def make_folds(items: Sequence[Tuple[str, int]], k: int) -> List[List[str]]:
    """Split sequences into k folds of near-equal FRAME count, deterministically.

    items: (sequence key, number of frames). A sequence is never split: frames
    of one surgery are near-duplicates at 1 Hz, so splitting inside a sequence
    would leak.
    """
    if k < 2:
        raise ValueError("need at least 2 folds")
    if k > len(items):
        raise ValueError(f"{k} folds but only {len(items)} sequences")
    folds: List[List[str]] = [[] for _ in range(k)]
    load = [0] * k
    for key, n in sorted(items, key=lambda kv: (-int(kv[1]), kv[0])):
        j = min(range(k), key=lambda i: (load[i], len(folds[i]), i))
        folds[j].append(key)
        load[j] += int(n)
    return [sorted(f) for f in folds]


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def diagnostics(r_hat: np.ndarray, r_star: np.ndarray, theta: float) -> Dict:
    ok = np.isfinite(r_hat) & np.isfinite(r_star)
    r_hat, r_star = r_hat[ok], r_star[ok]
    if r_hat.size == 0:
        return {}
    bad = (r_star < theta).astype(np.int32)
    return {"n": int(r_hat.size),
            "mse": float(np.mean((r_hat - r_star) ** 2)),
            "mae": float(np.mean(np.abs(r_hat - r_star))),
            "pearson_r": pearson(r_hat, r_star),
            "spearman_rho": spearman(r_hat, r_star),
            "auc_detect_bad_frames": roc_auc(-r_hat, bad),
            "bad_frame_rate": float(bad.mean())}


# --------------------------------------------------------------------------- #
# conformal risk control for the routing threshold
# --------------------------------------------------------------------------- #
def conformal_tau(r_hat: np.ndarray, r_star: np.ndarray, theta: float,
                  alpha: float, grid: Sequence[float] = None) -> Dict:
    """Smallest tau with  (n * Rhat(tau) + 1) / (n + 1) <= alpha.

    loss_i(tau) = 1{ r_hat_i >= tau and r_star_i < theta }   (bad, accepted)
    is non-increasing in tau, bounded by B = 1, which is exactly the setting
    of conformal risk control. Smallest admissible tau = least routing.
    """
    r_hat = np.asarray(r_hat, np.float64)
    r_star = np.asarray(r_star, np.float64)
    ok = np.isfinite(r_hat) & np.isfinite(r_star)
    r_hat, r_star = r_hat[ok], r_star[ok]
    n = r_hat.size
    bad = r_star < theta
    grid = np.round(np.arange(0.0, 1.0001, 0.005), 4) if grid is None else np.asarray(grid)
    rows = []
    chosen = None
    for tau in grid:
        accepted = r_hat >= tau
        risk = float(np.mean(accepted & bad)) if n else 1.0
        bound = (n * risk + 1.0) / (n + 1.0)
        row = {"tau": float(tau), "rho": float(1.0 - accepted.mean()) if n else 1.0,
               "empirical_risk": risk, "conformal_bound": bound}
        rows.append(row)
        if chosen is None and bound <= alpha:
            chosen = row
    feasible = chosen is not None
    if not feasible:                 # budget unattainable: route everything
        chosen = {"tau": 1.0 + 1e-9, "rho": 1.0, "empirical_risk": 0.0,
                  "conformal_bound": 1.0 / (n + 1.0)}
    return {"alpha": float(alpha), "theta": float(theta), "n": int(n),
            "tau": float(chosen["tau"]), "rho": float(chosen["rho"]),
            "empirical_risk": float(chosen["empirical_risk"]),
            "feasible": bool(feasible), "bad_frame_rate": float(bad.mean()) if n else 0.0,
            "criterion": "conformal risk control, loss = bad frame accepted",
            "grid": rows}


def realised_risk(r_hat, r_star, tau, theta) -> Dict:
    r_hat = np.asarray(r_hat, np.float64); r_star = np.asarray(r_star, np.float64)
    ok = np.isfinite(r_hat) & np.isfinite(r_star)
    r_hat, r_star = r_hat[ok], r_star[ok]
    acc = r_hat >= tau
    bad = r_star < theta
    return {"rho": float(1 - acc.mean()), "risk_bad_accepted": float(np.mean(acc & bad)),
            "bad_frames_caught": float((~acc & bad).sum() / max(bad.sum(), 1)),
            "good_frames_routed": float((~acc & ~bad).sum() / max((~bad).sum(), 1))}


# --------------------------------------------------------------------------- #
# history collection
# --------------------------------------------------------------------------- #
def collect(caches: Sequence[SequenceCache], cfg: StreamConfig,
            predict_fn: Callable = None, sched_prob: float = 0.0,
            seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {"feats": [], "r_star": [], "r_hat": [], "cold": [], "degenerate": []}
    for c in caches:
        o = run_sequence(c, cfg, predict_fn=predict_fn, sched_prob=sched_prob, rng=rng)
        for k in out:
            out[k].append(o[k])
    return {k: np.concatenate(v, 0) if v else np.zeros(0) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# meta-model families
# --------------------------------------------------------------------------- #
class MLPFamily:
    """The v1 estimator, trained for a FIXED number of epochs (no checkpoint
    selection on held-out data). Teacher forcing, then scheduled sampling."""

    name = "mlp"

    def __init__(self, epochs=300, lr=1e-3, weight_decay=1e-4, hidden=64,
                 dropout=0.2, batch_size=256, inner_steps=8, sched_max=0.5,
                 seed=0):
        self.__dict__.update(dict(epochs=epochs, lr=lr, weight_decay=weight_decay,
                                  hidden=hidden, dropout=dropout,
                                  batch_size=batch_size, inner_steps=inner_steps,
                                  sched_max=sched_max, seed=seed))
        self.model = None

    def predict_fn(self):
        m = self.model

        @torch.no_grad()
        def fn(v):
            return float(m(torch.as_tensor(v, dtype=torch.float32).view(1, -1)).item())
        return fn

    def fit(self, caches, cfg):
        torch.manual_seed(self.seed)
        in_dim = len(collect(caches[:1], cfg)["feats"][0])
        self.model = ReliabilityMLP(in_dim, self.hidden, self.dropout)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        tf = collect(caches, cfg, None, 0.0, self.seed)
        X, Y = torch.from_numpy(tf["feats"]), torch.from_numpy(tf["r_star"])
        half = max(1, self.epochs // 2)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            if ep >= half and (ep - half) % 10 == 0:     # refresh self-history
                p = self.sched_max * (ep - half + 1) / max(1, self.epochs - half)
                self.model.eval()
                d = collect(caches, cfg, self.predict_fn(), p, self.seed + ep)
                X, Y = torch.from_numpy(d["feats"]), torch.from_numpy(d["r_star"])
            self.model.train()
            for _ in range(self.inner_steps):
                idx = torch.randint(0, X.shape[0], (min(self.batch_size, X.shape[0]),),
                                    generator=g)
                opt.zero_grad(set_to_none=True)
                loss = torch.mean((self.model(X[idx]) - Y[idx]) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
        self.model.eval()
        return self

    def export(self):
        return {"kind": "mlp", "state_dict": self.model.state_dict(),
                "hidden": self.hidden, "dropout": self.dropout,
                "in_dim": self.model.in_dim}


class GBMFamily:
    """Gradient-boosted regression trees on the same indicators, trained with
    DAgger-style rounds so it also sees its own history at training time."""

    name = "gbm"

    def __init__(self, rounds=3, sched_max=0.5, max_iter=300, learning_rate=0.05,
                 max_leaf_nodes=15, min_samples_leaf=20, l2=1.0, seed=0):
        self.__dict__.update(dict(rounds=rounds, sched_max=sched_max,
                                  max_iter=max_iter, learning_rate=learning_rate,
                                  max_leaf_nodes=max_leaf_nodes,
                                  min_samples_leaf=min_samples_leaf, l2=l2, seed=seed))
        self.model = None

    def _new(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            loss="squared_error", max_iter=self.max_iter,
            learning_rate=self.learning_rate, max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2,
            early_stopping=False, random_state=self.seed)

    def predict_fn(self):
        m = self.model

        def fn(v):
            return float(np.clip(m.predict(np.asarray(v, np.float64).reshape(1, -1))[0],
                                 0.0, 1.0))
        return fn

    def fit(self, caches, cfg):
        d = collect(caches, cfg, None, 0.0, self.seed)
        Xs, Ys = [d["feats"]], [d["r_star"]]
        self.model = self._new().fit(np.concatenate(Xs), np.concatenate(Ys))
        for r in range(1, self.rounds + 1):
            p = self.sched_max * r / self.rounds
            d = collect(caches, cfg, self.predict_fn(), p, self.seed + r)
            Xs.append(d["feats"]); Ys.append(d["r_star"])
            self.model = self._new().fit(np.concatenate(Xs), np.concatenate(Ys))
        return self

    def export(self):
        return {"kind": "sklearn", "estimator": self.model}


FAMILIES = {"mlp": MLPFamily, "gbm": GBMFamily}


def cross_validate(caches_by_key: Dict[str, SequenceCache], folds: List[List[str]],
                   family: str, cfg: StreamConfig, theta: float,
                   family_kwargs: Dict = None, log=print) -> Dict:
    """Leave-one-fold-out: fit on the other folds, predict the held-out fold
    under SELF-PREDICTED history (the inference regime)."""
    r_hat, r_star = [], []
    for i, fold in enumerate(folds):
        tr = [caches_by_key[k] for f in folds if f is not fold for k in f]
        te = [caches_by_key[k] for k in fold]
        fam = FAMILIES[family](**(family_kwargs or {})).fit(tr, cfg)
        d = collect(te, cfg, fam.predict_fn(), 1.0, 1000 + i)
        r_hat.append(d["r_hat"]); r_star.append(d["r_star"])
        dd = diagnostics(d["r_hat"], d["r_star"], theta)
        log(f"      fold {i+1}/{len(folds)}  AUC={dd.get('auc_detect_bad_frames', float('nan')):.4f}"
            f"  MSE={dd.get('mse', float('nan')):.5f}")
    r_hat, r_star = np.concatenate(r_hat), np.concatenate(r_star)
    return {"r_hat": r_hat, "r_star": r_star,
            "diagnostics": diagnostics(r_hat, r_star, theta)}
