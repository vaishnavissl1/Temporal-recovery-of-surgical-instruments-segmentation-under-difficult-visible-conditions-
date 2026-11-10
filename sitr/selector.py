"""Predict-before-apply action selection WITH ABSTENTION (novelty items 2 and 3).

For a routed frame the candidate actions (sitr/actions.py) are A1 stereo, A2
temporal, A3 refine. One gradient-boosted regressor per action predicts the
change in instrument IoU that action would cause, from OBSERVABLE features only
(the gate indicators, the gate's reliability, the difficulty score and cheap
descriptors of what the action did). The selector then

    * takes the action with the highest predicted gain, and
    * ABSTAINS (keeps A0) when no action's predicted gain exceeds MARGIN.

Unlike JFS (judges AFTER refining, needs a reference mask, binary keep/revert),
the decision is made BEFORE applying anything and needs no reference mask.

Ground truth appears only as the TRAINING TARGET (`fit`'s delta arrays). The
inference-time methods `predict` / `choose` take features only.

MARGIN is a stated choice (the user-visible risk dial), kept in this one place.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

CANDIDATES = ("A1", "A2", "A3", "FUSE")
MARGIN = 0.01          # abstain unless the best predicted IoU gain exceeds this


class BenefitSelector:
    def __init__(self, margin: float = MARGIN, candidates: Sequence[str] = CANDIDATES,
                 max_iter: int = 200, learning_rate: float = 0.05, max_leaf_nodes: int = 8,
                 min_samples_leaf: int = 20, l2: float = 1.0, seed: int = 0):
        self.margin = float(margin)
        self.candidates = tuple(candidates)
        self.kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                       min_samples_leaf=min_samples_leaf, l2_regularization=l2,
                       early_stopping=False, random_state=seed)
        self.models: Dict[str, object] = {}
        self.fallback: Dict[str, float] = {}

    # ---- training: the ONLY place a ground-truth-derived number enters --------
    def fit(self, frame_feats: np.ndarray, action_feats: Dict[str, np.ndarray],
            delta: Dict[str, np.ndarray]) -> "BenefitSelector":
        """frame_feats [N,F]; action_feats[a] [N,G]; delta[a] [N] = measured IoU change
        (NaN where the action was unavailable). Rows with NaN are skipped."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        for a in self.candidates:
            X = np.concatenate([frame_feats, action_feats[a]], axis=1)
            y = np.asarray(delta[a], np.float64)
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            if ok.sum() < 40:                       # too little data: predict the mean, never guess more
                self.models[a] = None
                self.fallback[a] = float(y[ok].mean()) if ok.any() else 0.0
                continue
            self.models[a] = HistGradientBoostingRegressor(loss="squared_error", **self.kw).fit(X[ok], y[ok])
        return self

    # ---- inference: observable features only ---------------------------------
    def predict(self, frame_feats: np.ndarray, action_feats: Dict[str, Optional[np.ndarray]]
                ) -> Dict[str, np.ndarray]:
        """Predicted IoU change per action for N frames. NaN where the action is
        unavailable (its feature array is None or contains NaN)."""
        out = {}
        n = len(frame_feats)
        for a in self.candidates:
            g = action_feats.get(a)
            if g is None:
                out[a] = np.full(n, np.nan); continue
            X = np.concatenate([frame_feats, g], axis=1)
            ok = np.isfinite(X).all(1)
            p = np.full(n, np.nan)
            if self.models.get(a) is None:
                p[ok] = self.fallback.get(a, 0.0)
            elif ok.any():
                p[ok] = self.models[a].predict(X[ok])
            out[a] = p
        return out

    def choose(self, predicted: Dict[str, np.ndarray]) -> np.ndarray:
        """Per frame: the action with the highest predicted gain, or 'A0' (abstain)
        when the best predicted gain does not exceed `margin`."""
        names = list(self.candidates)
        M = np.stack([np.where(np.isfinite(predicted[a]), predicted[a], -np.inf) for a in names], 1)
        best = M.argmax(1)
        gain = M[np.arange(len(M)), best]
        return np.where(gain > self.margin, np.array(names, dtype=object)[best], "A0")


def policy_value(choice: np.ndarray, delta: Dict[str, np.ndarray]) -> np.ndarray:
    """Realised IoU change of a per-frame choice (A0 = 0). Unavailable action -> 0
    (it cannot be applied, so the frame is left alone)."""
    out = np.zeros(len(choice))
    for a, d in delta.items():
        m = choice == a
        out[m] = np.nan_to_num(np.asarray(d, float)[m], nan=0.0)
    return out


def oracle_choice(delta: Dict[str, np.ndarray], candidates: Sequence[str] = CANDIDATES) -> np.ndarray:
    """The truly best action per frame, or A0 when none helps. Needs ground truth:
    it is a CEILING for the report, never a deployable policy."""
    names = list(candidates)
    M = np.stack([np.nan_to_num(np.asarray(delta[a], float), nan=-np.inf) for a in names], 1)
    best = M.argmax(1)
    gain = M[np.arange(len(M)), best]
    return np.where(gain > 0.0, np.array(names, dtype=object)[best], "A0")
