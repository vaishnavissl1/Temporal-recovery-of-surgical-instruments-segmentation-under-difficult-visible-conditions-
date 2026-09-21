"""
Stage 2 — the reliability assessment module.

A small estimator maps per-frame indicators to a single score:

    r_t = sigma(f_theta(indicators))          eq. (11)

trained to regress the segmentation quality actually achieved,

    r*_t = IoU(S_t, Y_t)                      eq. (13)

and needing NO ground truth at inference time.

Two indicator sets are available.

`paper4` — the report's four, eqs. (6)-(9):
    q      prediction confidence: mean max-posterior over the instrument foreground
    T      temporal consistency: IoU against the last reliable mask, warped
    A      mask stability: area change against the last reliable frame
    F      feature consistency: cosine of E_t against the reliable-memory mean

`extended` (default) — those four plus five more, every one of which is
computed from quantities the SAME forward pass already produced, so c_rel in
eq. (2) stays negligible:
    H      confidence from normalised entropy over the whole posterior, not
           just its maximum. A frame where the model is split 0.5/0.45 between
           two instrument classes has a high max-probability but is not
           confident; q cannot see that and H can.
    M      top-1 minus top-2 margin. Separates "confident and correct" from
           "confidently torn between two classes" — the failure that dominates
           shaft/wrist/clasper confusion.
    Bnd    1 - (boundary pixels / foreground pixels). A correct instrument mask
           is compact; a failing one shatters into ragged fragments and its
           perimeter-to-area ratio explodes. This is the single cheapest
           detector of the fragmentation failure mode.
    Frag   largest connected component / total foreground. Directly targets the
           "instrument segmented in pieces" mode that occlusion and blur cause,
           and which none of the report's four indicators measure.
    Drift  cosine of E_t against E_{t-1}, the IMMEDIATELY previous frame,
           reliable or not. F only compares against reliable memory, so during
           a run of unreliable frames it goes stale; Drift still sees the
           frame-to-frame appearance shock.

All indicators are oriented so that HIGHER MEANS MORE RELIABLE and live in
[0, 1]. That is not required by the MLP, but it makes the per-indicator
correlations in `stage2_report.json` directly readable.

Circularity and exposure bias (report Sec. V-D): T and A are defined relative to
the most recent *reliable* frame t'. During training that reliability comes from
the ground-truth quality (teacher forcing); a scheduled-sampling phase then
progressively substitutes the estimator's own predictions, so it meets its own
error distribution before deployment. Both regimes are implemented, and the gap
between them is reported rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .align import IdentityAlign
from .metrics import instrument_iou

EPS = 1e-6

#: name -> human-readable description, in the order the MLP receives them.
INDICATOR_SETS: Dict[str, Tuple[str, ...]] = {
    "paper4": ("q_confidence", "T_temporal", "A_area_stability",
               "F_feature_cos"),
    "extended": ("q_confidence", "T_temporal", "A_area_stability",
                 "F_feature_cos", "H_entropy_conf", "M_margin",
                 "Bnd_compactness", "Frag_largest_cc", "Drift_prev_cos"),
}

#: indicators that come straight from the cached per-frame scalars
_STATIC = {"q_confidence": "q", "H_entropy_conf": "ent",
           "M_margin": "margin", "Bnd_compactness": "bnd",
           "Frag_largest_cc": "frag"}


def indicator_names(name: str) -> Tuple[str, ...]:
    if name not in INDICATOR_SETS:
        raise ValueError(f"unknown indicator set {name!r}; "
                         f"choose from {sorted(INDICATOR_SETS)}")
    return INDICATOR_SETS[name]


# --------------------------------------------------------------------------- #
# per-sequence cache produced by tools/cache_features.py
# --------------------------------------------------------------------------- #
@dataclass
class SequenceCache:
    name: str
    q: np.ndarray                 # [T] eq. (6)
    area: np.ndarray              # [T] |Omega^fg_t|
    emb: np.ndarray               # [T,D] eq. (9)
    masks: np.ndarray             # [T,h,w] predicted labels (downsampled)
    r_star: np.ndarray            # [T] eq. (13)
    gray: Optional[np.ndarray] = None
    ent: Optional[np.ndarray] = None       # [T] entropy-based confidence
    margin: Optional[np.ndarray] = None    # [T] top1 - top2
    bnd: Optional[np.ndarray] = None       # [T] compactness
    frag: Optional[np.ndarray] = None      # [T] largest-cc share
    num_classes: int = 2
    instrument_ids: Tuple[int, ...] = (1,)

    def __len__(self) -> int:
        return int(self.q.shape[0])

    @property
    def has_extended(self) -> bool:
        return all(getattr(self, k) is not None
                   for k in ("ent", "margin", "bnd", "frag"))

    def require(self, indicators: str) -> None:
        if indicators != "paper4" and not self.has_extended:
            raise ValueError(
                f"cache {self.name!r} has only the four paper indicators. "
                "Re-run tools/cache_features.py (it computes the extended set "
                "by default), or train with --indicators paper4.")

    def save(self, path: str) -> None:
        payload = dict(
            name=np.array(self.name), q=self.q.astype(np.float32),
            area=self.area.astype(np.int64), emb=self.emb.astype(np.float16),
            masks=self.masks.astype(np.uint8),
            r_star=self.r_star.astype(np.float32),
            num_classes=np.array(self.num_classes),
            instrument_ids=np.array(self.instrument_ids, dtype=np.int64))
        if self.gray is not None:
            payload["gray"] = self.gray.astype(np.uint8)
        for k in ("ent", "margin", "bnd", "frag"):
            v = getattr(self, k)
            if v is not None:
                payload[k] = v.astype(np.float32)
        np.savez_compressed(path, **payload)

    @staticmethod
    def load(path: str) -> "SequenceCache":
        z = np.load(path, allow_pickle=False)
        get = lambda k: (z[k].astype(np.float32) if k in z.files else None)  # noqa: E731
        return SequenceCache(
            name=str(z["name"]), q=z["q"].astype(np.float32),
            area=z["area"].astype(np.int64), emb=z["emb"].astype(np.float32),
            masks=z["masks"], r_star=z["r_star"].astype(np.float32),
            gray=z["gray"] if "gray" in z.files else None,
            ent=get("ent"), margin=get("margin"), bnd=get("bnd"),
            frag=get("frag"),
            num_classes=int(z["num_classes"]),
            instrument_ids=tuple(int(v) for v in z["instrument_ids"]))


# --------------------------------------------------------------------------- #
# indicator maths
# --------------------------------------------------------------------------- #
def mask_stability(a_t: float, a_prev: float) -> float:
    """Eq. (8): A_t = exp(-|a_t - a_t'| / (a_t' + eps)) in (0, 1]."""
    return float(np.exp(-abs(float(a_t) - float(a_prev)) / (float(a_prev) + EPS)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def feature_consistency(e_t: np.ndarray, e_bar: np.ndarray) -> float:
    """Eq. (9): cosine similarity between E_t and the memory mean embedding."""
    return cosine(e_t, e_bar)


@dataclass
class StreamConfig:
    memory_size: int = 5            # K in eq. (15)
    tau: float = 0.5                # reliability threshold defining B_t
    empty_both_iou: float = 1.0
    indicators: str = "extended"


def run_sequence(cache: SequenceCache,
                 cfg: StreamConfig,
                 predict_fn=None,
                 sched_prob: float = 0.0,
                 rng: Optional[np.random.Generator] = None,
                 align=None) -> Dict[str, np.ndarray]:
    """Stream one sequence in temporal order and extract its indicators.

    predict_fn : callable(indicator_vector) -> r_hat in [0,1]; None disables
                 self-predicted history and leaves r_hat as NaN.
    sched_prob : probability that a past frame's reliability decision uses the
                 estimator's own r_hat rather than the ground-truth r*
                 (0 = pure teacher forcing, 1 = the inference-time regime).
    """
    cache.require(cfg.indicators)
    names = indicator_names(cfg.indicators)
    align = align or IdentityAlign()
    rng = rng or np.random.default_rng(0)
    T, K = len(cache), int(cfg.memory_size)
    D = len(names)

    feats = np.zeros((T, D), dtype=np.float32)
    r_hat = np.full(T, np.nan, dtype=np.float32)
    reliable = np.zeros(T, dtype=bool)
    used_pred = np.zeros(T, dtype=bool)
    cold = np.zeros(T, dtype=bool)
    degenerate = np.zeros(T, dtype=bool)

    buffer: List[int] = []            # indices of reliable frames, ascending

    for t in range(T):
        # eq. (15): keep only entries that are both recent and reliable
        while buffer and buffer[0] < t - K:
            buffer.pop(0)

        a_t = float(cache.area[t])
        vals: Dict[str, float] = {n: 0.0 for n in names}

        if a_t <= 0:
            # eq. (10): total mask collapse -> reliability-minimising values.
            # Every indicator is left at 0; there is no foreground to measure.
            degenerate[t] = True
        else:
            for n, key in _STATIC.items():
                if n in vals:
                    arr = getattr(cache, key)
                    vals[n] = float(arr[t]) if arr is not None else 0.0

            if "Drift_prev_cos" in vals and t > 0:
                vals["Drift_prev_cos"] = cosine(cache.emb[t], cache.emb[t - 1])

            if not buffer:
                # B_t empty: propagation impossible, T/A/F undefined (left 0)
                cold[t] = True
            else:
                tp = buffer[-1]                       # t' = most recent reliable
                gs = cache.gray[tp] if cache.gray is not None else None
                gt_ = cache.gray[t] if cache.gray is not None else None
                warped = align(cache.masks[tp], gs, gt_)              # eq. (16)
                vals["T_temporal"] = instrument_iou(
                    cache.masks[t], warped, cache.num_classes,
                    cache.instrument_ids, empty_value=cfg.empty_both_iou)  # (7)
                vals["A_area_stability"] = mask_stability(
                    a_t, float(cache.area[tp]))                          # (8)
                vals["F_feature_cos"] = feature_consistency(
                    cache.emb[t], cache.emb[buffer].mean(axis=0))        # (9)

        feats[t] = [vals[n] for n in names]

        if predict_fn is not None:
            r_hat[t] = float(predict_fn(feats[t]))

        # ---- the decision that shapes the FUTURE history -----------------
        take_pred = (predict_fn is not None and sched_prob > 0.0
                     and float(rng.random()) < float(sched_prob))
        score = float(r_hat[t]) if take_pred else float(cache.r_star[t])
        used_pred[t] = take_pred
        reliable[t] = score >= float(cfg.tau)
        if reliable[t]:
            buffer.append(t)

    return {"feats": feats, "r_hat": r_hat, "reliable": reliable,
            "used_pred": used_pred, "cold": cold, "degenerate": degenerate,
            "r_star": cache.r_star.astype(np.float32),
            "indicator_names": np.array(names)}


# --------------------------------------------------------------------------- #
# threshold selection
# --------------------------------------------------------------------------- #
def sweep_threshold(r_hat: np.ndarray, r_star: np.ndarray,
                    theta: float = 0.5,
                    grid: Optional[Sequence[float]] = None) -> Dict:
    """Choose tau on the validation split.

    Stage 4 is what finally fixes tau in the report: the value maximising
    event-level recovery success without degrading quality on reliable frames.
    That is not computable from Stages 1-2, so this returns a PROVISIONAL tau
    from the measurable proxy — the threshold best separating frames the base
    model actually got wrong (r* < theta) from the rest, by Youden's J — along
    with the routing rate rho of eq. (2) at every candidate. Re-sweep once the
    recovery module exists.
    """
    r_hat = np.asarray(r_hat, dtype=np.float64).ravel()
    r_star = np.asarray(r_star, dtype=np.float64).ravel()
    ok = np.isfinite(r_hat) & np.isfinite(r_star)
    r_hat, r_star = r_hat[ok], r_star[ok]

    bad = r_star < float(theta)
    grid = np.asarray(grid if grid is not None else np.arange(0.05, 0.96, 0.01))

    rows = []
    for tau in grid:
        flagged = r_hat < tau
        tp = int(np.sum(flagged & bad))
        fp = int(np.sum(flagged & ~bad))
        fn = int(np.sum(~flagged & bad))
        tn = int(np.sum(~flagged & ~bad))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        rows.append({"tau": float(tau), "rho": float(flagged.mean()),
                     "tpr": float(tpr), "fpr": float(fpr),
                     "youden_j": float(tpr - fpr),
                     "precision": float(tp / max(tp + fp, 1))})

    best = max(rows, key=lambda r: r["youden_j"]) if rows else {"tau": 0.5}
    return {"grid": rows, "tau": float(best["tau"]), "criterion": "youden_j",
            "theta": float(theta), "n_frames": int(r_hat.size),
            "bad_frame_rate": float(bad.mean()) if bad.size else 0.0}
