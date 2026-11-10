"""Candidate corrections for a routed frame (Task 5, novelty items 2 and 3).

    A0  keep            no change
    A1  stereo          use the right camera's view where the left prediction is unsure
    A2  temporal        the existing TemporalRecovery (propagation from a reliable past frame)
    A3  refine          class-conditioned morphological clean-up (thin vs region classes)

Every action maps a frame's observable context to a class-id mask. NOTHING here
reads ground truth: the context carries none, and tests/test_actions.py passes a
ground-truth object that raises on any access to prove it. Ground truth is used
only by `delta_scores`, which the TRAINING-data builder calls AFTER the masks
exist, to label how much each action helped.

All thresholds are stated choices, collected in ActionParams so they live in one
place and are visible in the write-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .metrics import frame_iou, official_frame_iou

ACTION_NAMES: Tuple[str, ...] = ("A0", "A1", "A2", "A3")


@dataclass
class ActionParams:
    # A1 stereo: replace a left label by the right view's label only where ALL hold
    p_low: float = 0.70           # left max-probability below this = left is unsure
    p_high: float = 0.80          # right-view confidence at least this
    stereo_conf_min: float = 0.50   # per-pixel stereo evidence confidence at least this
    # A3 refine
    min_region_px: int = 200      # islands of non-thin classes smaller than this are re-labelled
    min_thin_px: int = 10         # specks of thin classes smaller than this are removed
    thin_close: int = 3           # closing kernel for thin classes (bridges 1-px gaps)
    thin_grow_prob: float = 0.15  # closing may only add a pixel whose class probability is at least this
    mode_window: int = 15         # neighbourhood for re-labelling islands
    thin_names: Tuple[str, ...] = ("instrument-shaft", "thread", "suturing-needle")


@dataclass
class ActionContext:
    """Observable inputs for one frame. No ground truth field exists."""
    state: object                                   # FrameState (left view, memory, reliability, ...)
    evidence: Optional[Dict[str, np.ndarray]] = None   # stereo: lr_ok, occluded, confidence, in_view (left grid)
    right_label: Optional[np.ndarray] = None        # right prediction on the LEFT grid, 255 = none
    right_conf: Optional[np.ndarray] = None         # its confidence, uint8 0..255, on the LEFT grid
    target_names: Sequence[str] = ()
    instrument_ids: Sequence[int] = ()
    recovery: object = None                         # a TemporalRecovery (A2)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mode_fill(mask: np.ndarray, probs: Optional[np.ndarray], island: np.ndarray,
               exclude: np.ndarray, window: int, num_classes: int) -> np.ndarray:
    """Re-label the `island` pixels with the majority class of their neighbourhood,
    never choosing a class in `exclude` (the one being removed)."""
    out = mask.copy()
    best = np.full(mask.shape, -1.0, np.float32)
    lab = out.copy()
    for c in range(num_classes):
        if exclude[c]:
            continue
        cnt = cv2.boxFilter((mask == c).astype(np.float32), -1, (window, window), normalize=True)
        upd = cnt > best
        best[upd] = cnt[upd]
        lab[upd] = c
    out[island] = lab[island]
    return out


def thin_ids_from_names(target_names: Sequence[str], thin_names: Sequence[str]) -> List[int]:
    return [i for i, n in enumerate(target_names) if n in set(thin_names)]


# --------------------------------------------------------------------------- #
# the four actions
# --------------------------------------------------------------------------- #
def a0_keep(ctx: ActionContext, p: ActionParams = ActionParams()) -> np.ndarray:
    return ctx.state.mask.copy()


def a1_stereo(ctx: ActionContext, p: ActionParams = ActionParams()) -> Optional[np.ndarray]:
    """Right-view correction. None when no stereo information is available."""
    if ctx.right_label is None or ctx.right_conf is None or ctx.evidence is None:
        return None
    probs = ctx.state.probs
    left_conf = probs.max(axis=0)
    ev = ctx.evidence
    usable = (ev["lr_ok"] & ~ev["occluded"] & ev["in_view"]
              & (ev["confidence"].astype(np.float32) / 255.0 >= p.stereo_conf_min))
    use = (usable & (ctx.right_label != 255)
           & (ctx.right_conf.astype(np.float32) / 255.0 >= p.p_high)
           & (left_conf < p.p_low))
    out = ctx.state.mask.copy()
    out[use] = ctx.right_label[use]
    return out


def a2_temporal(ctx: ActionContext, p: ActionParams = ActionParams()) -> Optional[np.ndarray]:
    """The existing temporal recovery. None when it cannot fire (empty memory)."""
    if ctx.recovery is None or not getattr(ctx.state, "memory", None):
        return None
    return ctx.recovery(ctx.state).mask.copy()


def a3_refine(ctx: ActionContext, p: ActionParams = ActionParams()) -> np.ndarray:
    """Class-conditioned morphological clean-up. Uses neither stereo nor time.

    Thin classes (shaft, thread, needle): drop specks, bridge 1-px gaps, but only
    onto pixels where the model gave that class a non-trivial probability.
    Everything else (organs, large instrument parts): small islands are
    re-labelled with the majority class around them."""
    mask = ctx.state.mask
    probs = ctx.state.probs
    nc = probs.shape[0]
    thin = set(thin_ids_from_names(ctx.target_names, p.thin_names))
    out = mask.copy()
    # 1) islands of non-thin classes
    exclude = np.zeros(nc, bool)
    island = np.zeros(mask.shape, bool)
    for c in range(nc):
        if c in thin:
            continue
        m = (mask == c).astype(np.uint8)
        if not m.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = np.where(stats[1:, cv2.CC_STAT_AREA] < p.min_region_px)[0] + 1
        if small.size:
            isl = np.isin(lab, small)
            island |= isl
    if island.any():
        # a pixel in an island must not be re-labelled with its own class
        for c in range(nc):
            sel = island & (mask == c)
            if sel.any():
                ex = np.zeros(nc, bool); ex[c] = True
                out = _mode_fill(out, probs, sel, ex, p.mode_window, nc)
    # 2) thin classes
    # square kernel: OpenCV's 3x3 "ellipse" is a plus shape, which cannot bridge a
    # gap in a line that is one pixel thick (found by tests/test_actions.py)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.thin_close, p.thin_close))
    for c in thin:
        m = (out == c).astype(np.uint8)
        if not m.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        specks = np.where(stats[1:, cv2.CC_STAT_AREA] < p.min_thin_px)[0] + 1
        if specks.size:
            sp = np.isin(lab, specks)
            ex = np.zeros(nc, bool); ex[c] = True
            out = _mode_fill(out, probs, sp, ex, p.mode_window, nc)
            m = (out == c).astype(np.uint8)
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        add = (closed > 0) & (m == 0) & (probs[c] >= p.thin_grow_prob)
        out[add] = c
    return out


def _with_mask(ctx: ActionContext, mask: np.ndarray) -> ActionContext:
    """The same context but with `mask` as the frame's base prediction."""
    import copy
    st = copy.copy(ctx.state)
    st.mask = mask
    c = copy.copy(ctx)
    c.state = st
    return c


def a_fuse(ctx: ActionContext, p: ActionParams = ActionParams()) -> np.ndarray:
    """The LACOSTE-style baseline (M4): stereo AND temporal evidence fused on every
    frame, blindly. Temporal propagation first (when a memory exists), then the
    stereo correction applied on top of it. Never None: with neither source it is A0."""
    m2 = a2_temporal(ctx, p)
    base = m2 if m2 is not None else ctx.state.mask.copy()
    m1 = a1_stereo(_with_mask(ctx, base), p)
    return m1 if m1 is not None else base


def a_fuse_refine(ctx: ActionContext, p: ActionParams = ActionParams()) -> np.ndarray:
    """M5: the fused mask followed by the class-conditioned refinement."""
    return a3_refine(_with_mask(ctx, a_fuse(ctx, p)), p)


ACTIONS = {"A0": a0_keep, "A1": a1_stereo, "A2": a2_temporal, "A3": a3_refine}
BASELINES = {"FUSE": a_fuse, "FUSE_REFINE": a_fuse_refine}


def apply_all(ctx: ActionContext, p: ActionParams = ActionParams()) -> Dict[str, Optional[np.ndarray]]:
    """Every action's mask for this frame; None = the action is unavailable."""
    return {name: fn(ctx, p) for name, fn in ACTIONS.items()}


# --------------------------------------------------------------------------- #
# observable per-action features (for the selector)
# --------------------------------------------------------------------------- #
ACTION_FEATURE_NAMES = ("changed_frac", "changed_instr_frac", "mean_left_conf_changed",
                        "instr_area_ratio", "stereo_usable_frac", "right_agree_frac",
                        "memory_len", "memory_mean_reliability")


def action_features(ctx: ActionContext, mask_a: np.ndarray) -> np.ndarray:
    """Cheap observable descriptors of what the action did. No ground truth."""
    st = ctx.state
    base = st.mask
    changed = mask_a != base
    n = float(base.size)
    inst = np.isin(base, np.asarray(list(ctx.instrument_ids), dtype=base.dtype))
    left_conf = st.probs.max(axis=0)
    cf = float(changed.mean())
    ci = float((changed & inst).sum() / max(int(inst.sum()), 1))
    mc = float(left_conf[changed].mean()) if changed.any() else 1.0
    a0 = float(inst.sum())
    a1 = float(np.isin(mask_a, np.asarray(list(ctx.instrument_ids), dtype=base.dtype)).sum())
    ev = ctx.evidence
    su = float((ev["lr_ok"] & ~ev["occluded"] & ev["in_view"]).mean()) if ev is not None else 0.0
    if ctx.right_label is not None:
        v = ctx.right_label != 255
        ra = float((ctx.right_label[v] == base[v]).mean()) if v.any() else 0.0
    else:
        ra = 0.0
    mem = getattr(st, "memory", []) or []
    ml = float(len(mem))
    mr = float(np.mean([m.reliability for m in mem])) if mem else 0.0
    return np.array([cf, ci, mc, a1 / max(a0, 1.0), su, ra, ml, mr], np.float32)


# --------------------------------------------------------------------------- #
# LABELS for training the selector -- the only place ground truth is read
# --------------------------------------------------------------------------- #
def delta_scores(before: np.ndarray, after: np.ndarray, gt: np.ndarray, num_classes: int,
                 instrument_ids: Sequence[int]) -> Dict[str, Optional[float]]:
    """How much an action changed quality vs ground truth (after - before).
    TRAINING-DATA CONSTRUCTION ONLY: never call this at inference."""
    ib = frame_iou(before, gt, num_classes, class_ids=list(instrument_ids))
    ia = frame_iou(after, gt, num_classes, class_ids=list(instrument_ids))
    ob = official_frame_iou(before, gt, num_classes)
    oa = official_frame_iou(after, gt, num_classes)
    return {"instr": ia - ib, "official": (None if ob is None or oa is None else oa - ob),
            "instr_before": ib, "instr_after": ia}
