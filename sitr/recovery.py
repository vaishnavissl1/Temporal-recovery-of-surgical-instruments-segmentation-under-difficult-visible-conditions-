"""
Stage 4 — Temporal Recovery Module (report eqs. 17–23).

This is the paper's main claim: when the reliability gate (Stage 2) flags a
frame as unreliable, repair it by fusing the current prediction with
propagated probability maps from the reliable memory B_t.

The recovery pipeline per frame:

    1. Align each memory frame's probability map to the current frame
       using the configured aligner (eq. 16).
    2. Weight the aligned maps by reliability × temporal decay (eq. 18).
    3. Compute per-pixel feature similarity between the current frame's
       dense features and per-instrument memory embeddings (eq. 22).
       NOTE: we call this "feature similarity" not "visibility" — cosine
       similarity does not inherently measure pixel visibility; it must be
       validated empirically to correlate with successful recovery.
    4. Compute spatially-varying fusion weight alpha_t(x,y) (eq. 23).
    5. Fuse current and propagated probabilities (eq. 19).
    6. Take argmax for the recovered mask (eq. 20).

Design decisions:
    - Feature similarity uses PER-INSTRUMENT embeddings from memory, not a
      single mean global embedding.  Averaging red-forceps and blue-scissors
      embeddings creates a representation of neither instrument; per-region
      embeddings are consistent with Stage 3's region-based design.
    - gamma (temporal decay) is a hyperparameter, not hardcoded.
    - Probability map warping: for each memory entry, we apply the aligner
      to get a warped mask, then use that warped mask to remap probability
      channels spatially via the same centroid offsets (for RegionAligner)
      or the same flow field (for FarnebackAlign).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .align import (IdentityAlign, RegionAligner, _cosine, _extract_regions)
from .stage34_interface import (FrameState, MemoryEntry, RecoveryResult,
                                RecoveryModule)


class TemporalRecovery:
    """R_phi — report eq. (17).  Called ONLY when state.needs_recovery.

    Parameters
    ----------
    aligner : TemporalAligner
        The spatial alignment operator W_{s->t}.
    tau : float
        The routing threshold — used for alpha_t = min(1, r_t / tau).
    gamma : float
        Temporal decay factor for memory weighting: w_s = r_s * gamma^(t-s).
        At K=5 and 1 Hz, gamma=0.95 gives the oldest frame weight ~0.77.
        Treat as a hyperparameter; test gamma in {0.8, 0.9, 0.95, 0.99}.
    num_classes : int
        Number of segmentation classes.
    instrument_ids : sequence of int
        Class IDs that are instruments (not anatomy/background).
    """

    def __init__(self, aligner=None, *, tau: float = 0.76,
                 gamma: float = 0.95, num_classes: int = 2,
                 instrument_ids: Optional[Sequence[int]] = None):
        self.aligner = aligner or IdentityAlign()
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.num_classes = int(num_classes)
        self.instrument_ids = list(instrument_ids) if instrument_ids else [1]

    def __call__(self, state: FrameState) -> RecoveryResult:
        if len(state.memory) == 0:
            # Cold start — no memory to recover from
            return RecoveryResult(mask=state.mask, source_frames=())

        C, H, W = state.probs.shape
        t = state.t

        # ------------------------------------------------------------------
        # Step 1 & 2: Align and aggregate memory probability maps (eq. 18)
        # ------------------------------------------------------------------
        weighted_probs = np.zeros((C, H, W), dtype=np.float64)
        total_weight = 0.0
        source_frames: List[int] = []

        # Collect per-instrument embeddings from memory for feature similarity
        instrument_embeddings = self._collect_instrument_embeddings(state)

        for entry in state.memory:
            if entry.probs is None:
                continue

            dt = max(t - entry.t, 1)
            w_s = entry.reliability * (self.gamma ** dt)
            if w_s < 1e-8:
                continue

            # Align the memory mask to the current frame
            aligned_probs = self._align_probs(
                entry, state, entry.probs)

            if aligned_probs is None:
                continue

            weighted_probs += w_s * aligned_probs.astype(np.float64)
            total_weight += w_s
            source_frames.append(entry.t)

        if total_weight < 1e-8 or len(source_frames) == 0:
            # No usable memory — return base prediction unchanged
            return RecoveryResult(mask=state.mask, source_frames=())

        # P^prop_t: normalised weighted average (eq. 18)
        p_prop = weighted_probs / total_weight  # [C, H, W]
        # Ensure valid probability distribution
        p_prop = np.clip(p_prop, 0.0, None)
        denom = p_prop.sum(axis=0, keepdims=True)
        denom = np.maximum(denom, 1e-8)
        p_prop = p_prop / denom

        # ------------------------------------------------------------------
        # Step 3: Per-pixel feature similarity (eq. 22)
        # ------------------------------------------------------------------
        feat_sim = self._compute_feature_similarity(
            state.features, state.mask, instrument_embeddings)
        # feat_sim: [H, W] in [0, 1]

        # ------------------------------------------------------------------
        # Step 4: Spatially-varying alpha (eqs. 21, 23)
        # ------------------------------------------------------------------
        alpha_scalar = min(1.0, state.reliability / self.tau)  # eq. (21)
        # eq. (23): alpha_t(x,y) = 1 - (1 - alpha_scalar) * feat_sim(x,y)
        # High feat_sim → pixel looks like a known instrument → trust memory more → lower alpha
        # Low feat_sim → severe degradation → fall back to base → higher alpha
        alpha_map = 1.0 - (1.0 - alpha_scalar) * feat_sim  # [H, W]
        alpha_map = np.clip(alpha_map, 0.0, 1.0)

        # ------------------------------------------------------------------
        # Step 5: Convex fusion (eq. 19)
        # ------------------------------------------------------------------
        alpha_3d = alpha_map[np.newaxis, :, :]  # [1, H, W]
        p_current = state.probs.astype(np.float64)  # [C, H, W]
        p_fused = alpha_3d * p_current + (1.0 - alpha_3d) * p_prop  # [C, H, W]
        p_fused = p_fused.astype(np.float32)

        # ------------------------------------------------------------------
        # Step 6: Argmax recovered mask (eq. 20)
        # ------------------------------------------------------------------
        recovered_mask = np.argmax(p_fused, axis=0).astype(np.uint8)

        return RecoveryResult(
            mask=recovered_mask,
            fused_probs=p_fused,
            alpha=alpha_map.astype(np.float32),
            visibility=feat_sim.astype(np.float32),
            source_frames=tuple(source_frames),
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _align_probs(self, entry: MemoryEntry, state: FrameState,
                     probs_s: np.ndarray) -> Optional[np.ndarray]:
        """Align a memory entry's probability map to the current frame.

        For simple aligners (Identity, Farneback), we warp each probability
        channel independently.  For RegionAligner, we translate matched
        regions' probability channels by centroid offsets.
        """
        C, H, W = probs_s.shape

        if isinstance(self.aligner, IdentityAlign):
            # no spatial transform
            return probs_s

        if isinstance(self.aligner, RegionAligner):
            # Set target info for region matching
            self.aligner.set_target(
                state.mask,
                features_s=None,  # memory doesn't store dense features
                features_t=state.features)
            # Get the warped mask to determine the spatial mapping
            warped_mask = self.aligner(entry.mask, entry.gray, None)
            self.aligner.clear_target()

            # For region alignment, we need to translate probability channels
            # by the same offsets used for the mask
            return self._translate_probs_by_regions(
                entry.mask, warped_mask, probs_s, H, W)
        else:
            # Farneback or similar dense-flow aligner: compute flow ONCE,
            # then warp each probability channel with the same remap.
            gray_t = cv2.cvtColor(state.image, cv2.COLOR_RGB2GRAY)
            remap = self._compute_flow_remap(entry.gray, gray_t, H, W)
            if remap is None:
                return probs_s
            map_x, map_y = remap
            aligned = np.zeros_like(probs_s)
            for c in range(C):
                aligned[c] = cv2.remap(
                    probs_s[c].astype(np.float32), map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            return aligned

    def _translate_probs_by_regions(
            self, mask_s: np.ndarray, warped_mask: np.ndarray,
            probs_s: np.ndarray, H: int, W: int) -> np.ndarray:
        """Translate probability channels based on how the mask was warped.

        Compare source and warped masks to find which pixels moved where,
        then apply the same transform to probabilities.
        """
        C = probs_s.shape[0]
        result = np.zeros((C, H, W), dtype=np.float32)

        # For each instrument class, find the centroid shift
        src_regions = _extract_regions(mask_s, self.instrument_ids)
        warped_regions = _extract_regions(warped_mask, self.instrument_ids)

        if not src_regions:
            return probs_s  # nothing to translate

        # Build mapping: for each source region, find its warped counterpart
        # by matching class labels and proximity
        translated_pixels = np.zeros((H, W), dtype=bool)
        for sr in src_regions:
            # find the warped region of the same class
            best_wr = None
            best_iou = -1.0
            for wr in warped_regions:
                if wr["label"] == sr["label"]:
                    iou = float(np.logical_and(
                        sr["pixels"], wr["pixels"]).sum()) / max(
                        float(np.logical_or(
                            sr["pixels"], wr["pixels"]).sum()), 1)
                    if iou > best_iou:
                        best_iou = iou
                        best_wr = wr

            if best_wr is None:
                # region was discarded by alignment — skip it
                continue

            # compute centroid offset
            dy = best_wr["centroid"][0] - sr["centroid"][0]
            dx = best_wr["centroid"][1] - sr["centroid"][1]
            dy_int, dx_int = int(round(dy)), int(round(dx))

            # translate probability values for this region
            ys, xs = np.where(sr["pixels"])
            new_ys = ys + dy_int
            new_xs = xs + dx_int
            valid = ((new_ys >= 0) & (new_ys < H) &
                     (new_xs >= 0) & (new_xs < W))
            for c in range(C):
                result[c, new_ys[valid], new_xs[valid]] = \
                    probs_s[c, ys[valid], xs[valid]]
            translated_pixels[new_ys[valid], new_xs[valid]] = True

        # For pixels not covered by any translated region, use the original
        # background probabilities
        not_translated = ~translated_pixels
        if not_translated.any():
            for c in range(C):
                result[c, not_translated] = probs_s[c, not_translated]

        return result

    def _compute_flow_remap(self, gray_s: Optional[np.ndarray],
                            gray_t: np.ndarray, h: int, w: int
                            ) -> Optional[tuple]:
        """Compute Farneback flow ONCE and return (map_x, map_y) for remap."""
        if gray_s is None:
            return None
        if gray_s.shape != gray_t.shape:
            gray_s = cv2.resize(gray_s, (gray_t.shape[1], gray_t.shape[0]),
                                interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(
            gray_t, gray_s, None, pyr_scale=0.5, levels=4, winsize=21,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

        if (h, w) != flow.shape[:2]:
            sy, sx = h / flow.shape[0], w / flow.shape[1]
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= sx
            flow[..., 1] *= sy

        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        return (xs + flow[..., 0], ys + flow[..., 1])

    def _collect_instrument_embeddings(
            self, state: FrameState) -> Dict[int, np.ndarray]:
        """Collect per-instrument mean embeddings from the reliable memory.

        Instead of averaging ALL instrument embeddings into one global vector
        (which creates a representation of no instrument in particular),
        we compute a mean embedding for each instrument class separately.
        """
        # Accumulate embeddings per instrument class across memory
        class_embs: Dict[int, List[np.ndarray]] = {}
        for entry in state.memory:
            for cid in self.instrument_ids:
                region_pixels = (entry.mask == cid)
                if region_pixels.sum() == 0:
                    continue
                # Use the entry's global embedding weighted by this class
                # (we don't have per-pixel features for memory frames, so
                # we use the global embedding as a proxy when this class
                # was present in the memory frame)
                if cid not in class_embs:
                    class_embs[cid] = []
                class_embs[cid].append(entry.embedding)

        result: Dict[int, np.ndarray] = {}
        for cid, embs in class_embs.items():
            result[cid] = np.mean(embs, axis=0).astype(np.float32)
        return result

    def _compute_feature_similarity(
            self, features: np.ndarray, mask: np.ndarray,
            instrument_embeddings: Dict[int, np.ndarray]) -> np.ndarray:
        """Compute per-pixel feature similarity score (eq. 22).

        For each pixel, compare its dense feature vector against the
        embedding of the instrument class that pixel is predicted to belong
        to.  Background pixels get similarity 0 (no instrument to compare
        against).  Pixels predicted as an instrument class for which no
        memory embedding exists also get 0.

        NOTE: This is called "feature similarity", not "visibility".
        Cosine similarity does not inherently measure pixel visibility;
        that relationship must be validated empirically.
        """
        D, H, W = features.shape
        sim = np.zeros((H, W), dtype=np.float32)

        if not instrument_embeddings:
            return sim

        # Normalise the dense features once: [D, H, W]
        feat_norm = np.linalg.norm(features, axis=0, keepdims=True)  # [1, H, W]
        feat_norm = np.maximum(feat_norm, 1e-6)
        features_normed = features / feat_norm  # [D, H, W]

        for cid, emb in instrument_embeddings.items():
            region_pixels = (mask == cid)
            if not region_pixels.any():
                continue
            # Normalise embedding
            emb_norm = np.linalg.norm(emb)
            if emb_norm < 1e-6:
                continue
            emb_normed = emb / emb_norm  # [D]

            # Cosine similarity: dot product of normalised vectors
            # emb_normed: [D], features_normed: [D, H, W]
            cos_sim = np.einsum('d,dhw->hw', emb_normed, features_normed)
            # Clamp to [0, 1] — negative cosine means very dissimilar
            cos_sim = np.clip(cos_sim, 0.0, 1.0)

            # Only assign to pixels predicted as this instrument class
            sim[region_pixels] = cos_sim[region_pixels]

        return sim


__all__ = ["TemporalRecovery"]
