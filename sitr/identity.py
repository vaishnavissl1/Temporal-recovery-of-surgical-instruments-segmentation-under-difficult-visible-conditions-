"""
Stage 5 — Identity Verification (report eqs. 24–25).

After segmentation (reliable or recovered), associate each instrument region
in the current frame with a persistent identity from the reliable memory.

The assignment uses Hungarian matching on a combined cost:
    C_ij = lambda_m * (1 - IoU(R_i, R_j))
         + lambda_a * (1 - cos(E_i, E_j))
         + lambda_d * ||c_i - c_j||_2 / diag

Matches with C_ij > c_max are REJECTED — the region is marked as "uncertain"
with identity = -1.

IMPORTANT DESIGN DECISION: identity uncertainty does NOT modify the
segmentation mask.  An uncertain region keeps its segmentation but gets
identity = -1 in the assignments dict.  The IdentityVerifier's job is
    region -> instrument identity
NOT
    uncertain -> delete from segmentation

This is applied on BOTH paths (reliable accepted frames AND recovered frames),
so the identity processing is fair across the two pipelines.

Identity switches are counted: when a region's matched identity differs from
what the memory had for the same spatial area.  This feeds the identity
continuity metric of eq. (37).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .align import _cosine, _extract_regions
from .stage34_interface import (FrameState, IdentityResult, IdentityVerifier,
                                MemoryEntry)


class HungarianIdentityVerifier:
    """Report eqs. (24)–(25).  Applied on BOTH paths, accepted and recovered.

    Parameters
    ----------
    lambda_m : float
        Weight for mask IoU cost term.
    lambda_a : float
        Weight for appearance (cosine) cost term.
    lambda_d : float
        Weight for normalised centroid distance cost term.
    c_max : float
        Maximum acceptable cost for a match.  Matches above this are
        rejected and the region is marked uncertain (identity = -1).
    instrument_ids : sequence of int, optional
        Instrument class IDs.  If None, all non-zero classes are instruments.
    """

    def __init__(self, lambda_m: float = 1 / 3, lambda_a: float = 1 / 3,
                 lambda_d: float = 1 / 3, c_max: float = 0.5,
                 instrument_ids: Optional[Sequence[int]] = None):
        self.lambda_m = float(lambda_m)
        self.lambda_a = float(lambda_a)
        self.lambda_d = float(lambda_d)
        self.c_max = float(c_max)
        self.instrument_ids = list(instrument_ids) if instrument_ids else None

    def __call__(self, mask: np.ndarray,
                 state: FrameState) -> IdentityResult:
        """Associate instrument regions with identities from memory."""
        inst_ids = self.instrument_ids
        if inst_ids is None:
            inst_ids = [int(v) for v in np.unique(mask) if v != 0]

        # If no memory, every region gets a fresh identity
        if not state.memory:
            assignments = self._fresh_assignments(mask, inst_ids)
            return IdentityResult(mask=mask, assignments=assignments,
                                  uncertain=(), switches=0)

        # Extract regions from current mask
        cur_regions = _extract_regions(mask, inst_ids)
        if not cur_regions:
            return IdentityResult(mask=mask, assignments={},
                                  uncertain=(), switches=0)

        # Extract regions from the most recent memory entry
        mem_entry = state.memory[-1]
        mem_regions = _extract_regions(mem_entry.mask, inst_ids)

        if not mem_regions:
            # Memory has no instrument regions — assign fresh identities
            assignments = {}
            for idx, r in enumerate(cur_regions):
                assignments[idx] = idx  # fresh ID
            return IdentityResult(mask=mask, assignments=assignments,
                                  uncertain=(), switches=0)

        # Compute embeddings for current regions from dense features
        cur_embeddings = []
        for r in cur_regions:
            if state.features is not None:
                pixels = r["pixels"]
                if pixels.any():
                    emb = state.features[:, pixels].mean(axis=1)
                else:
                    emb = state.embedding
            else:
                emb = state.embedding
            cur_embeddings.append(emb)

        # Compute embeddings for memory regions
        # Memory doesn't have dense features, so use global embedding
        # weighted by which class is present
        mem_embeddings = []
        for r in mem_regions:
            mem_embeddings.append(mem_entry.embedding)

        # Build cost matrix (eq. 24)
        H, W = mask.shape[:2]
        diag = float(np.sqrt(H ** 2 + W ** 2))
        n_cur = len(cur_regions)
        n_mem = len(mem_regions)
        cost = np.full((n_cur, n_mem), 1e6, dtype=np.float64)

        for i, cr in enumerate(cur_regions):
            for j, mr in enumerate(mem_regions):
                # Mask IoU
                iou = _region_iou_pixels(cr["pixels"], mr["pixels"])
                iou_cost = 1.0 - iou

                # Appearance cosine
                cos_sim = _cosine(cur_embeddings[i], mem_embeddings[j])
                app_cost = 1.0 - cos_sim

                # Normalised centroid distance
                dy = cr["centroid"][0] - mr["centroid"][0]
                dx = cr["centroid"][1] - mr["centroid"][1]
                dist = float(np.sqrt(dy ** 2 + dx ** 2)) / max(diag, 1e-6)

                cost[i, j] = (self.lambda_m * iou_cost
                               + self.lambda_a * app_cost
                               + self.lambda_d * dist)

        # Hungarian assignment
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        # Build assignments and detect switches
        assignments: Dict[int, int] = {}
        uncertain: List[int] = []
        switches = 0

        # Memory identities: map memory region index to its identity
        mem_identities = mem_entry.identities or {}

        matched_cur = set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] > self.c_max:
                # Reject — mark as uncertain
                assignments[ri] = -1
                uncertain.append(ri)
            else:
                # Accept the match — assign memory region's identity
                mem_id = mem_identities.get(ci, ci)
                assignments[ri] = mem_id
                matched_cur.add(ri)

                # Check for identity switch
                # A switch occurs if this region had a different identity before
                cr = cur_regions[ri]
                mr = mem_regions[ci]
                if cr["label"] != mr["label"]:
                    switches += 1

        # Unmatched current regions get uncertain identity
        for i in range(n_cur):
            if i not in assignments:
                assignments[i] = -1
                uncertain.append(i)

        # The mask is NOT modified by identity verification.
        # Uncertain regions keep their segmentation; only the identity
        # assignment changes.
        return IdentityResult(
            mask=mask,
            assignments=assignments,
            uncertain=tuple(uncertain),
            switches=switches,
        )

    def _fresh_assignments(self, mask: np.ndarray,
                           inst_ids: Sequence[int]) -> Dict[int, int]:
        """Assign fresh identities when no memory is available."""
        regions = _extract_regions(mask, inst_ids)
        return {i: i for i in range(len(regions))}


def _region_iou_pixels(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boolean pixel masks."""
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / max(union, 1)


__all__ = ["HungarianIdentityVerifier"]
