"""
THE HANDOFF BOUNDARY — Stages 1-2 (done) to Stages 3-5 (teammate).

Stages 1 and 2 are complete and frozen. This module is the contract between
them and the rest of the framework. Implementing the three Protocols below is
sufficient to finish the paper; **nothing in stages 1-2 needs to be edited**,
and no import runs the other way, so the two halves can be developed and tested
independently.

What Stages 1-2 hand over, per frame t, is exactly the left-hand side of the
report's central decision (eq. 26):

    S^final_t = S_t          if r_t >= tau       <- Stage 2 decided this
              = S-hat_t      if r_t <  tau       <- Stage 4 must produce this

`FrameState` below carries everything eqs. (17)-(25) need. It is produced by
`SelectiveInference.step()`, which already implements Algorithm 1 lines 1-6 and
20-23 (segment, score, decide, maintain the memory) and delegates lines 11-19
to the recovery and identity modules through these Protocols.

Teammate's job, in order:

  1. `TemporalAligner`  — eq. (16). An `IdentityAlign` and a Farneback
     implementation already exist in `sitr/align.py`; the report's *default* is
     region-level correspondence via eq. (24), which is still to be written.
  2. `RecoveryModule`   — eqs. (17)-(23): weighted aggregation of the memory,
     convex fusion with the current probability map, and the per-pixel
     visibility modulation nu_t that distinguishes recoverable from
     unrecoverable failure. This is the paper's main claim.
  3. `IdentityVerifier` — eqs. (24)-(25): Hungarian assignment of recovered
     regions to the identities in memory, rejecting matches above c_max.

Stage 6 (difficult-event construction) and Stage 7 (event-based metrics: RSR,
recovery latency, FRR, identity continuity) consume `FrameState` histories and
need no new model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


# --------------------------------------------------------------------------- #
# data carried across the boundary
# --------------------------------------------------------------------------- #
@dataclass
class MemoryEntry:
    """M_t = {S_t, P_t, E_t, ID_t, r_t} — report eq. (14)."""
    t: int
    mask: np.ndarray                      # S_t, [H,W] uint8 target class ids
    probs: Optional[np.ndarray]           # P_t, [C,H,W] float32 (may be None)
    embedding: np.ndarray                 # E_t, [D] float32
    identities: Dict[int, int]            # region label -> instrument identity
    reliability: float                    # r_t
    area: float = 0.0                     # a_s = |Omega^fg_s|, for eq. (8)
    gray: Optional[np.ndarray] = None     # for flow-based alignment


@dataclass
class FrameState:
    """Everything Stages 1-2 know about frame t, handed to Stages 3-5."""
    t: int
    image: np.ndarray                     # I_t, [H,W,3] uint8 RGB
    probs: np.ndarray                     # P_t, [C,H,W] float32, eq. (4)
    mask: np.ndarray                      # S_t, [H,W] uint8, eq. (5)
    features: np.ndarray                  # Phi_t, [D,H,W] float32, for eq. (22)
    embedding: np.ndarray                 # E_t, [D] float32, eq. (9)
    indicators: Tuple[float, float, float, float]   # (q_t, T_t, A_t, F_t)
    reliability: float                    # r_t, eq. (11)
    reliable: bool                        # r_t >= tau, eq. (12)
    foreground_area: int                  # a_t = |Omega^fg_t|
    degenerate: bool                      # Omega^fg_t empty, eq. (10)
    cold_start: bool                      # B_t empty: unrecoverable by eq. (15)
    memory: List[MemoryEntry] = field(default_factory=list)   # B_t
    #: ground truth Y_t, populated ONLY for offline evaluation. Stages 3-5 must
    #: never read this at inference time -- eq. (13) is a training target.
    ground_truth: Optional[np.ndarray] = None

    @property
    def needs_recovery(self) -> bool:
        return (not self.reliable) and (not self.cold_start)


@dataclass
class RecoveryResult:
    """What Stage 4 returns — eqs. (19)-(20)."""
    mask: np.ndarray                      # S-hat_t
    fused_probs: Optional[np.ndarray] = None       # P'_t
    alpha: Optional[np.ndarray] = None             # alpha_t(x,y), eqs. (21),(23)
    visibility: Optional[np.ndarray] = None        # nu_t(x,y), eq. (22)
    source_frames: Sequence[int] = ()              # which memory entries were used


@dataclass
class IdentityResult:
    """What Stage 5 returns — eqs. (24)-(25)."""
    mask: np.ndarray                      # S^final_t after identity association
    assignments: Dict[int, int]           # region label -> identity (or -1)
    uncertain: Sequence[int] = ()         # regions left unassigned (C_ij > c_max)
    switches: int = 0                     # identity switches vs the memory


# --------------------------------------------------------------------------- #
# the three Protocols to implement
# --------------------------------------------------------------------------- #
@runtime_checkable
class TemporalAligner(Protocol):
    """W_{s->t} — report eq. (16). Label maps only: nearest neighbour."""
    name: str

    def __call__(self, mask_s: np.ndarray, gray_s: Optional[np.ndarray] = None,
                 gray_t: Optional[np.ndarray] = None) -> np.ndarray: ...


@runtime_checkable
class RecoveryModule(Protocol):
    """R_phi — report eq. (17). Called ONLY when `state.needs_recovery`."""

    def __call__(self, state: FrameState) -> RecoveryResult: ...


@runtime_checkable
class IdentityVerifier(Protocol):
    """Report eqs. (24)-(25). Applied on BOTH paths, accepted and recovered."""

    def __call__(self, mask: np.ndarray, state: FrameState) -> IdentityResult: ...


# --------------------------------------------------------------------------- #
# the driver: Algorithm 1
# --------------------------------------------------------------------------- #
class SelectiveInference:
    """One time step of Algorithm 1, with stages 3-5 injected.

    Stages 1-2 are already wired: the model, the four indicators, the threshold
    decision and the bounded reliable memory. Passing `recovery=None` gives the
    honest Stage 1+2 system (accept everything, flag the rest), which is exactly
    "Baseline A + a reliability gate" and is runnable today.
    """

    def __init__(self, segmenter, reliability_model, *,
                 instrument_ids: Sequence[int], num_classes: int,
                 tau: float = 0.5, memory_size: int = 5,
                 tau_memory: Optional[float] = None,
                 aligner: Optional[TemporalAligner] = None,
                 recovery: Optional[RecoveryModule] = None,
                 identity: Optional[IdentityVerifier] = None,
                 device: str = "cpu", store_probs: bool = True):
        self.segmenter = segmenter
        self.reliability_model = reliability_model
        self.instrument_ids = list(instrument_ids)
        self.num_classes = int(num_classes)
        self.tau = float(tau)
        # TWO DISTINCT THRESHOLDS -- conflating them is a real bug that inflates
        # the routing rate.  `tau` (eq. 2) decides whether THIS frame is routed
        # to recovery.  `tau_memory` (eq. 15, the report's validation-set
        # "tau") decides whether a frame is good enough to BUILD HISTORY FROM.
        # Stage 2 trains and evaluates with the second one, so using the first
        # for admission makes the memory stricter at inference than it was in
        # training: fewer frames enter B_t -> more cold starts -> T/A/F fall to
        # zero -> r_t drops -> still more frames are routed.  A compounding
        # feedback loop, not a small offset.
        self.tau_memory = float(tau if tau_memory is None else tau_memory)
        self.K = int(memory_size)
        self.aligner = aligner
        self.recovery = recovery
        self.identity = identity
        self.device = device
        # eq. (18) aggregates the WARPED PROBABILITY MAPS of the memory, not
        # its hard masks, so P_s must actually be in B_t. At K=5 and 256x320
        # that is a few MB; storing it is the correct default and turning it
        # off silently breaks the recovery module.
        self.store_probs = store_probs
        self.memory: List[MemoryEntry] = []
        self.t = 0
        self.stats = {"frames": 0, "routed": 0, "unrecoverable": 0,
                      "degenerate": 0, "identity_switches": 0}

    def reset(self) -> None:
        """Call between sequences: memory must never cross a procedure."""
        self.memory.clear()
        self.t = 0

    # -- Algorithm 1 -------------------------------------------------------
    def step(self, state: FrameState) -> Tuple[np.ndarray, FrameState]:
        self.stats["frames"] += 1

        # eq. (15) B_t = {M_s | t-K <= s <= t-1, r_s >= tau}, |B_t| <= K.
        # Prune BEFORE handing the memory to Stages 3-5, or the entry appended
        # on the previous frame can leave the window unnoticed and recovery
        # propagates from a frame that is no longer admissible.
        while self.memory and self.memory[0].t < state.t - self.K:
            self.memory.pop(0)
        state.memory = list(self.memory)

        if state.reliable:                                   # line 5-6
            final = state.mask
        elif state.cold_start or self.recovery is None:      # lines 8-9
            final = state.mask                               # flag unrecoverable
            self.stats["unrecoverable"] += 1
        else:                                                # lines 11-16
            self.stats["routed"] += 1
            final = self.recovery(state).mask

        if state.degenerate:
            self.stats["degenerate"] += 1

        if self.identity is not None:                        # line 19
            res = self.identity(final, state)
            final = res.mask
            self.stats["identity_switches"] += int(res.switches)

        if state.reliability >= self.tau_memory:              # lines 20-22
            self.memory.append(MemoryEntry(
                t=state.t, mask=final,
                probs=state.probs if self.store_probs else None,
                embedding=state.embedding, identities={},
                reliability=state.reliability,
                area=float(state.foreground_area)))
            # Enforce the cardinality bound explicitly as well as the recency
            # window: eq. (15) caps |B_t| at K, which is what gives the system
            # constant per-frame memory however long the procedure runs.
            while len(self.memory) > self.K:
                self.memory.pop(0)

        self.t = state.t + 1
        return final, state                                  # line 23


__all__ = [
    "MemoryEntry", "FrameState", "RecoveryResult", "IdentityResult",
    "TemporalAligner", "RecoveryModule", "IdentityVerifier",
    "SelectiveInference",
]
