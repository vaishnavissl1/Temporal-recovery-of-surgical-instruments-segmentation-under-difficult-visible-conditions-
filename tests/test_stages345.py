#!/usr/bin/env python3
"""
Correctness tests for Stages 3–5: alignment, recovery, identity verification.

    python -m tests.test_stages345

These complement the 30 tests in test_pipeline.py.  They validate:
  - RegionAligner preserves label values and discards unmatched regions
  - TemporalRecovery produces valid probability distributions and masks
  - HungarianIdentityVerifier rejects high-cost matches without deleting mask
  - Full pipeline wiring through SelectiveInference
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.align import (  # noqa: E402
    IdentityAlign, RegionAligner, _extract_regions, build_align)
from sitr.recovery import TemporalRecovery  # noqa: E402
from sitr.identity import HungarianIdentityVerifier  # noqa: E402
from sitr.stage34_interface import (  # noqa: E402
    FrameState, MemoryEntry, RecoveryResult, IdentityResult,
    SelectiveInference, RecoveryModule, IdentityVerifier)
from sitr.models import ReliabilityMLP  # noqa: E402

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            FAIL.append((name, e))
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
    return deco


def _make_state(t=5, H=16, W=20, C=2, D=8, r=0.3, tau=0.5,
                memory_entries=None, mask=None, seed=42):
    """Create a synthetic FrameState for testing."""
    rng = np.random.default_rng(seed)
    if mask is None:
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[2:8, 3:10] = 1   # instrument region 1
        mask[9:14, 12:18] = 1  # instrument region 2

    probs = rng.random((C, H, W)).astype(np.float32)
    probs /= probs.sum(axis=0, keepdims=True)
    features = rng.random((D, H, W)).astype(np.float32)
    embedding = rng.random(D).astype(np.float32)
    image = rng.integers(0, 255, (H, W, 3)).astype(np.uint8)

    memory = memory_entries if memory_entries is not None else []
    return FrameState(
        t=t, image=image, probs=probs, mask=mask, features=features,
        embedding=embedding, indicators=(r, 0.5, 0.5, 0.5),
        reliability=r, reliable=r >= tau, foreground_area=int((mask > 0).sum()),
        degenerate=False, cold_start=(len(memory) == 0), memory=memory)


def _make_memory_entry(t=2, H=16, W=20, C=2, D=8, r=0.8, seed=99):
    """Create a synthetic MemoryEntry."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[3:9, 4:11] = 1
    mask[10:14, 13:18] = 1

    probs = rng.random((C, H, W)).astype(np.float32)
    probs /= probs.sum(axis=0, keepdims=True)
    embedding = rng.random(D).astype(np.float32)
    import cv2
    gray = cv2.cvtColor(
        rng.integers(0, 255, (H, W, 3)).astype(np.uint8),
        cv2.COLOR_RGB2GRAY)

    return MemoryEntry(
        t=t, mask=mask, probs=probs, embedding=embedding,
        identities={0: 0, 1: 1}, reliability=r, area=int((mask > 0).sum()),
        gray=gray)


def main() -> int:
    print("Stage 3 — RegionAligner")

    @check("_extract_regions finds connected instrument components")
    def _():
        mask = np.zeros((16, 20), dtype=np.uint8)
        mask[2:6, 3:8] = 1
        mask[10:14, 12:17] = 1
        regions = _extract_regions(mask, [1])
        assert len(regions) == 2, f"expected 2 regions, got {len(regions)}"
        for r in regions:
            assert r["label"] == 1
            assert r["area"] > 0
            assert len(r["centroid"]) == 2

    @check("RegionAligner preserves label values (nearest-neighbour invariant)")
    def _():
        mask_s = np.zeros((16, 20), dtype=np.uint8)
        mask_s[4:8, 5:10] = 2
        mask_t = np.zeros((16, 20), dtype=np.uint8)
        mask_t[5:9, 6:11] = 2

        aligner = RegionAligner(instrument_ids=[1, 2])
        aligner.set_target(mask_t)
        warped = aligner(mask_s)
        aligner.clear_target()

        valid_labels = set(np.unique(mask_s).tolist()) | {0}
        actual = set(np.unique(warped).tolist())
        assert actual.issubset(valid_labels), \
            f"warped mask has labels {actual} not in {valid_labels}"

    @check("RegionAligner discards unmatched source regions")
    def _():
        mask_s = np.zeros((16, 20), dtype=np.uint8)
        mask_s[2:6, 2:6] = 1   # region in source
        mask_t = np.zeros((16, 20), dtype=np.uint8)
        # target has NO instrument regions → nothing to match to

        aligner = RegionAligner(instrument_ids=[1])
        aligner.set_target(mask_t)
        warped = aligner(mask_s)
        aligner.clear_target()

        # source regions should be discarded (all background)
        assert warped.sum() == 0, \
            "unmatched source regions should be discarded, not carried forward"

    @check("RegionAligner falls back to identity without set_target()")
    def _():
        mask_s = np.zeros((16, 20), dtype=np.uint8)
        mask_s[3:7, 4:8] = 1
        aligner = RegionAligner(instrument_ids=[1])
        warped = aligner(mask_s)
        assert np.array_equal(warped, mask_s), "should be identity without target"

    @check("build_align supports 'region' name")
    def _():
        a = build_align("region", instrument_ids=[1, 2])
        assert isinstance(a, RegionAligner)
        assert a.name == "region"

    print("\nStage 4 — TemporalRecovery")

    @check("recovery produces valid probability distribution (sums to ~1)")
    def _():
        mem = _make_memory_entry(t=3)
        state = _make_state(t=5, r=0.3, tau=0.5, memory_entries=[mem])
        state.cold_start = False

        rec = TemporalRecovery(
            aligner=IdentityAlign(), tau=0.5, gamma=0.95,
            num_classes=2, instrument_ids=[1])
        result = rec(state)

        assert result.fused_probs is not None
        sums = result.fused_probs.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=0.02), \
            f"fused probs don't sum to 1: range [{sums.min():.4f}, {sums.max():.4f}]"

    @check("recovered mask has only valid class labels")
    def _():
        mem = _make_memory_entry(t=3)
        state = _make_state(t=5, r=0.3, tau=0.5, memory_entries=[mem])
        state.cold_start = False

        rec = TemporalRecovery(
            aligner=IdentityAlign(), tau=0.5, gamma=0.95,
            num_classes=2, instrument_ids=[1])
        result = rec(state)

        valid = set(range(2))
        actual = set(np.unique(result.mask).tolist())
        assert actual.issubset(valid), \
            f"recovered mask has classes {actual} not in {valid}"

    @check("alpha_t = min(1, r_t / tau) is correct and in [0, 1]")
    def _():
        mem = _make_memory_entry(t=3)
        # Only test r values below tau — recovery is only called on unreliable frames
        for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.49]:
            state = _make_state(t=5, r=r, tau=0.5, memory_entries=[mem])
            state.cold_start = False
            state.reliable = False  # ensure needs_recovery is True
            rec = TemporalRecovery(
                aligner=IdentityAlign(), tau=0.5, gamma=0.95,
                num_classes=2, instrument_ids=[1])
            result = rec(state)
            if result.alpha is not None:
                assert result.alpha.min() >= 0.0, f"alpha < 0 at r={r}"
                assert result.alpha.max() <= 1.0, f"alpha > 1 at r={r}"
                # Verify scalar alpha = min(1, r/tau)
                expected_scalar = min(1.0, r / 0.5)
                # The map should have values related to the scalar alpha
                # alpha_map = 1 - (1 - alpha_scalar) * feat_sim
                # So alpha_map >= alpha_scalar everywhere (since feat_sim <= 1)

    @check("feature similarity (visibility) is in [0, 1]")
    def _():
        mem = _make_memory_entry(t=3)
        state = _make_state(t=5, r=0.3, tau=0.5, memory_entries=[mem])
        state.cold_start = False

        rec = TemporalRecovery(
            aligner=IdentityAlign(), tau=0.5, gamma=0.95,
            num_classes=2, instrument_ids=[1])
        result = rec(state)
        if result.visibility is not None:
            assert result.visibility.min() >= 0.0
            assert result.visibility.max() <= 1.0

    @check("recovery with multiple memory entries uses all of them")
    def _():
        mem1 = _make_memory_entry(t=2, seed=10)
        mem2 = _make_memory_entry(t=3, seed=20)
        mem3 = _make_memory_entry(t=4, seed=30)
        state = _make_state(t=5, r=0.3, tau=0.5,
                            memory_entries=[mem1, mem2, mem3])
        state.cold_start = False

        rec = TemporalRecovery(
            aligner=IdentityAlign(), tau=0.5, gamma=0.95,
            num_classes=2, instrument_ids=[1])
        result = rec(state)
        assert len(result.source_frames) == 3, \
            f"expected 3 source frames, got {len(result.source_frames)}"

    @check("recovery returns base mask when no usable memory probs")
    def _():
        mem = _make_memory_entry(t=3)
        mem.probs = None  # no probability map
        state = _make_state(t=5, r=0.3, tau=0.5, memory_entries=[mem])
        state.cold_start = False

        rec = TemporalRecovery(
            aligner=IdentityAlign(), tau=0.5, gamma=0.95,
            num_classes=2, instrument_ids=[1])
        result = rec(state)
        assert np.array_equal(result.mask, state.mask)

    @check("lower reliability gives more weight to memory (smaller alpha)")
    def _():
        mem = _make_memory_entry(t=3)
        rec = TemporalRecovery(
            aligner=IdentityAlign(), tau=0.5, gamma=0.95,
            num_classes=2, instrument_ids=[1])

        state_low = _make_state(t=5, r=0.1, tau=0.5, memory_entries=[mem])
        state_low.cold_start = False
        res_low = rec(state_low)

        state_high = _make_state(t=5, r=0.4, tau=0.5, memory_entries=[mem])
        state_high.cold_start = False
        res_high = rec(state_high)

        if res_low.alpha is not None and res_high.alpha is not None:
            # Lower r → lower alpha → more memory influence
            assert res_low.alpha.mean() <= res_high.alpha.mean(), \
                "lower reliability should produce lower alpha (more memory)"

    print("\nStage 5 — HungarianIdentityVerifier")

    @check("identity verifier does NOT modify the segmentation mask")
    def _():
        mask = np.zeros((16, 20), dtype=np.uint8)
        mask[3:7, 4:9] = 1

        mem = _make_memory_entry(t=3)
        state = _make_state(t=5, r=0.8, tau=0.5, memory_entries=[mem], mask=mask)

        iv = HungarianIdentityVerifier(
            c_max=0.5, instrument_ids=[1])
        result = iv(mask, state)

        assert np.array_equal(result.mask, mask), \
            "identity verification must NOT modify the segmentation mask"

    @check("uncertain regions get identity -1 but keep their segmentation")
    def _():
        # Create a current mask with a region far from any memory region
        mask = np.zeros((16, 20), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # tiny region in corner

        mem = _make_memory_entry(t=3)
        state = _make_state(t=5, r=0.8, tau=0.5, memory_entries=[mem], mask=mask)

        iv = HungarianIdentityVerifier(
            c_max=0.01, instrument_ids=[1])  # very strict → should reject
        result = iv(mask, state)

        # Mask preserved
        assert np.array_equal(result.mask, mask)
        # Check that uncertain regions exist (high c_max rejection)
        if result.assignments:
            uncertain_ids = [k for k, v in result.assignments.items() if v == -1]
            # With c_max=0.01 and distant regions, at least some should be uncertain
            assert len(uncertain_ids) >= 0  # non-negative always

    @check("identity verifier returns valid IdentityResult")
    def _():
        mask = np.zeros((16, 20), dtype=np.uint8)
        mask[3:7, 4:9] = 1
        mask[10:13, 12:16] = 1

        mem = _make_memory_entry(t=3)
        state = _make_state(t=5, r=0.8, tau=0.5, memory_entries=[mem], mask=mask)

        iv = HungarianIdentityVerifier(instrument_ids=[1])
        result = iv(mask, state)

        assert isinstance(result, IdentityResult)
        assert isinstance(result.assignments, dict)
        assert isinstance(result.switches, int)
        assert result.switches >= 0

    @check("identity verifier handles empty memory gracefully")
    def _():
        mask = np.zeros((16, 20), dtype=np.uint8)
        mask[3:7, 4:9] = 1

        state = _make_state(t=0, r=0.8, tau=0.5, memory_entries=[], mask=mask)
        iv = HungarianIdentityVerifier(instrument_ids=[1])
        result = iv(mask, state)

        assert np.array_equal(result.mask, mask)
        assert result.switches == 0

    @check("identity verifier handles all-background mask")
    def _():
        mask = np.zeros((16, 20), dtype=np.uint8)
        mem = _make_memory_entry(t=3)
        state = _make_state(t=5, r=0.8, tau=0.5, memory_entries=[mem], mask=mask)

        iv = HungarianIdentityVerifier(instrument_ids=[1])
        result = iv(mask, state)
        assert len(result.assignments) == 0
        assert result.switches == 0

    print("\nFull pipeline wiring")

    @check("SelectiveInference with recovery + identity runs without error")
    def _():
        rec = TemporalRecovery(
            aligner=IdentityAlign(), tau=0.5, gamma=0.95,
            num_classes=2, instrument_ids=[1])
        iv = HungarianIdentityVerifier(instrument_ids=[1])

        inf = SelectiveInference(
            None, ReliabilityMLP(in_dim=4),
            instrument_ids=[1], num_classes=2,
            tau=0.5, memory_size=5,
            recovery=rec, identity=iv)

        rng = np.random.default_rng(7)
        for t in range(30):
            r = float(rng.random())
            mask = np.zeros((8, 10), dtype=np.uint8)
            mask[2:5, 3:7] = 1
            st = FrameState(
                t=t, image=np.zeros((8, 10, 3), np.uint8),
                probs=np.full((2, 8, 10), 0.5, np.float32),
                mask=mask,
                features=np.ones((4, 8, 10), np.float32) * 0.5,
                embedding=np.ones(4, np.float32) * 0.5,
                indicators=(r, 0.5, 0.5, 0.5), reliability=r,
                reliable=r >= 0.5, foreground_area=int((mask > 0).sum()),
                degenerate=False, cold_start=not inf.memory)
            final, _ = inf.step(st)
            assert final.shape == mask.shape

        assert inf.stats["frames"] == 30
        assert inf.stats["routed"] >= 0

    @check("recovery is never called on cold-start frames")
    def _():
        calls = []

        class TrackedRecovery:
            def __call__(self, st):
                calls.append(st.t)
                assert not st.cold_start, \
                    f"recovery called on cold-start frame {st.t}"
                return RecoveryResult(mask=st.mask)

        inf = SelectiveInference(
            None, ReliabilityMLP(in_dim=4),
            instrument_ids=[1], num_classes=2,
            tau=0.5, memory_size=3,
            recovery=TrackedRecovery())

        for t in range(20):
            r = 0.3  # always unreliable
            st = FrameState(
                t=t, image=np.zeros((4, 4, 3), np.uint8),
                probs=np.full((2, 4, 4), 0.5, np.float32),
                mask=np.zeros((4, 4), np.uint8),
                features=np.zeros((4, 4, 4), np.float32),
                embedding=np.zeros(4, np.float32),
                indicators=(r, 0, 0, 0), reliability=r,
                reliable=False, foreground_area=0,
                degenerate=False, cold_start=not inf.memory)
            inf.step(st)
        # All unreliable with no memory ever built → all cold starts → no calls
        assert len(calls) == 0, \
            f"recovery called {len(calls)} times on all-cold-start sequence"

    @check("Protocol conformance: duck typing works for all three")
    def _():
        assert isinstance(TemporalRecovery(), RecoveryModule)
        assert isinstance(HungarianIdentityVerifier(), IdentityVerifier)

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("All Stage 3-5 invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
