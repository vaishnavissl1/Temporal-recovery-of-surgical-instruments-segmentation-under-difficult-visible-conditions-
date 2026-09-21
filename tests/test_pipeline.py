#!/usr/bin/env python3
"""
Self-contained correctness tests. No dataset, no GPU, no pytest required.

    python -m tests.test_pipeline

These are the invariants that, if broken, corrupt results *silently* rather than
crashing — which is the only kind of bug worth writing a test for here. Run this
after any change, and before you trust a number.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.align import build_align  # noqa: E402
from sitr.labels import IGNORE_INDEX, is_junk_path  # noqa: E402
from sitr.losses import SegLoss  # noqa: E402
from sitr.metrics import (ConfusionMatrix, frame_iou, instrument_iou,  # noqa: E402
                          pearson, roc_auc, spearman)
from sitr.models import (ARCHITECTURES, ReliabilityMLP, build_model,  # noqa: E402
                         count_parameters, encoder_param_names,
                         predict_with_tta)
from sitr.reliability import (INDICATOR_SETS, SequenceCache,  # noqa: E402
                              StreamConfig, indicator_names, run_sequence,
                              sweep_threshold)
from sitr.stage34_interface import (FrameState, IdentityResult,  # noqa: E402
                                    IdentityVerifier, RecoveryModule,
                                    RecoveryResult, SelectiveInference)

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


def _cache(T=40, extended=True, seed=0):
    r = np.random.default_rng(seed)
    kw = dict(name="train/seq_1",
              q=r.random(T).astype(np.float32),
              area=r.integers(1, 500, T).astype(np.int64),
              emb=r.random((T, 32)).astype(np.float32),
              masks=r.integers(0, 2, (T, 16, 20)).astype(np.uint8),
              r_star=r.random(T).astype(np.float32),
              gray=r.integers(0, 255, (T, 16, 20)).astype(np.uint8),
              num_classes=2, instrument_ids=(1,))
    if extended:
        kw.update({k: r.random(T).astype(np.float32)
                   for k in ("ent", "margin", "bnd", "frag")})
    return SequenceCache(**kw)


def main() -> int:
    print("Dataset hygiene")

    @check("junk paths (__MACOSX, AppleDouble, .DS_Store) are excluded")
    def _():
        for p in ("a/__MACOSX/b/left_frames", "x/._frame000.png",
                  "r/__MACOSX/r/seq_1/labels", "d/.DS_Store"):
            assert is_junk_path(p), p
        for p in ("ok/seq_1/labels", "release_1/seq_4/left_frames"):
            assert not is_junk_path(p), p

    @check("same colour under two release names merges into ONE class")
    def _():
        import json
        import shutil
        import cv2
        from sitr.labels import build_class_map
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_tmp_colour_conflict")
        shutil.rmtree(root, ignore_errors=True)
        try:
            # EndoVis 2018 really does this: (124,155,5) is "small-intestine"
            # in one release's labels.json and "intestine" in another.
            for rel, cls in (
                ("ra", [{"name": "background-tissue", "color": [0, 0, 0], "classid": 0},
                        {"name": "instrument-shaft", "color": [0, 255, 0], "classid": 1},
                        {"name": "small-intestine", "color": [124, 155, 5], "classid": 10}]),
                ("rb", [{"name": "background-tissue", "color": [0, 0, 0], "classid": 0},
                        {"name": "intestine", "color": [124, 155, 5], "classid": 10}]),
            ):
                d = os.path.join(root, rel, rel)
                os.makedirs(os.path.join(d, "seq_1", "left_frames"), exist_ok=True)
                os.makedirs(os.path.join(d, "seq_1", "labels"), exist_ok=True)
                with open(os.path.join(d, "labels.json"), "w") as f:
                    json.dump(cls, f)
                blank = np.zeros((8, 8, 3), np.uint8)
                cv2.imwrite(os.path.join(d, "seq_1", "left_frames", "frame000.png"), blank)
                cv2.imwrite(os.path.join(d, "seq_1", "labels", "frame000.png"), blank)

            cm = build_class_map(root)
            hits = [c for c in cm.classes if c.color == (124, 155, 5)]
            assert len(hits) == 1, f"colour split across {len(hits)} classes"
            assert hits[0].aliases, "the dropped name was lost, not recorded"
            assert not hits[0].instrument, "intestine is anatomy, not an instrument"
            lut = cm.rgb_lut()
            assert lut[(124 << 16) | (155 << 8) | 5] == hits[0].id
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @check("one name with two colours is still FATAL, not merged")
    def _():
        import json
        import shutil
        import cv2
        from sitr.labels import build_class_map
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_tmp_ambiguous")
        shutil.rmtree(root, ignore_errors=True)
        try:
            for rel, col in (("ra", [0, 255, 0]), ("rb", [9, 9, 9])):
                d = os.path.join(root, rel, rel)
                os.makedirs(os.path.join(d, "seq_1", "left_frames"), exist_ok=True)
                os.makedirs(os.path.join(d, "seq_1", "labels"), exist_ok=True)
                with open(os.path.join(d, "labels.json"), "w") as f:
                    json.dump([{"name": "background", "color": [0, 0, 0], "classid": 0},
                               {"name": "shaft", "color": col, "classid": 1}], f)
                blank = np.zeros((8, 8, 3), np.uint8)
                cv2.imwrite(os.path.join(d, "seq_1", "left_frames", "frame000.png"), blank)
                cv2.imwrite(os.path.join(d, "seq_1", "labels", "frame000.png"), blank)
            try:
                build_class_map(root)
            except ValueError:
                return
            raise AssertionError("accepted an ambiguous label encoding")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @check("unlabelled TEST frames are dropped; unlabelled TRAIN frames are fatal")
    def _():
        import shutil
        import cv2
        from sitr.dataset import discover_sequences
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_tmp_partial_labels")
        shutil.rmtree(root, ignore_errors=True)

        def mk(base, n, drop=()):
            os.makedirs(os.path.join(base, "left_frames"), exist_ok=True)
            os.makedirs(os.path.join(base, "labels"), exist_ok=True)
            blank = np.zeros((8, 8, 3), np.uint8)
            for t in range(n):
                cv2.imwrite(os.path.join(base, "left_frames", f"frame{t:03d}.png"), blank)
                if t not in drop:
                    cv2.imwrite(os.path.join(base, "labels", f"frame{t:03d}.png"), blank)
        try:
            # The real test release does this: seq_3 has 250 frames, 249 labels.
            mk(os.path.join(root, "test_data", "test_data", "seq_3-x-001", "seq_3"),
               10, drop={9})
            mk(os.path.join(root, "rel", "rel", "seq_1"), 10)
            te = discover_sequences(root, splits=("test",))
            assert len(te) == 1 and len(te[0]) == 9, [len(s) for s in te]
            assert all(l is not None for l in te[0].labels)

            mk(os.path.join(root, "rel", "rel", "seq_2"), 10, drop={3})
            try:
                discover_sequences(root, splits=("train",))
            except FileNotFoundError:
                return
            raise AssertionError("a TRAINING frame with no label was accepted")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @check("split detection is relative to the root (a 'test' in the parent path)")
    def _():
        import shutil
        import cv2
        from sitr.dataset import discover_sequences
        # The root itself sits under a directory containing "test" -- exactly
        # what happens if the dataset lives in C:/latest/... or a tests/ folder.
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_tmp_in_test_dir", "endovis")
        shutil.rmtree(os.path.dirname(root), ignore_errors=True)
        try:
            for sub in ("left_frames", "labels"):
                os.makedirs(os.path.join(root, "rel", "rel", "seq_1", sub),
                            exist_ok=True)
            blank = np.zeros((8, 8, 3), np.uint8)
            for t in range(4):
                for sub in ("left_frames", "labels"):
                    cv2.imwrite(os.path.join(root, "rel", "rel", "seq_1", sub,
                                             f"frame{t:03d}.png"), blank)
            tr = discover_sequences(root, splits=("train",))
            assert len(tr) == 1 and tr[0].split == "train", \
                f"misfiled as {[s.split for s in tr]} -- training split empties"
        finally:
            shutil.rmtree(os.path.dirname(root), ignore_errors=True)

    print("\nMetrics")

    @check("IoU edge cases (empty/empty, disjoint, identical)")
    def _():
        z, o = np.zeros((8, 8), np.uint8), np.ones((8, 8), np.uint8)
        assert instrument_iou(z, z, 2, [1], empty_value=1.0) == 1.0
        assert instrument_iou(z, z, 2, [1], empty_value=0.0) == 0.0
        assert instrument_iou(z, o, 2, [1]) == 0.0
        assert frame_iou(o, o, 2, present="union") == 1.0

    @check("ignore-index pixels are excluded from the confusion matrix")
    def _():
        cm = ConfusionMatrix(2)
        gt = np.full((4, 4), IGNORE_INDEX, np.uint8)
        gt[0, 0] = 1
        cm.update(np.ones((4, 4), np.uint8), gt)
        assert cm.mat.sum() == 1, cm.mat

    @check("rank statistics and AUC")
    def _():
        assert abs(roc_auc(np.array([3., 2., 1., 0.]), np.array([1, 1, 0, 0])) - 1.0) < 1e-9
        assert spearman(np.arange(10), np.arange(10) ** 2) > 0.999
        assert np.isnan(pearson(np.ones(5), np.arange(5)))

    print("\nModels")

    @check("every architecture: shapes, gradients, encoder group, TTA")
    def _():
        x = torch.randn(2, 3, 64, 96)
        tgt = torch.randint(0, 3, (2, 64, 96))
        tgt[0, :4, :4] = IGNORE_INDEX
        for a in ARCHITECTURES:
            m = build_model(3, arch=a, pretrained=False)
            lo, ft = m(x)
            assert lo.shape == (2, 3, 64, 96), (a, lo.shape)
            assert ft.shape[-2:] == (64, 96), (a, ft.shape)
            enc = encoder_param_names(a)
            n_enc = sum(p.numel() for n, p in m.named_parameters()
                        if n.split(".")[0] in enc)
            assert n_enc > 0, f"{a}: encoder lr scaling would be a silent no-op"
            loss, parts = SegLoss(1, 1, 0.5)(lo, tgt)
            loss.backward()
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in m.parameters()), a
            assert parts["tversky"] > 0, a
            m.eval()
            with torch.no_grad():
                pr, _ = predict_with_tta(m, x)
            assert torch.allclose(pr.sum(1), torch.ones(2, 64, 96), atol=1e-4), a

    @check("loss with lambda_tversky=0 reduces to the report's eq. (29)")
    def _():
        lo = torch.randn(2, 3, 8, 8, requires_grad=True)
        tgt = torch.randint(0, 3, (2, 8, 8))
        _t, parts = SegLoss(1, 1, 0.0)(lo, tgt)
        assert parts["tversky"] == 0.0

    print("\nReliability (Stage 2)")

    @check("indicator sets wire through to the right feature width")
    def _():
        for setname in INDICATOR_SETS:
            out = run_sequence(_cache(), StreamConfig(indicators=setname))
            assert out["feats"].shape[1] == len(indicator_names(setname))

    @check("a paper4-only cache REFUSES the extended set instead of zero-filling")
    def _():
        try:
            run_sequence(_cache(extended=False), StreamConfig(indicators="extended"))
        except ValueError:
            return
        raise AssertionError("silently accepted a cache missing indicators")

    @check("eq. (10): empty foreground zeroes every indicator")
    def _():
        for setname in INDICATOR_SETS:
            c = _cache()
            c.area[:] = 0
            o = run_sequence(c, StreamConfig(indicators=setname))
            assert o["degenerate"].all() and np.allclose(o["feats"], 0), setname

    @check("cold start: empty memory leaves T/A/F at zero and is flagged")
    def _():
        c = _cache()
        c.r_star[:] = 0.0                       # nothing ever becomes reliable
        o = run_sequence(c, StreamConfig(tau=0.5, indicators="paper4"))
        assert o["cold"].all()
        assert np.allclose(o["feats"][:, 1:4], 0)

    @check("both alignment operators run and preserve label values")
    def _():
        for al in ("identity", "farneback"):
            c = _cache()
            o = run_sequence(c, StreamConfig(indicators="paper4"),
                             align=build_align(al))
            assert np.isfinite(o["feats"]).all(), al
        a = build_align("farneback")
        m = np.random.default_rng(0).integers(0, 4, (16, 20)).astype(np.uint8)
        g1 = np.random.default_rng(1).integers(0, 255, (16, 20)).astype(np.uint8)
        w = a(m, g1, g1)
        assert set(np.unique(w)).issubset(set(np.unique(m)) | {0}), \
            "warp invented a class that does not exist"

    @check("ground truth never leaks into the inference-time features")
    def _():
        c1, c2 = _cache(seed=1), _cache(seed=1)
        c2.r_star = np.zeros_like(c2.r_star)    # different target, same inputs
        cfg = StreamConfig(tau=-1.0)            # force identical history
        f1 = run_sequence(c1, cfg)["feats"]
        f2 = run_sequence(c2, cfg)["feats"]
        assert np.allclose(f1, f2), "features changed with r* alone -> leak"

    @check("tau sweep returns a full grid and a finite choice")
    def _():
        r = np.random.default_rng(0)
        s = sweep_threshold(r.random(200), r.random(200))
        assert len(s["grid"]) > 50 and 0 < s["tau"] < 1
        assert all(0 <= row["rho"] <= 1 for row in s["grid"])

    @check("ReliabilityMLP accepts both indicator widths and stays tiny")
    def _():
        for n in (4, 9):
            m = ReliabilityMLP(in_dim=n)
            out = m(torch.randn(5, n))
            assert out.shape == (5,) and (0 <= out).all() and (out <= 1).all()
            assert count_parameters(m) < 20000, "c_rel must stay negligible"

    print("\nStage 3-5 handoff contract")

    @check("memory obeys |B_t| <= K and the [t-K, t-1] window, for several K")
    def _():
        class PT:
            def __call__(self, st):
                return RecoveryResult(mask=st.mask)

        for K in (1, 3, 5, 8):
            inf = SelectiveInference(None, ReliabilityMLP(), instrument_ids=[1],
                                     num_classes=2, tau=0.5, memory_size=K,
                                     recovery=PT())
            rng = np.random.default_rng(0)
            for t in range(60):
                r = float(rng.random())
                st = FrameState(t=t, image=np.zeros((4, 4, 3), np.uint8),
                                probs=np.zeros((2, 4, 4), np.float32),
                                mask=np.zeros((4, 4), np.uint8),
                                features=np.zeros((8, 4, 4), np.float32),
                                embedding=np.zeros(8, np.float32),
                                indicators=(r, 0, 0, 0), reliability=r,
                                reliable=r >= 0.5, foreground_area=5,
                                degenerate=False, cold_start=(t == 0))
                inf.step(st)
                assert len(st.memory) <= K, (K, len(st.memory))
                assert all(m.t >= t - K and m.t < t for m in st.memory), K
                assert len(inf.memory) <= K, (K, len(inf.memory))

    @check("memory admission uses tau_memory, NOT the routing tau")
    def _():
        # Regression test for a real bug caught by tools/verify_release.py on
        # the first real release: the runtime admitted frames to B_t at the
        # DEPLOYED tau (0.76) while Stage 2 had trained and evaluated at
        # tau_train (0.50).  Memory was therefore far stricter at inference
        # than in training -> more cold starts -> T/A/F zeroed -> lower r_t ->
        # still more frames routed.  Observed routing rate came out 0.546
        # against a reported 0.339.  These two thresholds answer different
        # questions and must stay separate.
        def run(tau, tau_memory):
            inf = SelectiveInference(None, ReliabilityMLP(), instrument_ids=[1],
                                     num_classes=2, tau=tau,
                                     tau_memory=tau_memory, memory_size=5)
            admitted = 0
            for t in range(40):
                r = t / 40.0                       # sweeps 0.000 .. 0.975
                st = FrameState(t=t, image=np.zeros((4, 4, 3), np.uint8),
                                probs=np.zeros((2, 4, 4), np.float32),
                                mask=np.zeros((4, 4), np.uint8),
                                features=np.zeros((8, 4, 4), np.float32),
                                embedding=np.zeros(8, np.float32),
                                indicators=(r, 0, 0, 0), reliability=r,
                                reliable=r >= tau, foreground_area=5,
                                degenerate=False, cold_start=not inf.memory)
                before = len(inf.memory)
                inf.step(st)
                admitted += int(len(inf.memory) > before
                                or (before == 5 and len(inf.memory) == 5
                                    and inf.memory[-1].t == t))
            return admitted

        strict = run(tau=0.76, tau_memory=0.76)
        loose = run(tau=0.76, tau_memory=0.50)
        assert loose > strict, (loose, strict)

        # and the default must stay backward-compatible: tau_memory unset
        # behaves exactly as tau
        inf = SelectiveInference(None, ReliabilityMLP(), instrument_ids=[1],
                                 num_classes=2, tau=0.61, memory_size=3)
        assert inf.tau_memory == 0.61

    @check("the release bundle reads tau_memory from the manifest, not tau")
    def _():
        import sitr.runtime as rt
        man = {"tau": 0.76, "tau_train": 0.50, "memory_size": 5}
        b = rt.ReleaseBundle.__new__(rt.ReleaseBundle)
        b.manifest = man
        assert b.tau == 0.76 and b.tau_memory == 0.50
        # an older manifest without tau_train falls back to tau rather than
        # silently using a wrong default
        b.manifest = {"tau": 0.76, "memory_size": 5}
        assert b.tau_memory == 0.76

    @check("a release finds its image data wherever it is copied to")
    def _():
        # Regression: the manifest stores the BUILD machine's absolute path to
        # the prepared images. On a teammate's machine that path is dead, so
        # the bundle must prefer <release>/prepared.
        import tempfile
        import sitr.runtime as rt
        with tempfile.TemporaryDirectory() as d:
            rel = os.path.join(d, "release")
            os.makedirs(os.path.join(rel, "prepared"))
            open(os.path.join(rel, "prepared", "prepared_meta.json"), "w").write("{}")
            b = rt.ReleaseBundle.__new__(rt.ReleaseBundle)
            b.root, b.data_root = rel, None
            b.manifest = {"data_source": {"kind": "prepared",
                                          "path": r"C:\\nowhere\\on\\this\\machine"}}
            assert b.resolve_data_root() == os.path.join(rel, "prepared")
            # explicit data_root wins
            other = os.path.join(d, "elsewhere")
            os.makedirs(other)
            open(os.path.join(other, "prepared_meta.json"), "w").write("{}")
            b.data_root = other
            assert b.resolve_data_root() == other
            # nothing valid anywhere -> a clear error, not a crash deep inside
            b.data_root = None
            os.remove(os.path.join(rel, "prepared", "prepared_meta.json"))
            try:
                b.resolve_data_root()
                raise AssertionError("expected FileNotFoundError")
            except FileNotFoundError as e:
                assert "prepared" in str(e)

    @check("reset() clears memory so it cannot cross a sequence boundary")
    def _():
        inf = SelectiveInference(None, ReliabilityMLP(), instrument_ids=[1],
                                 num_classes=2, memory_size=3)
        st = FrameState(t=0, image=np.zeros((2, 2, 3), np.uint8),
                        probs=np.zeros((2, 2, 2), np.float32),
                        mask=np.zeros((2, 2), np.uint8),
                        features=np.zeros((4, 2, 2), np.float32),
                        embedding=np.zeros(4, np.float32),
                        indicators=(1, 1, 1, 1), reliability=1.0, reliable=True,
                        foreground_area=1, degenerate=False, cold_start=True)
        inf.step(st)
        assert inf.memory
        inf.reset()
        assert not inf.memory and inf.t == 0

    @check("recovery is called ONLY on unreliable, non-cold-start frames")
    def _():
        calls = []

        class Spy:
            def __call__(self, st):
                calls.append(st.t)
                assert st.needs_recovery, f"recovery ran on frame {st.t}"
                return RecoveryResult(mask=st.mask)

        inf = SelectiveInference(None, ReliabilityMLP(), instrument_ids=[1],
                                 num_classes=2, tau=0.5, memory_size=3,
                                 recovery=Spy())
        rng = np.random.default_rng(3)
        expected = 0
        for t in range(40):
            r = float(rng.random())
            cold = not inf.memory
            st = FrameState(t=t, image=np.zeros((2, 2, 3), np.uint8),
                            probs=np.zeros((2, 2, 2), np.float32),
                            mask=np.zeros((2, 2), np.uint8),
                            features=np.zeros((4, 2, 2), np.float32),
                            embedding=np.zeros(4, np.float32),
                            indicators=(r, 0, 0, 0), reliability=r,
                            reliable=r >= 0.5, foreground_area=1,
                            degenerate=False, cold_start=cold)
            if (not st.reliable) and not cold:
                expected += 1
            inf.step(st)
        assert len(calls) == expected == inf.stats["routed"], \
            (len(calls), expected, inf.stats)

    @check("memory carries the FULL entry M_s = {S,P,E,ID,r} of eq. (14)")
    def _():
        seen = {"n": 0}

        class Spy:
            def __call__(self, st):
                for m in st.memory:
                    assert m.probs is not None, "P_s missing: eq. (18) needs it"
                    assert m.mask is not None and m.embedding is not None
                    assert 0.0 <= m.reliability <= 1.0
                seen["n"] += 1
                return RecoveryResult(mask=st.mask)

        inf = SelectiveInference(None, ReliabilityMLP(), instrument_ids=[1],
                                 num_classes=2, tau=0.5, memory_size=3,
                                 recovery=Spy())
        rng = np.random.default_rng(1)
        for t in range(40):
            r = float(rng.random())
            st = FrameState(t=t, image=np.zeros((4, 4, 3), np.uint8),
                            probs=np.full((2, 4, 4), 0.5, np.float32),
                            mask=np.zeros((4, 4), np.uint8),
                            features=np.zeros((8, 4, 4), np.float32),
                            embedding=np.ones(8, np.float32),
                            indicators=(r, 0, 0, 0), reliability=r,
                            reliable=r >= 0.5, foreground_area=4,
                            degenerate=False, cold_start=not inf.memory)
            inf.step(st)
        assert seen["n"] > 0, "recovery never ran; test proved nothing"

    @check("Protocol conformance is structural (duck typing works)")
    def _():
        class R:
            def __call__(self, st):
                return RecoveryResult(mask=st.mask)

        class I:
            def __call__(self, mask, st):
                return IdentityResult(mask=mask, assignments={})

        assert isinstance(R(), RecoveryModule)
        assert isinstance(I(), IdentityVerifier)

    print("\nStage 0 checklist (preprocessing contract)")

    @check("masks resize with nearest neighbour and invent no new classes")
    def _():
        from sitr.dataset import Preprocessor
        pre = Preprocessor(128, 160)
        lab = np.kron(np.array([[0, 1], [2, 3]], np.uint8), np.ones((64, 80), np.uint8))
        out = pre.resize_mask(lab.astype(np.uint8))
        assert set(np.unique(out).tolist()).issubset({0, 1, 2, 3}), np.unique(out)

    @check("images are ImageNet-normalised, not raw 0-255")
    def _():
        from sitr.dataset import Preprocessor
        pre = Preprocessor(64, 64)
        x = pre.normalise(np.full((64, 64, 3), 128, np.uint8))
        assert x.shape == (3, 64, 64) and abs(float(x.mean())) < 1.5

    @check("right_frames is never a loadable frame directory before left_frames")
    def _():
        from sitr.dataset import _FRAME_DIRS
        assert _FRAME_DIRS[0] == "left_frames"
        assert "right_frames" not in _FRAME_DIRS, \
            "right frames have no ground truth and must never be loaded"

    print("\nNo synthetic data on the real path")

    @check("no training/eval module imports the dummy-dataset generator")
    def _():
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        offenders = []
        for d in ("sitr", "tools"):
            for fn in sorted(os.listdir(os.path.join(root, d))):
                if not fn.endswith(".py") or fn == "make_dummy_dataset.py":
                    continue
                src = open(os.path.join(root, d, fn)).read()
                if "make_dummy_dataset" in src:
                    offenders.append(f"{d}/{fn}")
        assert not offenders, f"synthetic generator referenced by {offenders}"

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("All invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
