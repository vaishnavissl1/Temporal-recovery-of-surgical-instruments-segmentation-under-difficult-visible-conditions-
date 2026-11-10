#!/usr/bin/env python3
"""
Tests for the calibrated-stereo module (sitr/stereo.py). No dataset, no GPU.

    python -m tests.test_stereo

They cover the failures found in the REAL calibration files (missing skew,
CRLF, a corrupted trailing line), the rectification geometry, disparity
recovery on a known synthetic shift, and the nearest-neighbour label warp.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.stereo import (Rectifier, Rig, SGBMParams, disparity_pair,  # noqa: E402
                         lr_consistent, parse_calibration_text, right_path)

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


GOOD = """Width: 1280   // w
Height: 1024  // h
Camera-0-F: 1084.21 1084.05
Camera-0-C: 580.02 506.79
Camera-0-Alpha: 0.00000
Camera-0-K: -0.00069 0.00195 0.00018 0.00000 0.00000
Camera-1-F: 1083.43 1083.22
Camera-1-C: 680.91 505.81
Camera-1-Alpha: 0.00000
Camera-1-K: -0.00085 0.00245 0.00004 0.00000 0.00000
Extrinsic-Omega: -0.0002 -0.0011 -0.0000
Extrinsic-T: -4.3499 0.0333 -0.0369
"""


def main() -> int:
    print("\nCalibration parsing")

    @check("parser reads a normal file and Rodrigues/T come through")
    def _():
        c = parse_calibration_text(GOOD)
        assert c["Camera-0-F"][0] == 1084.21 and c["Extrinsic-T"][0] == -4.3499
        assert c["Width"] == [1280.0] and c["Height"] == [1024.0]

    @check("missing Camera-0-Alpha defaults to 0 (8 of 19 real files omit it)")
    def _():
        txt = "\n".join(l for l in GOOD.splitlines() if "Camera-0-Alpha" not in l)
        c = parse_calibration_text(txt)
        assert c["Camera-0-Alpha"] == [0.0]

    @check("CRLF line endings and no final newline (real seq_14) parse")
    def _():
        txt = GOOD.replace("\n", "\r\n").rstrip("\r\n")
        c = parse_calibration_text(txt)
        assert c["Extrinsic-T"] == [-4.3499, 0.0333, -0.0369]

    @check("a corrupted trailing dict line (real seq_13/15/16) is ignored")
    def _():
        junk = ("\n\n{'Header': {'timestamp': 1531274945043147, 'entry_size': 166}, "
                "'Entry': {'intrinsics': [{'alpha': 0.0, 'c': [, ], 'k': [], "
                "'f': [900390625, 986328125]}], 'extrinsics': {'om': [], 'T': []}}}\n")
        c = parse_calibration_text(GOOD + junk)
        assert c["Camera-1-C"] == [680.91, 505.81]

    @check("scientific notation is accepted (real seq_13: -1.53345e-05)")
    def _():
        txt = GOOD.replace("-0.0002 -0.0011 -0.0000", "-1.53345e-05 -0.00118 -1.61941e-05")
        c = parse_calibration_text(txt)
        assert abs(c["Extrinsic-Omega"][0] + 1.53345e-05) < 1e-12

    @check("a file with a missing required field raises instead of guessing")
    def _():
        txt = "\n".join(l for l in GOOD.splitlines() if "Extrinsic-T" not in l)
        try:
            parse_calibration_text(txt)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    @check("right_path maps left_frames/x.png to right_frames/x.png only")
    def _():
        p = os.path.join("a", "seq_1", "left_frames", "frame007.png")
        assert right_path(p) == os.path.join("a", "seq_1", "right_frames", "frame007.png")
        try:
            right_path(os.path.join("a", "labels", "x.png"))
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    print("\nRectification")

    def rig_from(text):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "camera_calibration.txt")
            with open(p, "w") as f:
                f.write(text)
            return Rig.from_file(p)

    @check("rig: baseline is |T| and R is a rotation matrix")
    def _():
        rig = rig_from(GOOD)
        assert abs(rig.baseline_mm - 4.3501) < 1e-3
        assert np.allclose(rig.R @ rig.R.T, np.eye(3), atol=1e-9)

    @check("rectified focal/baseline reproduce d = f*B/Z (B = 4.35 mm)")
    def _():
        rc = Rectifier(rig_from(GOOD), alpha=1.0)
        assert abs(rc.baseline_mm - 4.35) < 0.01, rc.baseline_mm
        # alpha=1 keeps every source pixel, which zooms OUT: the rectified focal
        # is smaller than the original 1084 px (about 998 px here), so rectified
        # disparity is focal * B / Z with the rectified focal, not the original.
        assert 800 < rc.focal < 1200, rc.focal
        z = rc.depth_mm(np.array([rc.focal * rc.baseline_mm / 100.0]))
        assert abs(z[0] - 100.0) < 1e-6

    @check("zero-disparity rectification aligns the principal points")
    def _():
        rc = Rectifier(rig_from(GOOD), alpha=1.0)
        assert abs(rc.P1[0, 2] - rc.P2[0, 2]) < 1e-6 and abs(rc.P1[1, 2] - rc.P2[1, 2]) < 1e-6

    @check("ideal rig (same fy, cy, no rotation, no distortion): left and right "
           "maps read the SAME source row (epipolar alignment)")
    def _():
        ideal = GOOD.replace("1083.43 1083.22", "1084.21 1084.05") \
                    .replace("680.91 505.81", "680.91 506.79") \
                    .replace("-0.0002 -0.0011 -0.0000", "0 0 0") \
                    .replace("-4.3499 0.0333 -0.0369", "-4.35 0 0")
        txt = "\n".join(l if "-K:" not in l and "Camera-0-K" not in l and "Camera-1-K" not in l
                        else l.split(":")[0] + ": 0 0 0 0 0" for l in ideal.splitlines())
        rc = Rectifier(rig_from(txt), alpha=1.0)
        # the zoom of alpha=1 rescales rows about the principal row, so rows do
        # not map to themselves; what must hold is that, for every rectified
        # pixel, both cameras sample the same source row
        dy = np.abs(rc.mapL[1] - rc.mapR[1])
        assert dy.max() < 0.05, dy.max()

    @check("label warp is nearest-neighbour: only original ids or 255 appear")
    def _():
        rc = Rectifier(rig_from(GOOD), alpha=1.0)
        rng = np.random.default_rng(0)
        m = rng.integers(0, 12, (1024, 1280)).astype(np.uint8)
        out = rc.rectify_labels(m)
        assert set(np.unique(out).tolist()) <= set(range(12)) | {255}
        assert (out == 255).any()          # the offset sensors leave a border

    @check("evidence round trip: rectify then pull back onto the 512x640 grid == resized original")
    def _():
        rc = Rectifier(rig_from(GOOD), alpha=1.0)
        yy, xx = np.mgrid[0:1024, 0:1280].astype(np.float32)
        img = (127 + 60 * np.sin(xx / 90.0) + 60 * np.cos(yy / 70.0)).astype(np.uint8)
        rect = rc._remap(img, rc.mapL, cv2.INTER_LINEAR, 0)
        back = rc.to_original_grid(rect, (512, 640), interp=cv2.INTER_LINEAR)
        direct = cv2.resize(img, (640, 512), interpolation=cv2.INTER_AREA)
        inner = np.s_[40:-40, 40:-40]
        err = np.abs(back[inner].astype(np.float32) - direct[inner].astype(np.float32))
        assert err.mean() < 2.0, err.mean()

    @check("right->left warp: constant map stays constant where a disparity exists, 255 elsewhere")
    def _():
        from sitr.stereo import right_to_left_grid
        rc = Rectifier(rig_from(GOOD), alpha=1.0)
        disp = np.full((512, 640), 90.0, np.float32); disp[:100, :] = np.nan
        out = right_to_left_grid(np.full((512, 640), 5, np.uint8), disp, rc, None, (512, 640))
        assert (out[:100] == 255).all()
        inner = out[200:480, 150:600]
        assert (inner == 5).mean() > 0.95, (inner == 5).mean()
        assert set(np.unique(out).tolist()) <= {5, 255}

    @check("right->left warp geometry: with zero disparity a pattern shifts by the principal-point offset (~50 working px)")
    def _():
        from sitr.stereo import right_to_left_grid
        ideal = GOOD.replace("1083.43 1083.22", "1084.21 1084.05").replace("680.91 505.81", "680.91 506.79") \
                    .replace("-0.0002 -0.0011 -0.0000", "0 0 0").replace("-4.3499 0.0333 -0.0369", "-4.35 0 0")
        txt = "\n".join(l if "Camera-0-K" not in l and "Camera-1-K" not in l
                        else l.split(":")[0] + ": 0 0 0 0 0" for l in ideal.splitlines())
        rc = Rectifier(rig_from(txt), alpha=1.0)
        cols = (np.arange(640)[None, :].repeat(512, 0) // 3).astype(np.uint8)       # label = working column // 3 (max 213)
        out = right_to_left_grid(cols, np.zeros((512, 640), np.float32),
                                 rc, None, (512, 640), interp=cv2.INTER_NEAREST)
        # left working column u maps to right working column ~ u + (cx_R - cx_L)/2
        u = np.arange(100, 400)
        got = out[256, u].astype(np.float32) * 3 + 1.0
        assert (out[256, u] != 255).all()
        shift = np.median(got - u)
        assert 40 < shift < 60, shift
        assert np.std(got - u - shift) < 3.5

    print("\nStereo indicators")

    @check("stereo indicators: fractions over PREDICTED-instrument pixels in view only")
    def _():
        from sitr.stereo import instrument_stereo_indicators as ind
        pred = np.zeros((10, 10), np.uint8); pred[:, :4] = 1          # instrument = columns 0-3
        in_view = np.ones((10, 10), bool); in_view[:, 0] = False       # column 0 outside the view
        lr = np.ones((10, 10), bool); lr[:, 1] = False                 # column 1 fails LR
        occ = np.zeros((10, 10), bool); occ[:, 2] = True               # column 2 occluded
        conf = np.full((10, 10), 255, np.uint8); conf[:, 3] = 0        # column 3 no confidence
        sd, oc, dc = ind(pred, lr, occ, conf, in_view, [1])
        assert abs(sd - 1 / 3) < 1e-6 and abs(oc - 1 / 3) < 1e-6 and abs(dc - 2 / 3) < 1e-6
        assert ind(np.zeros((10, 10), np.uint8), lr, occ, conf, in_view, [1]) == (0.0, 0.0, 0.0)

    @check("SequenceCache stereo fields round-trip and require() enforces them")
    def _():
        import tempfile
        from sitr.reliability import SequenceCache
        T = 5
        c = SequenceCache(name="train/seq_1", q=np.ones(T), area=np.ones(T, np.int64) * 9,
                          emb=np.ones((T, 4)), masks=np.zeros((T, 4, 4), np.uint8),
                          r_star=np.ones(T), ent=np.ones(T), margin=np.ones(T), bnd=np.ones(T),
                          frag=np.ones(T), num_classes=3, instrument_ids=(1,))
        assert not c.has_stereo
        for name in ("extended_stereo", "nontemporal_stereo", "stereo3"):
            try:
                c.require(name)
            except ValueError:
                pass
            else:
                raise AssertionError(name)
        c.require("extended"); c.require("nontemporal")
        c.sd, c.oc, c.dc = np.full(T, .1), np.full(T, .2), np.full(T, .9)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.npz"); c.save(p)
            c2 = SequenceCache.load(p)
        c2.require("extended_stereo")
        assert np.allclose(c2.sd, .1) and np.allclose(c2.dc, .9)

    @check("indicator sets: nontemporal has no T/A/F/Drift, stereo sets add exactly the 3")
    def _():
        from sitr.reliability import indicator_names as nm
        tmp = {"T_temporal", "A_area_stability", "F_feature_cos", "Drift_prev_cos"}
        assert not (set(nm("nontemporal")) & tmp) and not (set(nm("nontemporal_stereo")) & tmp)
        assert set(nm("extended_stereo")) - set(nm("extended")) == set(nm("stereo3"))
        assert len(nm("extended_stereo")) == 12 and len(nm("extended")) == 9

    print("\nDisparity")

    def textured_pair(d=20, h=240, w=320):
        rng = np.random.default_rng(1)
        base = rng.integers(0, 255, (h, w + 60), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (0, 0), 1.2)
        base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
        left = base[:, 30:30 + w].copy()
        right = base[:, 30 + d:30 + d + w].copy()      # left(x) = right(x - d)
        return left, right

    @check("SGBM recovers a known 20 px shift; LR check keeps most pixels")
    def _():
        L, R = textured_pair(20)
        p = SGBMParams(num_disp=64, block=5, uniqueness=5)
        dl, dr = disparity_pair(L, R, p)
        inner = np.s_[20:-20, 90:-20]
        med = float(np.nanmedian(dl[inner]))
        assert abs(med - 20.0) < 1.0, med
        ok = lr_consistent(dl, dr)
        assert ok[inner].mean() > 0.6, ok[inner].mean()

    @check("LR check rejects a left disparity the right view contradicts")
    def _():
        dl = np.full((4, 40), 5.0, np.float32)
        dr = np.full((4, 40), 5.0, np.float32)
        assert lr_consistent(dl, dr)[:, 10:].all()
        dr_bad = np.full((4, 40), 12.0, np.float32)
        assert not lr_consistent(dl, dr_bad).any()
        dl_nan = dl.copy(); dl_nan[:] = np.nan
        assert not lr_consistent(dl_nan, dr).any()

    @check("occlusion: a NEARER surface at the match location hides the left pixel")
    def _():
        from sitr.stereo import occlusion_mask
        dl = np.full((3, 40), 10.0, np.float32)
        near = np.full((3, 40), 16.0, np.float32)     # right view sees a nearer surface
        same = np.full((3, 40), 10.0, np.float32)
        far = np.full((3, 40), 4.0, np.float32)       # smaller: false match, NOT occlusion
        assert occlusion_mask(dl, near)[:, 12:].all()
        assert not occlusion_mask(dl, same).any()
        assert not occlusion_mask(dl, far).any()

    @check("compute_evidence returns aligned maps of the right shape and dtype")
    def _():
        from sitr.stereo import compute_evidence, SGBMParams as P
        rc = Rectifier(rig_from(GOOD), alpha=1.0)
        rng = np.random.default_rng(7)
        base = cv2.GaussianBlur(rng.integers(0, 255, (1024, 1400), dtype=np.uint8), (0, 0), 1.5)
        L = cv2.cvtColor(base[:, 60:1340].copy(), cv2.COLOR_GRAY2BGR)
        R = cv2.cvtColor(base[:, 100:1380].copy(), cv2.COLOR_GRAY2BGR)
        ev = compute_evidence(L, R, rc, None, P(num_disp=192, block=5, uniqueness=5), (512, 640))
        assert ev.disparity.shape == (512, 640) and ev.disparity.dtype == np.float16
        for m in (ev.lr_ok, ev.occluded, ev.specular, ev.in_view):
            assert m.shape == (512, 640) and m.dtype == bool
        assert ev.confidence.shape == (512, 640) and ev.confidence.dtype == np.uint8
        assert ev.in_view.mean() > 0.8
        assert ev.lr_ok.mean() > 0.05 and (ev.confidence[ev.lr_ok] > 0).any()
        assert not (ev.lr_ok & ~ev.in_view).any()          # evidence only inside the view
        assert ev.image_cues.shape == (6,) and ev.image_cues.dtype == np.float32

    @check("LR check rejects a match that points outside the image")
    def _():
        dl = np.full((2, 10), 8.0, np.float32)       # x=0..7 point at x<0
        dr = np.full((2, 10), 8.0, np.float32)
        ok = lr_consistent(dl, dr)
        assert not ok[:, :8].any() and ok[:, 8:].all()

    print("\nPer-sequence vertical self-calibration")
    from sitr.stereo import VerticalCorrection, _tukey_fit  # noqa: E402

    @check("robust fit recovers a + b*x + c*y despite 30% false matches")
    def _():
        rng = np.random.default_rng(3)
        n = 600
        x, y = rng.uniform(0, 1280, n), rng.uniform(0, 1024, n)
        true = np.array([-9.0, 5.5, 1.5])
        dy = true[0] + true[1] * x / 1000 + true[2] * y / 1000 + rng.normal(0, 0.4, n)
        bad = rng.random(n) < 0.30
        dy[bad] = rng.uniform(-20, 20, bad.sum())              # false matches
        A = np.c_[np.ones(n), x / 1000, y / 1000]
        beta, mad = _tukey_fit(A, dy)
        assert np.allclose(beta, true, atol=0.5), beta
        assert mad < 0.8, mad

    @check("correction moves a feature at right row y to row y + dy (matches left)")
    def _():
        img = np.zeros((200, 300), np.uint8); img[100, :] = 255
        out = VerticalCorrection(a=4.0, b=0.0, c=0.0, applied=True).apply(img)
        assert out[104, 150] > 200 and out[100, 150] < 50, (out[104, 150], out[100, 150])

    @check("a correction flagged not-applied is the identity (noise-level fits)")
    def _():
        img = (np.arange(200 * 300) % 251).astype(np.uint8).reshape(200, 300)
        assert np.array_equal(VerticalCorrection(a=0.4, applied=False).apply(img), img)

    @check("tilt in x is applied per column (dy = b*x/1000)")
    def _():
        img = np.zeros((300, 1000), np.uint8); img[150, :] = 255
        out = VerticalCorrection(a=0.0, b=10.0, c=0.0, applied=True).apply(img)
        # column x=0 -> shift 0 ; column x=900 -> shift 9 rows
        assert out[150, 0] > 200 and out[159, 900] > 200

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("All stereo invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
