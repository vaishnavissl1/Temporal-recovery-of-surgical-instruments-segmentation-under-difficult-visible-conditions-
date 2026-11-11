#!/usr/bin/env python3
"""
Tests for the cross-view indicators (sitr/crossview.py). No dataset, no GPU.

    python -m tests.test_crossview

The required invariants: identical masks give xv_class_jaccard 1.0, xv_area_ratio_max 0.0, xv_rowprofile_l1 0.0 and
xv_rowspan_iou 1.0; deleting one class gives xv_class_jaccard < 1 and xv_area_ratio_max 1.0. Identical masks only agree
exactly when both views cover the same columns, so those two tests run with the field-of-view guard off; the guard is
tested separately (it must shrink the retained window and it must change nothing about row-only quantities that do not
depend on column position).
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr import crossview as XV  # noqa: E402
from sitr.stereo import Rectifier, Rig, VerticalCorrection  # noqa: E402

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


IDS = (1, 2, 3)


def identity_rect(h=64, w=80):
    """A rectifier stand-in whose maps are the identity: rectified grid == label grid."""
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ones = np.ones((h, w), bool)
    return SimpleNamespace(mapL=(xs, ys), mapR=(xs.copy(), ys.copy()), valid_left=ones, valid_right=ones.copy(),
                           rig=SimpleNamespace(size=(w, h)))


def scene(h=64, w=80):
    m = np.zeros((h, w), np.uint8)
    m[8:30, 10:40] = 1          # a large class-1 blob
    m[40:56, 50:70] = 2         # class 2
    m[44:50, 15:22] = 2         # a second class-2 component
    m[20:26, 60:72] = 3
    return m


GOOD = """Width: 1280
Height: 1024
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


def real_rectifier():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "camera_calibration.txt")
        open(p, "w").write(GOOD)
        return Rectifier(Rig.from_file(p), alpha=1.0)


def main() -> int:
    print("\nRequired invariants (guard off, identity geometry)")
    rect, m = identity_rect(), scene()

    @check("identical masks: jaccard 1, area ratio 0, row profile 0, row-span IoU 1")
    def _():
        v = XV.as_dict(XV.crossview_indicators(m, m.copy(), rect, None, IDS, guard_full_px=0))
        assert v["xv_class_jaccard"] == 1.0, v
        assert v["xv_area_ratio_max"] == 0.0, v
        assert v["xv_rowprofile_l1"] == 0.0, v
        assert v["xv_rowspan_iou"] == 1.0, v
        assert v["xv_ncomp_diff"] == 0.0 and v["xv_runcount_corr"] == 1.0, v

    @check("deleting one class from one view: jaccard < 1 and area ratio exactly 1.0")
    def _():
        r = m.copy()
        r[r == 3] = 0
        v = XV.as_dict(XV.crossview_indicators(m, r, rect, None, IDS, guard_full_px=0))
        assert v["xv_class_jaccard"] < 1.0, v
        assert v["xv_area_ratio_max"] == 1.0, v
        assert v["xv_rowspan_iou"] < 1.0 and v["xv_rowprofile_l1"] > 0.0, v

    print("\nEmpty cases (no NaN may reach the gate)")

    @check("no instrument class in either view returns the stated constants")
    def _():
        z = np.zeros_like(m)
        v = XV.crossview_indicators(z, z, rect, None, IDS, guard_full_px=0)
        assert tuple(v) == XV.EMPTY_BOTH and not any(np.isnan(v)), v

    @check("one view empty, the other not: finite, jaccard 0, area ratio 1, span IoU 0")
    def _():
        v = XV.as_dict(XV.crossview_indicators(m, np.zeros_like(m), rect, None, IDS, guard_full_px=0))
        assert all(np.isfinite(list(v.values()))), v
        assert v["xv_class_jaccard"] == 0.0 and v["xv_area_ratio_max"] == 1.0 and v["xv_rowspan_iou"] == 0.0, v

    @check("a class absent from both views is skipped (adding an unused id changes nothing)")
    def _():
        a = XV.crossview_indicators(m, m, rect, None, (1, 2, 3), guard_full_px=0)
        b = XV.crossview_indicators(m, m, rect, None, (1, 2, 3, 6, 7), guard_full_px=0)
        assert np.allclose(a, b)

    @check("Pearson with a constant vector is defined (1.0 if both equal constants, else 0.0), never NaN")
    def _():
        assert XV._pearson(np.ones(5), np.ones(5)) == 1.0
        assert XV._pearson(np.ones(5), np.arange(5)) == 0.0
        assert abs(XV._pearson(np.arange(5), np.arange(5)) - 1.0) < 1e-12

    @check("all outputs are finite on random label maps")
    def _():
        rng = np.random.default_rng(0)
        for _i in range(30):
            a = rng.integers(0, 4, (64, 80)).astype(np.uint8)
            b = rng.integers(0, 4, (64, 80)).astype(np.uint8)
            v = XV.crossview_indicators(a, b, rect, None, IDS)
            assert np.all(np.isfinite(v)) and len(v) == 6

    print("\nGeometry and the field-of-view guard")

    @check("the guard scales to the grid: 186 px at 1280 wide is ceil(186*W/1280) on a W-wide grid")
    def _():
        r = identity_rect(256, 1280)                  # full-resolution rig; the label grid below is 4x coarser
        mm = np.zeros((64, 320), np.uint8)
        mm[10:20, 100:200] = 1
        _, info = XV.crossview_indicators(mm, mm, r, None, IDS, with_info=True)
        assert info["guard_px"] == int(np.ceil(186.0 * 320 / 1280)), info          # 47 on a 320-wide grid
        assert abs(info["surviving_left"] - (320 - 47) / 320) < 1e-9, info
        assert abs(info["surviving_right"] - (320 - 47) / 320) < 1e-9, info

    @check("the guard drops the LEFT strip of the left view and the RIGHT strip of the right view")
    def _():
        h, w = 20, 100
        r = identity_rect(h, w)
        left = np.zeros((h, w), np.uint8)
        right = np.zeros((h, w), np.uint8)
        left[:, :10] = 1            # only in the left edge strip of the left view: must be dropped
        right[:, 90:] = 1           # only in the right edge strip of the right view: must be dropped
        v = XV.as_dict(XV.crossview_indicators(left, right, r, None, IDS, guard_full_px=15))
        assert tuple(v.values()) == XV.EMPTY_BOTH, v      # nothing survives in either view

    @check("without the guard the same input is a total mismatch, so the guard is what removes the bias")
    def _():
        h, w = 20, 100
        r = identity_rect(h, w)
        left = np.zeros((h, w), np.uint8)
        right = np.zeros((h, w), np.uint8)
        left[:, :10] = 1
        right[:, 90:] = 1
        v = XV.as_dict(XV.crossview_indicators(left, right, r, None, IDS, guard_full_px=0))
        assert v["xv_area_ratio_max"] == 0.0 and v["xv_class_jaccard"] == 1.0   # equal areas, same class set
        assert v["xv_rowprofile_l1"] == 0.0     # the same rows are occupied

    @check("pixels outside valid_left are ignored (valid_left intersection)")
    def _():
        r = identity_rect(30, 60)
        r.valid_left = r.valid_left.copy()
        r.valid_left[:, 30:] = False
        a = np.zeros((30, 60), np.uint8)
        a[5:15, 40:50] = 1          # lies wholly in the invalid part of the left view
        _, info = XV.crossview_indicators(a, a, r, None, IDS, guard_full_px=0, with_info=True)
        assert abs(info["surviving_left"] - 0.5) < 1e-9 and info["surviving_right"] == 1.0, info
        v = XV.as_dict(XV.crossview_indicators(a, a, r, None, IDS, guard_full_px=0))
        assert v["xv_class_jaccard"] == 0.0 and v["xv_area_ratio_max"] == 1.0, v

    @check("row-based indicators ignore a horizontal shift of the content (no disparity needed)")
    def _():
        r = identity_rect(64, 200)
        a = np.zeros((64, 200), np.uint8)
        a[10:30, 40:70] = 1
        b = np.roll(a, 25, axis=1)                    # same object, shifted by a disparity-like offset
        v = XV.as_dict(XV.crossview_indicators(a, b, r, None, IDS, guard_full_px=0))
        assert v["xv_rowprofile_l1"] == 0.0 and v["xv_rowspan_iou"] == 1.0 and v["xv_area_ratio_max"] == 0.0, v

    @check("row-based indicators do see a vertical shift (the rectification claim)")
    def _():
        r = identity_rect(64, 200)
        a = np.zeros((64, 200), np.uint8)
        a[10:30, 40:70] = 1
        b = np.roll(a, 15, axis=0)
        v = XV.as_dict(XV.crossview_indicators(a, b, r, None, IDS, guard_full_px=0))
        assert v["xv_rowprofile_l1"] > 0.0 and v["xv_rowspan_iou"] < 1.0, v

    @check("labels are never interpolated: a real rectifier returns only ids that were in the input")
    def _():
        rc = real_rectifier()
        a = np.zeros((128, 160), np.uint8)
        a[20:60, 30:90] = 1
        a[70:100, 100:140] = 2
        g = XV._grid_maps(rc, a.shape)
        out = XV._rectify(a, g["L"])
        assert set(np.unique(out)).issubset({0, 1, 2, XV.OUTSIDE}), np.unique(out)

    @check("real rectifier, identical masks, guard on: finite, six values, surviving fraction in (0,1)")
    def _():
        rc = real_rectifier()
        a = np.zeros((128, 160), np.uint8)
        a[20:60, 30:90] = 1
        a[70:100, 100:140] = 2
        v, info = XV.crossview_indicators(a, a, rc, None, (1, 2), with_info=True)
        assert len(v) == 6 and np.all(np.isfinite(v)), v
        assert 0.0 < info["surviving_left"] < 1.0 and 0.0 < info["surviving_right"] < 1.0, info
        assert info["guard_px"] == int(np.ceil(186.0 * 160 / 1280))

    @check("an applied VerticalCorrection moves the right view's rows; an unapplied one changes nothing")
    def _():
        rc = real_rectifier()
        a = np.zeros((128, 160), np.uint8)
        a[40:70, 40:100] = 1
        off = VerticalCorrection(a=-8.0, b=0.0, c=0.0, applied=True)       # 8 full-res px, 2 rows on a 4x-coarser grid
        none = VerticalCorrection(a=-8.0, applied=False)
        v0 = XV.crossview_indicators(a, a, rc, None, (1,))
        v1 = XV.crossview_indicators(a, a, rc, none, (1,))
        v2 = XV.crossview_indicators(a, a, rc, off, (1,))
        assert np.allclose(v0, v1)
        assert not np.allclose(v0, v2)

    print(f"\n{'='*66}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED: {n} -> {e}")
        return 1
    print("All cross-view invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
