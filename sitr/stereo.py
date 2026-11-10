"""Calibrated stereo geometry for EndoVis 2018.

    camera_calibration.txt -> Rig -> Rectifier -> SGBM disparity -> LR check

Everything here is classical OpenCV: no learning, no GPU, deterministic.

Facts about the dataset this module is written around (DATASET_FACTS.md s.5):
  * calibration is per camera RIG, shared by several sequences (7 files, 19 seqs)
  * 8 files omit `Camera-0-Alpha` (skew) -> default 0
  * 3 files end with a corrupted dict line; 1 has CRLF and no final newline
  * Extrinsic-Omega is a Rodrigues vector (radians); Extrinsic-T is in mm
  * the sensors are offset (principal points differ by ~100 px) while the
    rotation is ~0.06 degrees, so UNRECTIFIED disparity is f*B/Z minus ~100 px.
    SGBM must run on rectified images.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .difficulty import image_cues

_KEYS =("Width", "Height", "Camera-0-F", "Camera-0-C", "Camera-0-Alpha",
         "Camera-0-K", "Camera-1-F", "Camera-1-C", "Camera-1-Alpha",
         "Camera-1-K", "Extrinsic-Omega", "Extrinsic-T")
_REQUIRED = ("Width", "Height", "Camera-0-F", "Camera-0-C", "Camera-1-F",
             "Camera-1-C", "Extrinsic-Omega", "Extrinsic-T")


# --------------------------------------------------------------------------- #
# calibration parsing
# --------------------------------------------------------------------------- #
def parse_calibration(path: str) -> Dict[str, list]:
    """Parse a camera_calibration.txt, tolerating everything found in the data.

    Only known keys are read; lines that are not `Key: numbers` (such as the
    corrupted dict at the end of some files) are ignored; CRLF and a missing
    final newline are fine; scientific notation is fine. A missing skew
    defaults to 0, missing distortion defaults to zeros.
    """
    with open(path, "r", errors="replace") as f:
        text = f.read()
    return parse_calibration_text(text, source=path)


def parse_calibration_text(text: str, source: str = "<text>") -> Dict[str, list]:
    out: Dict[str, list] = {}
    for line in text.splitlines():
        line = line.split("//")[0].strip()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        if key not in _KEYS:
            continue
        try:
            out[key] = [float(x) for x in val.split()]
        except ValueError:
            continue
    missing = [k for k in _REQUIRED if k not in out]
    if missing:
        raise ValueError(f"{source}: calibration is missing required fields {missing}")
    for k in ("Camera-0-Alpha", "Camera-1-Alpha"):
        out.setdefault(k, [0.0])
    for k in ("Camera-0-K", "Camera-1-K"):
        out.setdefault(k, [0.0] * 5)
    if len(out["Extrinsic-Omega"]) != 3 or len(out["Extrinsic-T"]) != 3:
        raise ValueError(f"{source}: Extrinsic-Omega/T must have 3 numbers")
    return out


@dataclass
class Rig:
    """One calibrated stereo rig (left = camera 0, right = camera 1)."""
    size: Tuple[int, int]            # (width, height)
    K1: np.ndarray
    K2: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    R: np.ndarray                    # right-camera rotation w.r.t. left
    T: np.ndarray                    # right-camera translation, mm (Tx < 0)
    name: str = ""

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    @classmethod
    def from_file(cls, path: str, dist: str = "opencv",
                  rot: str = "as_is") -> "Rig":
        """`dist`: 'opencv' = the 5 numbers are (k1,k2,p1,p2,k3); 'none' = zeros.
        `rot`: 'as_is' uses Rodrigues(Omega); 'inverse' uses its transpose.
        Both switches exist only so the data can decide (tools/check_stereo.py)."""
        c = parse_calibration(path)

        def K(f, cc, alpha):
            fx, fy = f[0], f[1] if len(f) > 1 else f[0]
            return np.array([[fx, alpha[0] * fx, cc[0]], [0.0, fy, cc[1]],
                             [0.0, 0.0, 1.0]], dtype=np.float64)

        def D(k):
            if dist == "none":
                return np.zeros(5)
            k = list(k) + [0.0] * (5 - len(k))
            return np.array(k[:5], dtype=np.float64)

        R, _ = cv2.Rodrigues(np.array(c["Extrinsic-Omega"], dtype=np.float64))
        if rot == "inverse":
            R = R.T
        return cls(size=(int(c["Width"][0]), int(c["Height"][0])),
                   K1=K(c["Camera-0-F"], c["Camera-0-C"], c["Camera-0-Alpha"]),
                   K2=K(c["Camera-1-F"], c["Camera-1-C"], c["Camera-1-Alpha"]),
                   D1=D(c["Camera-0-K"]), D2=D(c["Camera-1-K"]), R=R,
                   T=np.array(c["Extrinsic-T"], dtype=np.float64).reshape(3, 1),
                   name=os.path.basename(os.path.dirname(path)))


def calibration_path(frame_dir: str) -> str:
    """camera_calibration.txt sits next to left_frames/ in every sequence."""
    return os.path.join(os.path.dirname(os.path.abspath(frame_dir)),
                        "camera_calibration.txt")


def right_path(left_path: str) -> str:
    """The right frame has the same file name inside right_frames/."""
    d, f = os.path.split(left_path)
    parent, last = os.path.split(d)
    if last.lower() != "left_frames":
        raise ValueError(f"not inside a left_frames folder: {left_path}")
    return os.path.join(parent, "right_frames", f)


# --------------------------------------------------------------------------- #
# rectification
# --------------------------------------------------------------------------- #
class Rectifier:
    """cv2.stereoRectify + initUndistortRectifyMap for one rig, built once.

    alpha=1 keeps every source pixel (black borders appear, and are flagged
    invalid in `valid_left` / `valid_right`); alpha=0 crops to the valid area
    but zooms in. Disparity is d = f * B / Z with f = `focal`, B = `baseline_mm`.
    """

    def __init__(self, rig: Rig, alpha: float = 1.0):
        self.rig, self.alpha = rig, alpha
        w, h = rig.size
        (self.R1, self.R2, self.P1, self.P2, self.Q,
         self.roi1, self.roi2) = cv2.stereoRectify(
            rig.K1, rig.D1, rig.K2, rig.D2, (w, h), rig.R, rig.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
        self.mapL = cv2.initUndistortRectifyMap(rig.K1, rig.D1, self.R1, self.P1,
                                                (w, h), cv2.CV_32FC1)
        self.mapR = cv2.initUndistortRectifyMap(rig.K2, rig.D2, self.R2, self.P2,
                                                (w, h), cv2.CV_32FC1)
        ones = np.full((h, w), 255, np.uint8)
        self.valid_left = self._remap(ones, self.mapL, cv2.INTER_NEAREST, 0) > 0
        self.valid_right = self._remap(ones, self.mapR, cv2.INTER_NEAREST, 0) > 0

    @staticmethod
    def _remap(img, maps, interp, border):
        return cv2.remap(img, maps[0], maps[1], interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    @property
    def focal(self) -> float:
        return float(self.P1[0, 0])

    @property
    def baseline_mm(self) -> float:
        return float(-self.P2[0, 3] / self.P2[0, 0])

    def rectify(self, left: np.ndarray, right: np.ndarray):
        """Rectify a pair (bilinear: these are photographs)."""
        return (self._remap(left, self.mapL, cv2.INTER_LINEAR, 0),
                self._remap(right, self.mapR, cv2.INTER_LINEAR, 0))

    def rectify_labels(self, mask: np.ndarray, outside: int = 255) -> np.ndarray:
        """Warp a LEFT class-id mask into rectified space. Nearest neighbour only
        (bilinear would invent class ids); pixels with no source get `outside`."""
        return self._remap(mask, self.mapL, cv2.INTER_NEAREST, outside)

    def orig_grid_maps(self, out_hw: Tuple[int, int]):
        """For every pixel of an (H, W) grid that is a uniform downscale of the
        ORIGINAL left image, the matching coordinates in RECTIFIED-left space.
        Segmentation masks live on that grid, so rectified-space evidence is
        pulled back through these maps and lines up with them pixel for pixel.
        Pixel centres follow cv2.resize's convention."""
        key = tuple(out_hw)
        if getattr(self, "_og", None) is None:
            self._og = {}
        if key not in self._og:
            H, W = key
            w0, h0 = self.rig.size
            sx, sy = w0 / W, h0 / H
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            pts = np.stack([(u + 0.5) * sx - 0.5, (v + 0.5) * sy - 0.5], -1)
            pts = pts.reshape(-1, 1, 2).astype(np.float32)
            r = cv2.undistortPoints(pts, self.rig.K1, self.rig.D1, R=self.R1,
                                    P=self.P1).reshape(H, W, 2)
            self._og[key] = (np.ascontiguousarray(r[..., 0], np.float32),
                             np.ascontiguousarray(r[..., 1], np.float32))
        return self._og[key]

    def to_original_grid(self, rect_map: np.ndarray, out_hw: Tuple[int, int],
                         interp=cv2.INTER_NEAREST, border=0):
        """Resample a rectified-left-space map onto the original-left (H, W) grid.
        Nearest by default (masks, disparity: never blend across a depth edge)."""
        mx, my = self.orig_grid_maps(out_hw)
        return cv2.remap(rect_map, mx, my, interp, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=border)

    def depth_mm(self, disparity: np.ndarray) -> np.ndarray:
        """Z = f * B / d (rectified geometry)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.focal * self.baseline_mm / disparity


_rect_cache: Dict[Tuple[str, float], Rectifier] = {}


def get_rectifier(calib_path: str, alpha: float = 1.0, dist: str = "opencv",
                  rot: str = "as_is") -> Rectifier:
    key = (os.path.abspath(calib_path), alpha, dist, rot)
    if key not in _rect_cache:
        _rect_cache[key] = Rectifier(Rig.from_file(calib_path, dist, rot), alpha)
    return _rect_cache[key]


# --------------------------------------------------------------------------- #
# disparity
# --------------------------------------------------------------------------- #
@dataclass
class SGBMParams:
    """StereoSGBM settings. Stated choices (not learned): num_disp = 192 from the
    expected 25-160 px range (rounded up to a multiple of 16); P1/P2 use the
    standard 8/32 * block^2 smoothness scaling; min_disp = 0 because rectified
    disparity is positive."""
    num_disp: int = 192
    block: int = 5
    uniqueness: int = 10
    speckle_window: int = 0
    speckle_range: int = 2
    p1_mult: int = 8
    p2_mult: int = 32
    min_disp: int = 0


def make_matcher(p: SGBMParams):
    return cv2.StereoSGBM_create(
        minDisparity=p.min_disp, numDisparities=p.num_disp, blockSize=p.block,
        P1=p.p1_mult * p.block ** 2, P2=p.p2_mult * p.block ** 2,
        disp12MaxDiff=-1,                       # we do our own explicit LR check
        uniquenessRatio=p.uniqueness, speckleWindowSize=p.speckle_window,
        speckleRange=p.speckle_range, preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def disparity_pair(left_rect: np.ndarray, right_rect: np.ndarray,
                   p: SGBMParams = SGBMParams()):
    """Disparity of the LEFT view and of the RIGHT view (float32, NaN = invalid).

    The right-view disparity uses the mirror trick (match flipped right against
    flipped left), which needs no opencv-contrib. Convention: x_left = x_right + d.
    """
    m = make_matcher(p)
    Lg, Rg = to_gray(left_rect), to_gray(right_rect)
    dl = m.compute(Lg, Rg).astype(np.float32) / 16.0
    dr = m.compute(cv2.flip(Rg, 1), cv2.flip(Lg, 1)).astype(np.float32) / 16.0
    dr = cv2.flip(dr, 1)
    dl[dl < p.min_disp] = np.nan                  # SGBM marks invalid as min-1
    dr[dr < p.min_disp] = np.nan
    return dl, dr


def lr_consistent(dl: np.ndarray, dr: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """True where the left disparity is confirmed by the right view:
    |dR(x - dL) - dL| <= tol. Pixels with no disparity are False."""
    h, w = dl.shape
    ys, xs = np.mgrid[0:h, 0:w]
    valid = np.isfinite(dl)
    xr = np.rint(xs - np.where(valid, dl, 0)).astype(np.int64)
    inside = valid & (xr >= 0) & (xr < w)
    xr = np.clip(xr, 0, w - 1)
    back = dr[ys, xr]
    return inside & np.isfinite(back) & (np.abs(back - dl) <= tol)


def photometric_residual(left_rect: np.ndarray, right_rect: np.ndarray,
                         dl: np.ndarray) -> np.ndarray:
    """|L(x) - R(x - dL)| in gray levels (NaN where the disparity is invalid).
    Independent evidence that a disparity is right, used for parameter tuning
    where no ground-truth disparity exists."""
    Lg = to_gray(left_rect).astype(np.float32)
    Rg = to_gray(right_rect).astype(np.float32)
    h, w = dl.shape
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    ys = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    mapx = xs - np.where(np.isfinite(dl), dl, 0).astype(np.float32)
    warped = cv2.remap(Rg, mapx, ys, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    res = np.abs(Lg - warped)
    res[~np.isfinite(dl)] = np.nan
    return res


def occlusion_mask(dl: np.ndarray, dr: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """Left pixels the RIGHT camera cannot see.

    Left pixel p with disparity d lands on right pixel q = p - d. If q itself was
    assigned a LARGER disparity (a nearer surface occupies q in the right view),
    p is hidden behind that surface: dR(q) - d > tol. A smaller dR(q) is a
    false match, not an occlusion, so it is deliberately NOT flagged here
    (it shows up as an LR failure instead)."""
    h, w = dl.shape
    ys, xs = np.mgrid[0:h, 0:w]
    valid = np.isfinite(dl)
    xr = np.rint(xs - np.where(valid, dl, 0)).astype(np.int64)
    inside = valid & (xr >= 0) & (xr < w)
    back = dr[ys, np.clip(xr, 0, w - 1)]
    return inside & np.isfinite(back) & ((back - dl) > tol)


@dataclass
class StereoEvidence:
    """Per-pixel stereo evidence on the ORIGINAL-LEFT working grid (H, W), so it
    lines up pixel for pixel with the segmentation masks."""
    disparity: np.ndarray      # float16, NaN = no disparity; px in rectified geometry
    lr_ok: np.ndarray          # bool: passed the left-right consistency check
    occluded: np.ndarray       # bool: hidden from the right camera
    specular: np.ndarray       # bool: saturated, unsaturated-colour highlight
    confidence: np.ndarray     # uint8 0..255: LR-consistent AND photometrically matched
    in_view: np.ndarray        # bool: the pixel has a rectified counterpart
    image_cues: Optional[np.ndarray] = None   # float32[6] sitr.difficulty.IMAGE_CUE_NAMES


def compute_evidence(left_bgr: np.ndarray, right_bgr: np.ndarray, rect: "Rectifier",
                     corr: Optional["VerticalCorrection"], params: SGBMParams,
                     out_hw: Tuple[int, int], lr_tol: float = 1.0, occ_tol: float = 1.0,
                     residual_scale: float = 20.0, spec_v: int = 235,
                     spec_s: int = 40) -> StereoEvidence:
    """Everything the stereo indicators need for ONE frame, computed once.

    Stated thresholds (choices, not learned): LR tolerance 1 px; occlusion
    tolerance 1 px; confidence = 1 - residual/20 gray levels for LR-consistent
    pixels, 0 otherwise; specular = value > 235 and saturation < 40 (0-255).
    """
    Lr, Rr = rect.rectify(left_bgr, right_bgr)
    if corr is not None:
        Rr = corr.apply(Rr)
    dl, dr = disparity_pair(Lr, Rr, params)
    ok = lr_consistent(dl, dr, lr_tol)
    occ = occlusion_mask(dl, dr, occ_tol)
    res = photometric_residual(Lr, Rr, dl)
    conf = np.where(ok, np.clip(1.0 - np.nan_to_num(res, nan=residual_scale) / residual_scale, 0, 1), 0.0)
    H, W = out_hw
    to = rect.to_original_grid
    disp = to(dl.astype(np.float32), out_hw, cv2.INTER_NEAREST, border=float("nan"))
    small = cv2.resize(left_bgr, (W, H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    spec = (hsv[..., 2] > spec_v) & (hsv[..., 1] < spec_s)
    return StereoEvidence(
        disparity=disp.astype(np.float16),
        lr_ok=to(ok.astype(np.uint8), out_hw).astype(bool),
        occluded=to(occ.astype(np.uint8), out_hw).astype(bool),
        specular=spec,
        confidence=to((conf * 255).astype(np.uint8), out_hw),
        in_view=to(rect.valid_left.astype(np.uint8), out_hw).astype(bool),
        image_cues=image_cues(left_bgr))


def right_to_left_grid(right_map: np.ndarray, disparity: np.ndarray, rect: "Rectifier",
                       corr: Optional["VerticalCorrection"], out_hw: Tuple[int, int],
                       interp: int = cv2.INTER_NEAREST, outside=255) -> np.ndarray:
    """Pull a map defined on the ORIGINAL RIGHT working grid onto the ORIGINAL LEFT
    working grid, through the cached left disparity.

    For each left pixel: its rectified-left position (xl, yl); the matching
    rectified-right position (xl - d, yl) (the self-calibration offset is undone
    first); that position's coordinates in the ORIGINAL right image, read from the
    right camera's own rectification maps; and finally the right map sampled there.
    Pixels without a disparity or that land outside the right image get `outside`.
    Label maps use nearest neighbour (never blend class ids)."""
    H, W = out_hw
    xl, yl = rect.orig_grid_maps(out_hw)
    d = disparity.astype(np.float32)
    valid = np.isfinite(d)
    xr = (xl - np.where(valid, d, 0)).astype(np.float32)
    ysrc = yl - (corr.offset(xr, yl) if corr is not None else 0.0)
    ysrc = ysrc.astype(np.float32)
    big = -1e4
    ox = cv2.remap(rect.mapR[0], xr, ysrc, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=big)
    oy = cv2.remap(rect.mapR[1], xr, ysrc, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=big)
    w0, h0 = rect.rig.size
    ux = ((ox + 0.5) * (W / w0) - 0.5).astype(np.float32)
    uy = ((oy + 0.5) * (H / h0) - 0.5).astype(np.float32)
    out = cv2.remap(right_map, ux, uy, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=outside)
    out[~valid | (ox < -1e3)] = outside
    return out


def instrument_stereo_indicators(pred_mask: np.ndarray, lr_ok: np.ndarray,
                                 occluded: np.ndarray, confidence: np.ndarray,
                                 in_view: np.ndarray, instrument_ids) -> Tuple[float, float, float]:
    """The three frame-level stereo indicators of the reliability gate, over the
    PREDICTED-instrument pixels that lie inside the stereo field of view:

      stereo_disagreement  fraction failing the left-right consistency check
      occluded_fraction    fraction hidden from the right camera
      disparity_confidence mean per-pixel confidence (0..1)

    All inputs are on the same grid (the cache stores evidence at the mask grid's
    2x; resize before calling if they differ). Ground truth never enters here.
    Returns (0, 0, 0) when the predicted foreground is empty, matching the repo's
    eq. (10) convention that an empty foreground leaves every indicator at 0."""
    fg = np.isin(pred_mask, np.asarray(list(instrument_ids), dtype=pred_mask.dtype)) & in_view
    n = int(fg.sum())
    if n == 0:
        return 0.0, 0.0, 0.0
    return (float((~lr_ok[fg]).mean()), float(occluded[fg].mean()),
            float(confidence[fg].astype(np.float32).mean() / 255.0))


# --------------------------------------------------------------------------- #
# per-sequence self-calibration (label-free)
# --------------------------------------------------------------------------- #
# MEASURED in Task 2: the shipped per-RIG calibration is sub-pixel accurate on
# most sequences but leaves a stable vertical tilt of 3-14 px on others (the
# files list Omega_z ~ 0 while the cameras are rolled ~0.3 degrees relative to
# each other in those procedures). The residual is well described by
#       dy(x, y) = y_left - y_right = a + b * x/1000 + c * y/1000
# which this estimates from matched features of a few frames of the SEQUENCE
# ITSELF (images only, no labels) and removes by a vertical-only warp of the
# right rectified image.
def _tukey_fit(A: np.ndarray, y: np.ndarray, iters: int = 25, c: float = 4.685):
    """Robust least squares (Tukey biweight IRLS), starting from the median so a
    minority of false matches cannot drag the fit. Returns (beta, residual MAD)."""
    beta = np.zeros(A.shape[1]); beta[0] = float(np.median(y))
    for _ in range(iters):
        r = y - A @ beta
        s = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-6
        u = r / (c * s)
        w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if w.sum() < A.shape[1] + 3:
            break
        sw = np.sqrt(w)
        new = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        if np.allclose(new, beta, atol=1e-6):
            beta = new
            break
        beta = new
    r = y - A @ beta
    return beta, float(1.4826 * np.median(np.abs(r - np.median(r))))


@dataclass
class VerticalCorrection:
    """dy(x, y) = a + b*x/1000 + c*y/1000, in rectified RIGHT-image coordinates."""
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    n_matches: int = 0
    mad: float = float("nan")
    max_abs_px: float = 0.0        # largest |dy| the fit implies over the image
    applied: bool = False          # False -> identity (correction below noise)

    def offset(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        if not self.applied:
            return np.zeros_like(xs, dtype=np.float32)
        return (self.a + self.b * xs / 1000.0 + self.c * ys / 1000.0).astype(np.float32)

    def apply(self, right_rect: np.ndarray) -> np.ndarray:
        """Right image so that features sit on the same row as in the left image:
        R'(x, y) = R(x, y - dy(x, y)). Bilinear (photograph, not a label map)."""
        if not self.applied:
            return right_rect
        h, w = right_rect.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        return cv2.remap(right_rect, xs, ys - self.offset(xs, ys), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def to_dict(self) -> Dict:
        return dict(a=self.a, b=self.b, c=self.c, n_matches=self.n_matches,
                    mad=self.mad, max_abs_px=self.max_abs_px, applied=self.applied)


def estimate_vertical_correction(rect: "Rectifier", left_paths, min_matches: int = 150,
                                 apply_threshold_px: float = 1.0,
                                 max_dy: float = 30.0) -> VerticalCorrection:
    """Estimate the correction of ONE sequence from the frames in `left_paths`.

    `apply_threshold_px` (a stated choice): the correction is applied only if it
    implies more than this many pixels anywhere in the image, because below
    about 1 px it is SIFT localisation noise, not misalignment.
    """
    pls, prs = [], []
    for lp in left_paths:
        L, R = cv2.imread(lp, cv2.IMREAD_COLOR), cv2.imread(right_path(lp), cv2.IMREAD_COLOR)
        Lr, Rr = rect.rectify(L, R)
        pl, pr, _ = epipolar_matches(Lr, Rr, max_dy=max_dy)
        if len(pl):
            pls.append(pl); prs.append(pr)
    if not pls:
        return VerticalCorrection()
    pl, pr = np.concatenate(pls), np.concatenate(prs)
    if len(pl) < min_matches:
        return VerticalCorrection(n_matches=len(pl))
    dy = pl[:, 1] - pr[:, 1]                                   # signed
    A = np.c_[np.ones(len(pr)), pr[:, 0] / 1000.0, pr[:, 1] / 1000.0]
    beta, mad = _tukey_fit(A, dy)
    w, h = rect.rig.size
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
    mx = float(np.max(np.abs(beta[0] + beta[1] * corners[:, 0] / 1000.0
                             + beta[2] * corners[:, 1] / 1000.0)))
    return VerticalCorrection(a=float(beta[0]), b=float(beta[1]), c=float(beta[2]),
                              n_matches=int(len(pl)), mad=mad, max_abs_px=mx,
                              applied=mx > apply_threshold_px)


# --------------------------------------------------------------------------- #
# rectification quality
# --------------------------------------------------------------------------- #
def epipolar_matches(left_img: np.ndarray, right_img: np.ndarray,
                     max_dy: float = 20.0,
                     dx_range: Tuple[float, float] = (-5.0, 260.0),
                     ratio: float = 0.75, n_features: int = 4000):
    """Mutual-nearest-neighbour SIFT matches that pass Lowe's ratio test in both
    directions, a plausible horizontal shift (`dx_range`) and |dy| < max_dy.

    Returns (pts_left [N,2], pts_right [N,2], n_mutual_before_plausibility)."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    gl, gr = to_gray(left_img), to_gray(right_img)
    kl, dl_ = sift.detectAndCompute(gl, None)
    kr, dr_ = sift.detectAndCompute(gr, None)
    empty = np.zeros((0, 2))
    if dl_ is None or dr_ is None or len(kl) < 8 or len(kr) < 8:
        return empty, empty, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def good(a, b):
        out = {}
        for m, n in bf.knnMatch(a, b, k=2):
            if m.distance < ratio * n.distance:
                out[m.queryIdx] = m.trainIdx
        return out
    f, r = good(dl_, dr_), good(dr_, dl_)
    pairs = [(i, j) for i, j in f.items() if r.get(j) == i]
    if not pairs:
        return empty, empty, 0
    pl = np.array([kl[i].pt for i, _ in pairs])
    pr = np.array([kr[j].pt for _, j in pairs])
    dx, dy = pl[:, 0] - pr[:, 0], np.abs(pl[:, 1] - pr[:, 1])
    keep = (dx >= dx_range[0]) & (dx <= dx_range[1]) & (dy < max_dy)
    return pl[keep], pr[keep], len(pairs)


def epipolar_errors(left_img: np.ndarray, right_img: np.ndarray,
                    max_dy: float = 20.0, dx_range: Tuple[float, float] = (-5.0, 260.0),
                    ratio: float = 0.75, n_features: int = 4000) -> Dict[str, float]:
    """Vertical disparity |y_left - y_right| of matched SIFT features.

    Matches are kept only if they pass Lowe's ratio test in both directions
    (mutual), the horizontal shift is a plausible disparity (`dx_range`) and
    |dy| < max_dy. The count dropped by the last two filters is reported so the
    number cannot hide wild mismatches. Good rectification: mean about 1 px or less.
    """
    pl, pr, n_mut = epipolar_matches(left_img, right_img, max_dy, dx_range,
                                     ratio, n_features)
    if n_mut == 0:
        return dict(n_matches=0)
    dx, dy_k, dx_k = pl[:, 0] - pr[:, 0], np.abs(pl[:, 1] - pr[:, 1]), pl[:, 0] - pr[:, 0]
    if dy_k.size == 0:
        return dict(n_matches=0, n_mutual=n_mut)
    return dict(n_matches=int(dy_k.size), n_mutual=n_mut,
                frac_kept=float(dy_k.size / n_mut),
                dy_mean=float(dy_k.mean()), dy_median=float(np.median(dy_k)),
                dy_p95=float(np.percentile(dy_k, 95)),
                dx_p5=float(np.percentile(dx_k, 5)), dx_p50=float(np.median(dx_k)),
                dx_p95=float(np.percentile(dx_k, 95)))
