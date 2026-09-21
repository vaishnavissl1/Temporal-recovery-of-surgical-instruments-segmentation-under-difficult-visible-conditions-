"""
Temporal alignment operators W_{s->t} (report Sec. IV-E, eq. 16).

The report is explicit that at the EndoVis release rate (1 Hz, arXiv:2001.11190) dense optical
flow is NOT a safe default: consecutive released frames are ~0.5-1 s apart and
instrument displacement routinely exceeds the range over which flow estimates
stay accurate.  Identity alignment is therefore the default operator here, and
Farneback flow is provided as the ablation the report describes — not as the
primary mechanism.

Both operators warp a *label map* and so always use nearest-neighbour
interpolation: interpolating class indices is meaningless.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class IdentityAlign:
    """W_{s->t} = identity.  Honest default for 1 Hz released frames."""

    name = "identity"

    def __call__(self, mask_s: np.ndarray, gray_s: Optional[np.ndarray] = None,
                 gray_t: Optional[np.ndarray] = None) -> np.ndarray:
        return mask_s


class FarnebackAlign:
    """Dense optical flow alignment (ablation, Sec. V-J.4).

    Flow is computed from frame t back to frame s, so that every pixel of the
    target frame reads its source location in frame s directly -- this is the
    correct direction for backward warping and avoids forward-scatter holes.
    """

    name = "farneback"

    def __init__(self, pyr_scale: float = 0.5, levels: int = 4, winsize: int = 21,
                 iterations: int = 3, poly_n: int = 7, poly_sigma: float = 1.5):
        self.kw = dict(pyr_scale=pyr_scale, levels=levels, winsize=winsize,
                       iterations=iterations, poly_n=poly_n,
                       poly_sigma=poly_sigma, flags=0)
        self._grid_cache: Optional[tuple] = None

    def _grid(self, h: int, w: int):
        if self._grid_cache is None or self._grid_cache[0] != (h, w):
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            self._grid_cache = ((h, w), xs, ys)
        return self._grid_cache[1], self._grid_cache[2]

    def __call__(self, mask_s: np.ndarray, gray_s: Optional[np.ndarray] = None,
                 gray_t: Optional[np.ndarray] = None) -> np.ndarray:
        if gray_s is None or gray_t is None:
            return mask_s
        if gray_s.shape != gray_t.shape:
            gray_s = cv2.resize(gray_s, (gray_t.shape[1], gray_t.shape[0]),
                                interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(gray_t, gray_s, None, **self.kw)

        h, w = mask_s.shape[:2]
        if (h, w) != flow.shape[:2]:
            sy, sx = h / flow.shape[0], w / flow.shape[1]
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= sx
            flow[..., 1] *= sy

        xs, ys = self._grid(h, w)
        map_x = xs + flow[..., 0]
        map_y = ys + flow[..., 1]
        return cv2.remap(mask_s, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def build_align(name: str):
    name = (name or "identity").lower()
    if name == "identity":
        return IdentityAlign()
    if name in ("flow", "farneback", "optical_flow"):
        return FarnebackAlign()
    raise ValueError(f"unknown alignment operator {name!r} (identity|farneback)")
