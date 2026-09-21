"""Selective temporal recovery for surgical instrument segmentation — Stages 1-5.

Stage 1  base segmentation model f_seg          (report Sec. IV-B / V-C)
Stage 2  reliability assessment module          (report Sec. IV-C / V-D)
Stage 3  temporal alignment (region-level)      (report Sec. IV-E, eq. 16/24)
Stage 4  temporal recovery module               (report Sec. IV-D, eqs. 17-23)
Stage 5  identity verification                  (report Sec. IV-F, eqs. 24-25)
"""

__version__ = "2.0.0"

from . import (align, calibration, dataset, identity, labels, losses,  # noqa: F401
               metrics, models, recovery, reliability, runtime,
               stage34_interface, utils)

