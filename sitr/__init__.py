"""Selective temporal recovery for surgical instrument segmentation — Stages 1-2.

Stage 1  base segmentation model f_seg          (report Sec. IV-B / V-C)
Stage 2  reliability assessment module          (report Sec. IV-C / V-D)

Stages 3-7 (temporal memory, recovery, identity association, event construction,
event-based evaluation) are deliberately NOT implemented here.
"""

__version__ = "1.0.0"

from . import (align, calibration, dataset, labels, losses, metrics,  # noqa: F401
               models, reliability, runtime, stage34_interface, utils)
