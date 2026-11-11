# Table notes (captions and footnotes for the `.tex` fragments)

Write these as the caption and footnote text; the fragments are bare `tabular` environments, to be wrapped by the paper's own `table` float.

**failure_detection.tex.** Failure detection on the official test set (4 sequences) and out of fold (15 training sequences). Risk is 1 minus the official frame IoU; a frame is bad when
its official IoU is below 0.5. AURC lower is better, AUROC higher is better. ms is measured per frame on an RTX 3050 Ti laptop GPU at batch 1. Intervals (sequence-clustered) are in
`results/taskC/failure_detection_cost.md`; they are wide and overlap, so no row is shown to beat another. The MC row is MC stochastic depth, because the segmenter has no dropout layer.
$^{\dagger}$ Test only: every training sequence lies inside some member's training set, so no out-of-fold ensemble exists. $^{\ddagger}$ Adds a second forward pass (the right image) and
did not pass its pre-set out-of-fold test (results/taskD/SUMMARY.md). The gate regresses instrument IoU while the risk here is the 12-class official IoU.

**policies.tex.** Correction policies on the 997 official test frames, with the gate flagging 303 (30.4%). $\Delta$ flagged: mean change in instrument IoU on the flagged frames;
$\Delta$ all: over all frames; $\Delta$ official: change in the official frame IoU over all frames; Harm: share of corrected frames whose instrument IoU fell by more than 0.01. Instrument IoU
counts a class if it is in the ground truth or the prediction, so most of the A3 gain is removal of invented classes, which the official score cannot see (the official change of A3 is
-0.0001). $^{\dagger}$ Uses ground truth: a ceiling, not a method. These are the Task 5/6 policies, including A1 and the fusion, which were later retired (results/taskF).

**per_class.tex.** Official per-class IoU on the test set (class counted only if in the ground truth; the mean is over the frames where the class is present). Frames: number of frames
in which the class is present. suction-instrument has no test pixels. Thread, clamps and suturing-needle come from a single test sequence each, so their variance is large by construction.
mcIoU averages the 10 scored classes equally. Baseline columns appear when `results/taskP/baselines.json` exists.

**metric_ablation.tex.** The same single-model predictions scored under different definitions, one change from the official protocol per row (the first row is the reference). Points are
mIoU points relative to the reference. Not for comparison with published numbers, whose protocols differ.

**baseline_comparison.tex.** Models trained under an identical protocol (see results/paper/P1_PROTOCOL.md). Parameters in millions; GFLOPs counted as 2 x multiply-adds with the torch
flop counter at 512x640; ms measured at batch 1 on the same GPU. Published numbers are NOT in this table; they belong in a separate block with a protocol column.
