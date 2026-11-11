# Findings: stereo-gated selective correction on EndoVis 2018

Plain-language summary of what was measured. Every number comes from a file in `results/`. Nothing was tuned after seeing a test number. Official test = the 4 EndoVis 2018 test sequences (997 labelled frames), 12-class scene task, scored with the official challenge metric at 1280x1024.

## The short version

1. The base segmenter (M1) scores **0.6128 mIoU** on the official test set (our implementation reproduces the official `run.py` to 1e-16).
2. **No correction method changes that number in any meaningful way.** M1 to M6 all lie between 0.6128 and 0.6141. The largest gain (M5, +0.0012) is smaller than the gap between sequences by two orders of magnitude.
3. **The stereo premise is not supported.** Stereo evidence is viable geometrically (Task 2) but adds nothing to the reliability gate (Task 4) and its correction action changes the answer by +0.0001 to +0.0009 (Tasks 5, 6). The temporal indicators carry the gate.
4. The selective selector (M6) **does reduce harmful corrections per frame processed** versus blind fusion (0.80% vs 1.20% of all frames) but **it is worse per correction made** (3.1% vs 1.2% of corrected frames) and it **does not beat the simplest baseline** (always apply the morphological refinement), which has about twice its mean gain.
5. On the frames the gate flags (30.4% of test), always-applying the morphological refinement raises instrument IoU by +0.062 when invented classes count, and that gain is almost entirely the removal of predicted instrument classes absent from the ground truth, which the official metric ignores (Task B). Better temporal alignment did not help (Task A).
6. What we can defend is a cost result: M6 reaches the same mIoU as blind fusion at about a third of the latency (792 ms vs 2201 ms mean per frame, composed from measured components).

## Numbers (official test, 997 frames)

| Model | What it does | mIoU | Boundary F1 | Activated | Harmful, % of all frames | Mean latency |
|---|---|---|---|---|---|---|
| M1 | segmentation only | 0.6128 | 0.424 | 0% | 0.0 | 76 ms |
| M2 | + stereo evidence, every frame | 0.6137 | 0.428 | 99.9% | 0.9 | 2015 ms |
| M3 | + temporal only | 0.6129 | 0.424 | 20.5% | 0.4 | 258 ms |
| M4 | stereo-temporal fusion, every frame | 0.6138 | 0.429 | 99.9% | 1.2 | 2201 ms |
| M5 | M4 + class-conditioned refinement | 0.6141 | 0.427 | 100% | 1.7 | 2311 ms |
| M6 | selective action choice with abstention | 0.6128 | 0.423 | 25.9% | 0.8 | 792 ms |

"Harmful" means a correction lowered the frame's instrument IoU by more than 0.01. Per sequence, mIoU varies from 0.41 (seq 4) to 0.81 (seq 3); that spread dwarfs every difference above.

## What each task found

**Dataset (Task 1).** 17 places where the dataset differs from the brief are listed at the top of `DATASET_FACTS.md`. The two that matter most: the supplied calibration leaves a several-pixel vertical tilt on some training sequences, and the previous code's "challenge IoU" wrongly included background.

**Stereo (Task 2).** Rectification works. Mean vertical error is 2.64 px with calibration only and 1.21 px after a label-free per-sequence correction. The four test sequences were already within about 1 px; the correction was applied to one of them (seq 3, 1.3 px implied) with negligible effect (0.66 to 0.62 px). Stereo is therefore not blocked by geometry.

**Does correction help (Task 3).** Yes, by a very small amount. On test, always-correct moves instrument IoU from 0.6105 to 0.6140 (+0.0035, CI +0.0002 to +0.0081) and the official-metric IoU from 0.6397 to 0.6397 (+0.0001, CI -0.0002 to +0.0002). Out-of-fold on training sequences the picture is the same (+0.0030 instrument, +0.0003 official). Almost all routed frames are unchanged.

**Evidence and gate (Task 4).** The difficulty index did not validate (Spearman about 0 out-of-fold and +0.08 on test; the within-sequence interval includes 0). Stereo indicators did not improve the reliability gate: out-of-fold AUC difference +0.005 with an interval that spans 0, and +0.004 on test. Temporal indicators did carry weight (largest leave-one-out AUC drop +0.038). The brief expected +0.05 to +0.15 from stereo; we measured none.

**Actions and selector (Task 5).** The oracle has real headroom over always-temporal (+0.0215 vs +0.0031 on test, both averaged over all 997 frames), so selection was worth trying. The learned selector captures about half of that on test (+0.0096 over all frames) but always applying the morphological refinement (A3) gets +0.0189 over all frames (+0.0621 over the 303 flagged frames). An earlier version of this paragraph called the +0.0189 a routed-frame figure; it is an all-frame figure. Selector predictions are weak for A1 (Pearson -0.03) and A3 (+0.14), moderate for A2 (+0.33) and fusion (+0.46). These changes are in instrument IoU where a class counts if it is in the ground truth or the prediction; see Task B below for what that means.

**Task A: the aligner bug (results/taskA/SUMMARY.md).** The action data behind Tasks 5 and 6 was built with the temporal action pasting the previous mask on unmoved (identity alignment) because the tool hardcoded it. It was regenerated with Farneback optical flow and with the region aligner, and the whole comparison was repeated. Neither helps. On flagged frames out of fold the temporal action gains +0.0047 (identity), +0.0030 (Farneback), +0.0007 (region); the share of flagged frames it leaves unchanged stays at 96.0%, 96.0%, 96.7% (97.0%, 97.4%, 98.0% on test; Task 3 had 95.7%). Region is lower than identity with an interval that excludes zero; Farneback's interval includes zero. The oracle ceiling on flagged test frames does not rise (+0.0707, +0.0706, +0.0641). So the null result for the temporal action was not caused by the bug. Official mIoU for M1 to M6 changes by at most 0.0002 under either aligner, so the published table stands. Identity stays the system of record because it is the best of the three out of fold. Why alignment does not help is not established here; it is consistent with the low overlap between consecutive ground-truth instrument masks recorded in docs/ (0.30 to 0.45 IoU), which Task K writes up.

**Task B: the population the method is designed for, and what the +0.062 is (results/taskB/).** Flagged frames are 30.4% of the test set (303 of 997). On them, in instrument IoU (class counted if in the ground truth or the prediction): oracle +0.0707, always-A3 +0.0621, selector +0.0315, always-A2 +0.0102, always-fusion +0.0072, always-A1 +0.0004. Over all frames these become +0.0215, +0.0189, +0.0096, +0.0031, +0.0022, +0.0001. Multiplying the flagged-frame figure by the flagged share (0.0621 x 0.304) gives the all-frame figure (0.0189): that step is dilution. The next step is not. The official 12-class frame IoU changes by -0.00008 over all frames under always-A3. The reason is the class rule, not class averaging: on the 303 flagged test frames always-A3 changes instrument IoU by +0.0621 when a class counts if it is in the ground truth or the prediction, by +0.0001 when it counts only if it is in the ground truth (the official rule), and the official frame IoU by -0.0003. A3 removed 74 predicted instrument classes that are not in the ground truth and added none (191 flagged frames had at least one such class before, 168 after). So the +0.062 is almost entirely the removal of invented instrument classes, which the official metric does not penalise and therefore cannot reward. It is a real effect (the model does invent instrument classes), but it is not a gain in the official score, and earlier wording that attributed the small global number to dilution alone was incomplete. The same decomposition holds for A2 (union +0.0102, ground-truth-only +0.0003) and fusion (+0.0072, +0.0008). Two more readings of the table: always-A3 reaches 88% of the oracle on flagged test frames (76% out of fold), so the oracle's headroom over the best fixed policy is only +0.0086 (test) and +0.0062 (out of fold); and the learned selector is below always-A3 on both sets, so the oracle gap does not show that a better chooser is what is missing, only that choosing per frame could in principle beat any fixed action by that margin. Not measured here: the average number of classes per frame.

**Final comparison (Task 6).** See the table. The headline test "M6 beats M4 on harmful-correction rate" is true when counted over all frames (0.80 vs 1.20) and false when counted over corrected frames (3.1 vs 1.2). Both are reported; neither shows a gain in mIoU.

## Claims

Supported by our measurements:
- The evaluation pipeline reproduces the official metric exactly.
- Calibrated stereo rectification is usable on this data, and a label-free vertical correction fixes the sequences where it is not.
- A reliability gate built from prediction-confidence and temporal indicators detects bad frames (test AUROC about 0.84).
- Selective correction can match blind fusion's mIoU at roughly a third of the compute.

Not supported:
- That stereo evidence improves failure detection or correction (measured: no).
- That any correction raises mIoU in a way that matters (measured: at most +0.0012).
- That the difficulty index predicts failures.
- That the learned selector beats a fixed morphological refinement.
- Novelty claim 6 from the brief (stereo-temporal advantage) and the premise that stereo beats temporal cues.

## Defensible and indefensible sentences

Defensible: "On the four official test sequences, the selective policy achieves the same mIoU as always-on fusion (0.613) while activating on 26% of frames and costing about 0.8 s instead of 2.2 s per frame."
Defensible: "In our experiments stereo disparity evidence did not improve reliability estimation beyond temporal indicators."
Defensible: "On the frames the gate flags, always applying the morphological refinement raises instrument IoU (counting invented classes) by 0.062, almost entirely by removing predicted instrument classes that are not in the ground truth; the official score does not change because it does not count those classes."
Defensible: "Motion-compensated temporal alignment (Farneback, region) did not improve the temporal action over pasting the previous mask unmoved."
Indefensible: "Correction improves segmentation by 6 points on difficult frames."
Indefensible: "Stereo gating improves surgical segmentation."
Indefensible: "Our method outperforms published results." No published number was compared; protocols differ.

## Limitations

- Four test sequences: sequence-level intervals are coarse; read them as warnings, not tests.
- Stage 2 conformal risk on test is 0.077 against a 0.05 budget; the guarantee is not met on this test set.
- The temporal memory inside the action data holds uncorrected base predictions (policy-independent). Chained corrections are not simulated.
- Suturing-needle IoU is 0 and suction-instrument does not appear in the test set; 11 of 12 classes are scored.
- The latency figures are composed from separately measured components on an RTX 3050 Ti laptop GPU (4 GB), not timed end to end.
- Single training run; no seed variation (the gate learner is deterministic, the segmenter was trained once per fold).
- Resolution was limited by 4 GB of VRAM (training at 512x640, evaluation at full resolution).

## Where to look

`results/task6/final_results.json` (all numbers above), `results/task5/selector_results.json`, `results/task4/`, `results/task3/`, `results/task2/STEREO_REPORT.md`, `RUNLOG.md` (every error and decision), `DATASET_FACTS.md`.
