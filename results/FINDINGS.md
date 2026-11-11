# Findings: stereo-gated selective correction on EndoVis 2018

Plain-language summary of what was measured. Every number comes from a file in `results/`. Nothing was tuned after seeing a test number. Official test = the 4 EndoVis 2018 test sequences (997 labelled frames), 12-class scene task, scored with the official challenge metric at 1280x1024.

## The short version

1. The base segmenter (M1) scores **0.6128 mIoU** on the official test set (our implementation reproduces the official `run.py` to 1e-16).
2. **No correction method changes that number in any meaningful way.** M1 to M6 all lie between 0.6128 and 0.6141. The largest gain (M5, +0.0012) is smaller than the gap between sequences by two orders of magnitude.
3. **The stereo premise is not supported.** Stereo evidence is viable geometrically (Task 2) but adds nothing to the reliability gate (Task 4) and its correction action changes the answer by +0.0001 to +0.0009 (Tasks 5, 6). The temporal indicators carry the gate.
4. The selective selector (M6) **does reduce harmful corrections per frame processed** versus blind fusion (0.80% vs 1.20% of all frames) but **it is worse per correction made** (3.1% vs 1.2% of corrected frames) and it **does not beat the simplest baseline** (always apply the morphological refinement), which has about twice its mean gain.
5. On the frames the gate flags (30.4% of test), always-applying the morphological refinement raises instrument IoU by +0.062 when invented classes count, and that gain is almost entirely the removal of predicted instrument classes absent from the ground truth, which the official metric ignores (Task B). Better temporal alignment did not help (Task A).
6. What we can defend is a cost result. The earlier form, "M6 reaches the same mIoU as blind fusion at about a third of the latency (792 ms against 2201 ms)", compared against a baseline that contains the stereo action A1. A1 was retired (Task F: it never helped); without it the flagged frames no longer need stereo matching, and the selector costs 153 ms per frame (composed from measured components), 2.35 times faster than applying the temporal and refinement corrections to every frame (360 ms), with the same official mIoU and mcIoU as before.
7. The simple failure detectors are as good as the gate on the official risk. Under risk = 1 - official frame IoU, one-pass max-softmax and mean entropy match or beat the production gate (out-of-fold AURC 0.269 and 0.270 against 0.303; test 0.285 against 0.308), and the 4-model ensemble is no better (test 0.312). The gate is the best or near-best detector only on the instrument-IoU label it was trained for (AUROC 0.788 out of fold, 0.844 on test), which is where the earlier "0.84" comes from (Task C).
8. The cross-view indicators did not pass their pre-set test (Task D), TTA and the ensemble do not raise mIoU (Task E), per-pixel occlusion does not localise errors (Task H), and a ground-truth ceiling shows that identity relabelling could add +5.4 mcIoU points while the shipped Stage 5 never relabels (Task Q).

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

## Measurements added in the final stretch (Tasks C to L and Q)

**Failure-detection baselines (Task C; results/taskC).** Same 3,232 frames, risk = 1 - official frame IoU, bad frame = official IoU < 0.5 (13.9% of out-of-fold frames, 29.5% of test). Out of fold, max-softmax scores AURC 0.269 and AUROC 0.786, mean entropy 0.270 and 0.773, Soft Dice Confidence 0.285 and 0.734, MC stochastic depth (10 passes) 0.288 and 0.741, TTA entropy (6 passes) 0.271 and 0.764, the production gate 0.303 and 0.740 (floor 0.230, random 0.347). On test: max-softmax 0.285 and 0.792, TTA entropy 0.273 and 0.850, the 4-model ensemble 0.312 and 0.840 (test only), the gate 0.308 and 0.771. All sequence-clustered intervals overlap, so no detector is shown to beat another; nothing more expensive than one-pass max-softmax is shown to be better. Measured per-frame cost: 80 ms for max-softmax, 535 ms for TTA, 782 ms for 10 MC passes, 312 ms for the ensemble, 109 ms for the gate, 288 ms for the gate with cross-view indicators. The gate regresses instrument IoU while this risk is the 12-class official IoU, which includes the anatomy classes; on the instrument-IoU label the gate leads (0.788 out of fold, 0.844 test, against 0.704 and 0.710 for max-softmax). Cross-view (two passes) lands near the ensemble on test AURC (0.305 against 0.312) at half the passes, but is behind one-pass max-softmax and failed its own test.

**Cross-view indicators (Task D; results/taskD).** Six indicators that compare the left and right predicted masks with no disparity. Pre-set rule: out of fold, paper4 + crossview6 must beat paper4 by at least +0.02 AUROC with a sequence-clustered interval excluding zero. Result: 0.7937 against 0.7449, a gain of +0.049 with interval -0.040 to +0.160: **not passed**, stopped, no variant searched. On test (once): 0.8988 against 0.8464. AURC, a metric the rule did not name, improves with an interval excluding zero (-0.027, -0.051 to -0.008, out of fold); that is recorded as an observation and does not change the verdict. The disparity control (+3 disparity indicators) moved +0.030 out of fold and -0.016 on test.

**TTA and the ensemble (Task E; results/taskE).** Single model mIoU 0.6128, mcIoU 0.4956; TTA (6 passes) 0.6139 and 0.4971; the 4-model ensemble (a different system, four trained models, test only) 0.6105 and 0.4977. Differences in mIoU points with sequence-clustered intervals: TTA +0.10 (-0.20 to +0.28), ensemble -0.23 (-0.91 to +0.45). Needle stays at IoU 0.000 in all three. A negative result.

**mcIoU, hallucination, A1 retired, A4 (Task F; results/taskF).** mcIoU (10 of 11 foreground classes; suction has no test pixels) is 0.4954 to 0.4962 for M1 to M6, so no correction moves it. On the official test set 58.4% of frames contain a class the model invents (absent from the ground truth, area at least 0.05% of the frame), 15.2% of all predicted (frame, class) pairs are invented, and the official metric cannot see any of it. Action A1 is retired together with the fusion that contains it; the selector re-run over {A2, A3} gains +0.0176 on test (always-A3 +0.0189) with the same official mIoU and mcIoU as M6. Action A4 (drop a class predicted in only one view and unsupported by memory) fires on 2.9% of test frames, is right 84% of the time it fires, lowers the share of frames with an invented class from 58.4% to 57.1% and leaves mIoU unchanged (-0.01 points).

**Stereo against temporal information (Task G; results/taskG).** The right view has no labels, so the stereo axis is bounded by a round trip of the ground-truth left mask. On the 56% of ground-truth instrument pixels where the round trip is defined, using two independently computed disparities, 0.941 of the mask (instrument IoU) survives, against 0.426 for the best temporal aligner (Farneback) on the same pixels; the same-map round trip reads 0.991, so reusing one disparity map inflates the figure by about 5 points. The comparison is an upper bound and not like for like (same instant versus a different instant; mask carried through geometry versus predicted). It does not contradict the measured failure of A1, which pulls the right view's predicted labels.

**Per-pixel occlusion (Task H; results/taskH).** Out of fold, pixel-level AUROC for predicting where an instrument prediction is wrong: occluded 0.501 (0.497 to 0.504), left-right check failure 0.522, low disparity confidence 0.529, against 0.741 for plain proximity to a mask boundary. At chance after removing the boundary confound. A negative result.

**Metric definitions (Task I; results/taskI).** On the same predictions the official mIoU 0.6128 becomes 0.6452 (+3.23 points) if background is counted, 0.5021 (-11.07) if a class also counts when only the prediction has it, 0.4956 (-11.72) as a per-class mean (mcIoU), and changes by under 0.05 points with unweighted sequence averaging or evaluation at 512x640. Two definition choices each outweigh every correction effect measured here.

**Per class, per sequence, rigs (Task J; results/taskJ).** Test mIoU / mcIoU by sequence: 0.668 / 0.535, 0.564 / 0.512, 0.806 / 0.776, 0.413 / 0.433. On test sequence 4 kidney-parenchyma collapses to 0.153 against 0.800 on the other three; ultrasound-probe, shaft and wrist also drop. Sequence 4 shares a camera rig with training sequences 13, 15, 16, so an unseen rig is not the explanation. Seven calibration files describe five rigs; rig A holds both suturing-needle sequences, so leave-one-rig-out is infeasible (it would leave zero needle frames), and the official test set shares rigs with training and does not measure generalisation to an unseen rig.

**Consecutive frames (Task K; results/CONSECUTIVE_FRAME_IOU.md).** Ground-truth instrument masks of consecutive frames overlap at a mean IoU of 0.323 (0.286 to 0.361), median 0.272, with 76% of pairs below 0.5; random pairs 0.056. A 1 Hz sample thinned of near-duplicates on purpose (arXiv:2001.11190).

**Label noise (Task L; results/taskL).** The 20 highest out-of-fold-loss training frames (14 from sequence 16, 6 from sequence 5) all look like model error on tissue classes with plausible ground truth; none looks like a label error (exact 95% upper bound for such frames 13.9%).

**Stage 5 identity verification (Task Q; results/taskQ).** The shipped Stage 5 returns the input mask by design and never counts identity switches, so M7 equals M6 by construction. A ground-truth ceiling for identity relabelling on the M6 masks is +2.07 mIoU points and +5.37 mcIoU points (needle 0.000 to 0.134); 20.3% of ground-truth wrist pixels are predicted as clasper. The literature's relabelling gain exists on this data as a ceiling; the shipped stage captures none of it.

**Not done, stated.** Task M (rare-class training: clDice loss and copy-paste augmentation of the 12 real annotated needle instances) was dropped for the deadline; suturing-needle stays at IoU 0.000 and clDice 0.000, and that work is stated future work. Extra databases (Task P.3) were not approved, so every result here is from one dataset. Task N (release packaging) was not done.

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
