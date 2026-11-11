# Metric definitions (Task P.4)

Every metric used in this study: its formula, what it rewards, its range, what it is blind to, whether higher or lower is better, the file that computes it, and the interval method.
Unless stated, intervals are 95% percentile intervals from a bootstrap that resamples whole SEQUENCES (2,000 draws, seed 0): 15 clusters out of fold, 4 on the official test set (so test
intervals are coarse and are read as a warning, not a test). Frame-level intervals are never used.

Notation. A frame has a ground-truth label map Y and a prediction S over classes 0..11 (0 = background-tissue). TP_c, FP_c, FN_c are pixel counts for class c; G_c = TP_c + FN_c is the
ground-truth area of c.

## 1. Official mIoU (the challenge metric)
- **Formula.** For each frame, the mean over classes c in 1..11 with G_c > 0 of IoU_c = TP_c / (TP_c + FP_c + FN_c); frames with no scored class are skipped; the frame values are then
  averaged (test_data/utils.py: per-sequence frame mean, sequences weighted by their scored frames, which equals the pooled frame mean). Evaluated at 1280x1024.
- **Rewards.** Overlap on every class that is actually in the frame. **Range.** 0 to 1; higher is better.
- **Blind to.** A class the model predicts that is NOT in the ground truth: it adds nothing to the frame mean (utils.py:143), so invented classes are free. Also blind to how rare classes
  are treated, because each frame's mean is over only 4-8 classes.
- **Computed by.** `sitr/metrics.py` (`official_class_ious`, `official_frame_iou`), `tools/final_eval.py`; cross-checked against the official run.py to 1e-16 (results/task6).

## 2. mcIoU
- **Formula.** For each class, the mean of IoU_c over the frames in which the class is in the ground truth; then the mean over classes. On the test set 10 of the 11 foreground classes are
  scored (suction-instrument has no test pixels; "11" would count background).
- **Rewards.** Getting every class right, including rare ones, because a class counts equally however few frames contain it. **Range.** 0 to 1; higher is better.
- **Blind to.** Invented classes (same rule as mIoU). Dominated by classes present in a single test sequence (thread, clamps, needle), so its variance is large.
- **Computed by.** `tools/mciou_hallucination.py`, `tools/tta_ensemble_eval.py`, `tools/per_class_report.py`, `tools/metric_ablation.py`.

## 3. Boundary F1
- **Formula.** On the union of instrument classes: the boundary of a mask is its morphological gradient (3x3); boundary precision is the share of predicted boundary pixels within 3 px of a
  ground-truth boundary pixel, recall the converse; F1 is their harmonic mean (0 if both are 0); the frame values are averaged over frames where both boundaries exist. The 3 px tolerance
  is a stated choice. **Rewards.** Boundary placement. **Range.** 0 to 1; higher is better.
- **Blind to.** Class identity (it is on the instrument union) and to interior errors.
- **Computed by.** `tools/final_eval.py` (`boundary_f1`).

## 4. clDice (centreline Dice)
- **Formula.** For a thin class: with skeletons S_p and S_g of the prediction and ground truth, Tprec = |S_p ∩ G| / |S_p|, Tsens = |S_g ∩ P| / |S_g|; clDice = 2 Tprec Tsens / (Tprec + Tsens).
  Reported for shaft, thread and needle, averaged over frames where the class is in the ground truth (0 when the prediction is empty).
- **Rewards.** Connectivity and completeness of thin structures. **Range.** 0 to 1; higher is better. **Blind to.** Thickness errors.
- **Computed by.** `tools/final_eval.py` (`cl_dice`, skimage skeletonize at full resolution).

## 5. AURC (area under the risk-coverage curve), the recommended headline for failure detection
- **Formula.** Risk of a frame = 1 - official frame IoU. Order frames from most to least reliable by the detector's score; at coverage k/n the risk is the mean risk of the k accepted frames;
  AURC is the mean of that selective risk over all coverages. The floor is the oracle ranking, the ceiling the random ranking (the overall mean risk).
- **Rewards.** A detector whose most trusted frames really are the best ones, at every operating point. **Range.** between the floor and the random value; lower is better.
- **Blind to.** Calibration (only the ranking matters) and to which frames are chosen to be flagged at a fixed threshold.
- **Computed by.** `sitr/metrics.py` (`aurc`, `risk_coverage`), `tools/failure_detect_table.py`. Sequence-clustered bootstrap as above.

## 6. AUROC (detector ranking quality; secondary, for comparability)
- **Formula.** The probability that a randomly chosen bad frame (official IoU < 0.5, a standing threshold, not tuned) receives a lower reliability score than a randomly chosen good frame; ties
  count one half. **Rewards.** Ranking bad frames below good ones. **Range.** 0.5 (chance) to 1; higher is better.
- **Blind to.** How bad a bad frame is, and to the 0.5 threshold choice; the bad-frame rate is 13.9% out of fold and 29.5% on test, so AUROC values are not comparable across sets.
  A second column uses the instrument-IoU < 0.5 label, which is the gate's own target.
- **Computed by.** `sitr/metrics.py` (`roc_auc`), `tools/failure_detect_table.py`.

## 7. Harmful-correction rate (both denominators)
- **Formula.** A correction is harmful when the instrument IoU of the corrected frame is lower than that of the uncorrected frame by more than 0.01 (a stated choice). Reported as the share of
  corrected frames and as the share of ALL frames (the two orderings of methods can differ: a method that corrects few frames can be worse per correction and better overall).
- **Rewards (when low).** Not damaging what it touches. **Range.** 0 to 100%; lower is better. **Blind to.** Harm too small to cross the margin, and harm on the official metric, which can be
  zero while the instrument IoU moves (invented classes).
- **Computed by.** `tools/final_eval.py`, `tools/train_selector.py`, `tools/policy_table.py`.

## 8. Hallucination rate
- **Formula.** A hallucination event is a (frame, class) pair with the class absent from the ground truth and predicted with at least 0.05% of the frame (655 px at 1280x1024; a stated choice;
  0 and 0.5% stored as a check). Reported: share of frames with at least one event, events per frame, share of predicted (frame, class) pairs that are invented, and per class the share of
  the frames where the class is absent in which it is predicted anyway.
- **Rewards (when low).** Not inventing classes. **Range.** 0 to 1 (or percent); lower is better. **Blind to.** Whether an invented region is large; the floor decides what counts.
  It is the failure the official metric cannot see.
- **Computed by.** `tools/mciou_hallucination.py`.

## 9. Activation rate
- **Formula.** Share of frames on which a correction was applied (the action changed the mask); and the routing rate, the share of frames the gate flags (reliability below tau = 0.64).
- **Range.** 0 to 100%. **Meaning.** Cost driver, not quality: lower activation is cheaper. **Blind to.** Whether the corrections helped.
- **Computed by.** `tools/final_eval.py`, `tools/train_selector.py`, `tools/policy_table.py`.

## 10. Latency
- **Formula.** Wall-clock milliseconds per frame, mean and 95th percentile, with the GPU synchronised before and after, batch 1, image already on the GPU (decode excluded). Hardware: NVIDIA RTX 3050 Ti
  laptop GPU (4 GB) with a 20-thread laptop CPU. Failure-detector latencies are MEASURED end to end per method (`results/taskC/latency.json`); correction-policy latencies are COMPOSED from measured
  components (segmentation pass, stereo matching, right-view pass, per-action time), exactly as `tools/final_eval.py` composes them, and are marked as composed.
- **Range.** Positive; lower is better. **Blind to.** Throughput gains from batching, and to other hardware.

## 11. Oracle gain
- **Formula.** The gain a perfect chooser would reach: on each flagged frame the best of {keep, available actions} by measured change in instrument IoU (uses ground truth), averaged over the
  flagged frames, or over all frames. Reported with both denominators (per corrected frame and per flagged frame); they differ by the share of flagged frames that the oracle leaves alone.
- **Meaning.** Headroom, not a result: no deployable method can reach it. **Range.** >= 0; higher means more headroom. **Blind to.** Everything the oracle cannot know at inference time.
- **Computed by.** `tools/train_selector.py`, `tools/policy_table.py`.

## Related definitions used in the text
- **Instrument IoU** (the gate's target and the action data's label): per class over the 8 instrument classes, a class counted if it is in the ground truth OR the prediction, an empty pair
  scoring 1.0 (`sitr/metrics.py: instrument_iou`). Because it counts the prediction, removing an invented class raises it; the official metric does not.
- **Region IoU** (Task K): IoU of the union of instrument classes, class identity ignored.
- **Pixel-level AUROC** (Task H): the probability that a wrong predicted-instrument pixel gets a higher suspicion score than a correct one; 0.5 is chance.
