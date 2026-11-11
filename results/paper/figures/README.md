# Paper figures

Every figure is drawn by `tools/paper_figures.py` (figures 3, 4 and 6a by `tools/paper_figures_p1.py`) from the files listed under it; none contains typed numbers. Each exists as PDF (vector, for LaTeX) and PNG at 300 dpi, with the caption in the `.txt` file of the same name. Palette: Okabe and Ito (colour-blind safe), with a distinct line style or marker per series. Sized for a single IEEE column (3.5 in) unless the figure needs the double width.

## fig01_risk_coverage_oof

**Risk-coverage curves, oof.** Risk-coverage curves of every failure detector on the same 2235 out-of-fold training frames; risk is 1 minus the official frame IoU and a frame is bad when its official IoU is below 0.5 (bad rate 13.9%). AURC in the legend. No out-of-fold ensemble exists. The MC row is MC stochastic depth (the segmenter has no dropout layer). Source: results/taskC/failure_detection.json and per-frame scores in work_v2/failure_detection.

Source files: `results/taskC/failure_detection.json`, `work_v2/failure_detection/scores`, `results/taskD/gate_scores_*.npz`

## fig01_risk_coverage_test

**Risk-coverage curves, test.** Risk-coverage curves of every failure detector on the same 997 official test frames; risk is 1 minus the official frame IoU and a frame is bad when its official IoU is below 0.5 (bad rate 29.5%). AURC in the legend. The 4-model ensemble is test-only: every training sequence lies inside some member's training set. The MC row is MC stochastic depth (the segmenter has no dropout layer). Source: results/taskC/failure_detection.json and per-frame scores in work_v2/failure_detection.

Source files: `results/taskC/failure_detection.json`, `work_v2/failure_detection/scores`, `results/taskD/gate_scores_*.npz`

## fig02_roc_oof

**ROC curves, oof.** ROC curves of every failure detector on the same 2235 out-of-fold training frames; risk is 1 minus the official frame IoU and a frame is bad when its official IoU is below 0.5 (bad rate 13.9%). AUROC in the legend. No out-of-fold ensemble exists. The MC row is MC stochastic depth (the segmenter has no dropout layer). Source: results/taskC/failure_detection.json and per-frame scores in work_v2/failure_detection.

Source files: `results/taskC/failure_detection.json`, `work_v2/failure_detection/scores`, `results/taskD/gate_scores_*.npz`

## fig02_roc_test

**ROC curves, test.** ROC curves of every failure detector on the same 997 official test frames; risk is 1 minus the official frame IoU and a frame is bad when its official IoU is below 0.5 (bad rate 29.5%). AUROC in the legend. The 4-model ensemble is test-only: every training sequence lies inside some member's training set. The MC row is MC stochastic depth (the segmenter has no dropout layer). Source: results/taskC/failure_detection.json and per-frame scores in work_v2/failure_detection.

Source files: `results/taskC/failure_detection.json`, `work_v2/failure_detection/scores`, `results/taskD/gate_scores_*.npz`

## fig05_consecutive_frame_iou

**Consecutive-frame ground-truth IoU.** Overlap of the ground-truth instrument masks of consecutive frames in the 15 training sequences (2220 pairs, 1 Hz sampling). Left: distribution, pooled mean 0.323, median 0.272, 76% of pairs below 0.5, against random pairs from the same sequence. Right: per sequence (box = quartiles, whiskers = 1.5 IQR). Source: results/taskK/consecutive_frame_iou.json.

Source files: `results/taskK/consecutive_frame_iou.json`, `results/taskK/pair_scores.npz`

## fig07_action_change_distributions

**Per-action IoU change distributions with the harmful range.** Distribution of the per-frame change in instrument IoU produced by each correction action, only for frames the action actually changed (the unchanged share is printed in each panel). The shaded region is the harmful range, a drop larger than 0.01 (the standing harm margin). Top: out-of-fold training frames; bottom: official test frames. Values beyond +-0.4 are clipped into the end bins. The instrument IoU counts a class if it is in the ground truth or the prediction (see Task I). Source: work_v2/actions/*.npz (keys d_A1 .. d_FUSE).

Source files: `work_v2/actions/*.npz`

## fig08_accuracy_vs_cost

**Accuracy against cost per policy.** Accuracy against cost for every correction policy on the official test set: mean change in instrument IoU on the frames the gate flags (y) against mean latency per frame (x, log scale). Latency is composed from components measured on this machine (RTX 3050 Ti laptop GPU, CPU for the correction steps): the selector and the every-frame fusion rows are the measured M6, M4, M5 latencies of Task 6; routed policies add the measured per-action time on the flagged share of frames; the oracle runs every candidate. Instrument IoU counts a class if it is in the ground truth or the prediction, so most of the A3 gain is removal of invented classes (Task B). Source: results/taskB/policy_table.json, results/task6/final_results.json, work_v2/actions.

Source files: `results/taskB/policy_table.json`, `results/task6/final_results.json`, `work_v2/actions/test__*.npz`

## fig09_indicator_importance

**Leave-one-out indicator importance.** Leave-one-indicator-out importance, out of fold (15 training sequences, gradient-boosted gate, bad frame = instrument IoU below 0.5). Left: the 12-indicator gate, the three disparity indicators in vermillion. Right: the 4 original indicators plus the 6 cross-view indicators (green). Negative bars mean removing the indicator RAISED the AUROC. The cross-view set did not pass its pre-set test (Task D); differences of a few thousandths are within the noise of 15 sequences. Source: results/task4/gate_comparison_gbm.json and results/taskD/decision.json.

Source files: `results/task4/gate_comparison_gbm.json`, `results/taskD/decision.json`

## fig10_aligner_comparison

**Aligner comparison for the temporal action.** The temporal action (A2) built three ways, mean change in instrument IoU on the frames the gate flags. Whiskers on the two motion-compensated bars are the 95% sequence-clustered interval of their difference from the original (previous mask pasted on unmoved), drawn at the difference added to the original's value; both intervals are of a PAIRED difference. Neither motion-compensated version beat the original; region matching was lower with an interval excluding zero out of fold. Source: results/taskA/aligner_comparison.json.

Source files: `results/taskA/aligner_comparison.json`

## fig11_memory_starvation

**Memory starvation per sequence.** Memory starvation: for every sequence, the share of frames whose gate reliability clears the memory-admission threshold tau_memory = 0.5 (the value the gate was trained and the temporal memory admits with). Frames that do not clear it cannot enter the reliable memory the temporal action draws on. Blue: training sequences (out of fold); vermillion: official test sequences. Source: gate reliability per frame in work_v2/actions/*.npz (key r), threshold from work_v2/release/manifest.json (tau_train).

Source files: `work_v2/actions/*.npz`, `work_v2/release/manifest.json`

## fig12_metric_ablation

**Metric-implementation ablation.** What each definition choice is worth in mIoU points, on the same predictions (the single model, 997 test frames). Reference: the exact official protocol (class 0 never scored, class counted only if in the ground truth, per-frame mean, full 1280x1024). Each bar changes exactly one choice. Two choices (counting a class on the prediction as well, and averaging per class instead of per frame) each move the score by more than 11 points, more than any correction effect in this study. Source: results/taskI/metric_ablation.json.

Source files: `results/taskI/metric_ablation.json`

## Not produced

- **Figure 13, qualitative frames.** Publishing EndoVis frames is governed by the dataset's own licence and data-use terms, which are not the owner's alone to approve (see NEEDS OWNER in RUNLOG.md); it is not generated on assumption.
- Any figure of the list in task P.2 that does not appear above had its inputs missing at the time of the last run; `_manifest.json` lists the reason.
