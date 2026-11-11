# Task D: cross-view indicators without disparity

**Result: the pre-set test was not passed. Stopped, as the rule requires; no variant was searched.**

## What was tested

Six indicators (`sitr/crossview.py`) that compare the left and right PREDICTED label maps directly, with no disparity and no
photometric matching: Jaccard of the predicted class sets, largest per-class area ratio, difference in connected-component counts,
per-row class-profile L1, row-span IoU and the correlation of per-row run counts. The three row-based ones run on the rectified
grid, with nearest-neighbour warps, the stored vertical correction, and a field-of-view guard (186 px at full resolution, scaled to
the 320-wide grid, dropped from the left edge of the left view and the right edge of the right view). All 3,232 frames were used
(2,235 training frames out of fold, 997 test frames). 16 unit tests pass (`tests/test_crossview.py`).

## The rule, fixed in `tools/compare_crossview.py` before any number was seen

Out of fold, `paper4 + crossview6` beats `paper4` by at least +0.02 AUROC with a sequence-clustered 95% interval excluding zero
(2,000 draws that resample whole sequences; 15 training sequences). Fixed configuration: gradient-boosted gate, memory K=5, folds from
`work_v2/folds.json`, bad frame = instrument IoU < 0.5 (the gate's standing label). The control `paper4` reproduces the committed
out-of-fold AUROC 0.7449 and test AUROC 0.8464 exactly.

## Out of fold

| gate indicators | AUROC | AURC |
|---|---|---|
| paper4 (4 confidence/temporal) | 0.7449 | 0.2877 |
| paper4 + 3 disparity indicators (known-negative control) | 0.7747 | 0.2797 |
| paper4 + 6 cross-view | 0.7937 | 0.2601 |
| 6 cross-view alone | 0.7133 | 0.2852 |

**Decision:** `paper4 + crossview6` minus `paper4` = +0.0488 AUROC, sequence-clustered 95% interval -0.0396 to +0.1601. The point
estimate clears +0.02 but the interval includes zero, so the rule is **not** met. FAIL.

The paired differences (AUROC, 95% interval): cross-view over paper4 +0.049 (-0.040 to +0.160); disparity control over paper4 +0.030
(-0.015 to +0.093); cross-view alone against paper4 -0.032 (-0.150 to +0.121).

## Which indicators carried weight (leave one indicator out of the 10-indicator set; AUROC drop when removed)

`q_confidence` +0.050 (by far the largest). Among the cross-view six: `xv_area_ratio_max` +0.006, `xv_ncomp_diff` +0.004, `xv_runcount_corr`
+0.003; removing `xv_class_jaccard` (-0.003), `xv_rowprofile_l1` (-0.003) or `xv_rowspan_iou` (-0.021) would have RAISED the AUROC.
Nothing was dropped on that basis: dropping indicators until a bar is cleared is exactly what the rule forbids, and all of these differences
are within the noise of 15 sequences.

## Test set, scored once after the decision

| gate indicators | AUROC | AURC |
|---|---|---|
| paper4 | 0.8464 | 0.3264 |
| paper4 + 3 disparity indicators | 0.8301 | 0.3366 |
| paper4 + 6 cross-view | 0.8988 | 0.3056 |
| 6 cross-view alone | 0.8287 | 0.3107 |

Cross-view over paper4: +0.052 AUROC, interval +0.026 to +0.217; the interval comes from 4 sequences and is coarse. This does not change the
out-of-fold decision.

## What can and cannot be said

- Said: the pre-set test failed. On 15 training sequences the AUROC gain is not distinguishable from zero.
- Observed, outside the rule, not a decision: the direction is positive on both the out-of-fold and the test data, and for AURC (the
  headline for failure detection) the out-of-fold difference is -0.027 with an interval that excludes zero (-0.051 to -0.008), and on test -0.022
  (-0.042 to -0.009). AURC was not the rule's metric; reading it as a pass would be choosing the metric after seeing the result. At most this
  justifies a follow-up on more sequences, which this dataset cannot supply (15 training sequences).
- The disparity-based control behaved inconsistently (+0.030 out of fold, -0.016 on test), which is what noise looks like.
- The earlier negative stands. It is consistent with stereo matching being least reliable on specular instruments (left-right consistency 45.6% on
  instrument pixels against 65.2% on tissue, a measured association, not a tested cause). The second camera as such was not found useless.
- Detection and propagation are separate claims. Task G measured propagation separately.
