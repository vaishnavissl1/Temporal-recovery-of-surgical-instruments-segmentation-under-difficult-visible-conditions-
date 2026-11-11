# Consecutive-frame ground-truth IoU

How much does the ground-truth instrument mask change from one frame to the next in EndoVis 2018? This bounds what any method that carries a mask forward in time (copy the previous mask, warp it, vote over a memory) can achieve at this frame rate: it is the overlap of the two TRUE masks, so a perfect propagation method could not exceed it by copying. Produced by `tools/consecutive_frame_iou.py`; every number below is read from `results/taskK/consecutive_frame_iou.json`.

## Why the frames are so different: how the dataset was built

The challenge paper (Allan et al., *2018 Robotic Scene Segmentation Challenge*, arXiv:2001.11190) describes the sampling. Two sentences, checked against the paper's HTML rendering: sections were "extracted and subsampled to 1 Hz", and "Similar frames were manually removed until the sequence contained 300 frames." The dataset is therefore a 1 Hz sample that was then thinned of near-duplicates on purpose, which removes exactly the frame pairs that would have overlapped most.

What the paper does and does not say about participants: in the rendering checked, it does not state that any team used information from other time steps, and one method (Method 4, NTNU) takes a left/right frame pair as input, so stereo input was used by at least one team. The absence of a statement is not proof that no team used temporal information; a claim that none did should be checked against the full method descriptions before it is cited.

## Definitions

- **Instrument IoU**: per-class IoU over the 8 instrument classes, class counted if in either mask, empty pair = 1.0 (the definition the gate's target uses, and DATASET_FACTS.md section 6).
- **Region IoU**: IoU of the union of the instrument classes.
- Resolution 1280x1024 (label PNGs); pairs are (t-1, t) of one sequence, used only when the two frame numbers differ by exactly 1 (no pair had to be skipped in any sequence).
- Random pairs: for every frame, one partner drawn uniformly from the same sequence at least 20 frames away (seed 0), for scale.
- Intervals: 95% percentile interval of the mean, bootstrap over sequences (2000 draws, seed 0). "<0.5" is the share of pairs with IoU below the repo's standing 0.5 (not tuned).

## Training sequences (15): pooled

| pairs | n | mean (95%) | median | 5th pct | 25th pct | 75th pct | 95th pct | < 0.5 |
|---|---|---|---|---|---|---|---|---|
| consecutive, instrument IoU | 2220 | 0.323 (0.286 to 0.361) | 0.272 | 0.003 | 0.108 | 0.488 | 0.800 | 76% |
| consecutive, region IoU | 2220 | 0.420 (0.388 to 0.452) | 0.408 | 0.021 | 0.209 | 0.618 | 0.862 | 61% |
| random (20+ apart), instrument IoU | 2235 | 0.056 (0.041 to 0.076) | 0.023 | 0.000 | 0.003 | 0.075 | 0.232 | 100% |
| random (20+ apart), region IoU | 2235 | 0.137 (0.119 to 0.159) | 0.108 | 0.000 | 0.036 | 0.201 | 0.380 | 98% |

Histogram of consecutive instrument IoU (10 equal bins from 0 to 1, counts): 528, 356, 309, 271, 221, 168, 132, 124, 53, 58.

### Per sequence (train), consecutive pairs

| sequence | instrument IoU mean | median | 25th | 75th | < 0.5 | region IoU mean |
|---|---|---|---|---|---|---|
| train/seq_1 | 0.286 | 0.260 | 0.117 | 0.430 | 87% | 0.384 |
| train/seq_2 | 0.293 | 0.248 | 0.109 | 0.379 | 84% | 0.447 |
| train/seq_3 | 0.217 | 0.155 | 0.037 | 0.346 | 90% | 0.310 |
| train/seq_4 | 0.281 | 0.265 | 0.123 | 0.389 | 87% | 0.467 |
| train/seq_5 | 0.355 | 0.345 | 0.169 | 0.514 | 72% | 0.432 |
| train/seq_6 | 0.211 | 0.145 | 0.035 | 0.317 | 87% | 0.310 |
| train/seq_7 | 0.316 | 0.232 | 0.118 | 0.454 | 77% | 0.429 |
| train/seq_9 | 0.382 | 0.257 | 0.093 | 0.693 | 68% | 0.458 |
| train/seq_10 | 0.376 | 0.309 | 0.108 | 0.586 | 67% | 0.476 |
| train/seq_11 | 0.345 | 0.293 | 0.064 | 0.566 | 71% | 0.397 |
| train/seq_12 | 0.401 | 0.382 | 0.169 | 0.627 | 59% | 0.486 |
| train/seq_13 | 0.521 | 0.567 | 0.311 | 0.718 | 43% | 0.556 |
| train/seq_14 | 0.273 | 0.234 | 0.100 | 0.407 | 84% | 0.393 |
| train/seq_15 | 0.257 | 0.231 | 0.083 | 0.403 | 85% | 0.382 |
| train/seq_16 | 0.326 | 0.288 | 0.155 | 0.479 | 76% | 0.372 |

## Official test sequences (4): pooled

| pairs | n | mean (95%) | median | 5th pct | 25th pct | 75th pct | 95th pct | < 0.5 |
|---|---|---|---|---|---|---|---|---|
| consecutive, instrument IoU | 993 | 0.400 (0.283 to 0.518) | 0.310 | 0.000 | 0.107 | 0.624 | 1.000 | 66% |
| consecutive, region IoU | 993 | 0.474 (0.369 to 0.580) | 0.431 | 0.010 | 0.180 | 0.737 | 1.000 | 56% |
| random (20+ apart), instrument IoU | 997 | 0.086 (0.034 to 0.167) | 0.012 | 0.000 | 0.000 | 0.064 | 0.493 | 95% |
| random (20+ apart), region IoU | 997 | 0.148 (0.113 to 0.201) | 0.067 | 0.000 | 0.005 | 0.191 | 0.671 | 95% |

Histogram of consecutive instrument IoU (10 equal bins from 0 to 1, counts): 239, 129, 114, 98, 78, 71, 55, 27, 19, 163.

### Per sequence (test), consecutive pairs

| sequence | instrument IoU mean | median | 25th | 75th | < 0.5 | region IoU mean |
|---|---|---|---|---|---|---|
| test/seq_1 | 0.317 | 0.290 | 0.127 | 0.488 | 77% | 0.381 |
| test/seq_2 | 0.249 | 0.200 | 0.070 | 0.361 | 86% | 0.356 |
| test/seq_3 | 0.460 | 0.454 | 0.172 | 0.702 | 56% | 0.536 |
| test/seq_4 | 0.576 | 0.729 | 0.098 | 1.000 | 46% | 0.624 |

## Reading it

- The distribution is wide and has a heavy low end, so the mean alone hides it: use the percentiles and the histogram above. A large share of consecutive pairs overlap by less than half, and a visible share overlap by almost nothing.
- Consecutive masks overlap far more than random pairs from the same sequence, so there is temporal information; it is just not enough for copying a mask forward to be reliable at this rate. This is consistent with the measured temporal-recovery result (`results/taskA/SUMMARY.md`): the temporal action leaves most flagged frames unchanged.
- Sequences differ a lot. The ranking of sequences here and the ranking of segmentation difficulty are different questions; this file measures only how fast the scene moves between samples.
- The test sequences are scored on the same definitions and are shown for completeness; the measurement described in the task is the training-sequence block.
- Ground truth is used here, and only here, to measure the dataset. Nothing in this file feeds any gate input, action input or model.
