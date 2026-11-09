# Task 2 — calibrated stereo go / no-go

Generated from `stereo_check.json` by `tools/report_task2.py`. Tool: `tools/check_stereo.py`.

## Verdict

**Calibrated stereo is VIABLE.** None of the three stop conditions tripped, with calibration alone (not tripped) or with the self-calibration (not tripped).

**But the shipped calibration alone is not enough for every sequence.** It leaves a stable vertical tilt of several pixels on some TRAINING sequences (4 of 20 evaluation pairs have mean vertical error above 3 px). A label-free per-sequence correction (estimated from each sequence's first 40 s, evaluated on other frames) removes most of it. The four TEST sequences are fine with calibration alone.

## 1. Vertical disparity after rectification (matched SIFT features, px)

Unrectified pairs, mean abs(dy) averaged over the 20 pairs: **2.93 px**.

| Geometry | mean abs(dy) | median | mean p95 | worst pair (mean) | pairs with mean > 3 px |
|---|---|---|---|---|---|
| Calibration only | 2.64 | 0.68 | 5.04 | 12.62 | 4 / 20 |
| Calibration + self-calibration | 1.21 | 0.57 | 3.43 | 5.16 | 1 / 20 |

Distortion / rotation convention, decided by the data (calibration only, mean abs(dy)):

| Convention | mean abs(dy) |
|---|---|
| opencv/as_is  **(chosen)** | 2.640 |
| opencv/inverse | 2.782 |
| none/as_is | 2.648 |
| none/inverse | 2.747 |

Per pair:

| Pair | calibration only: mean / median / p95 | + self-calibration: mean / median / p95 |
|---|---|---|
| train/seq_1 frame074.png | 0.77 / 0.57 / 2.06 | 0.77 / 0.57 / 2.06 |
| train/seq_2 frame074.png | 6.10 / 5.61 / 10.58 | 1.31 / 0.74 / 4.15 |
| train/seq_3 frame074.png | 2.29 / 1.67 / 6.10 | 2.29 / 1.67 / 6.10 |
| train/seq_4 frame074.png | 6.31 / 6.12 / 9.80 | 1.12 / 0.56 / 4.73 |
| train/seq_5 frame074.png | 12.62 / 12.38 / 18.33 | 5.16 / 4.36 / 10.17 |
| train/seq_6 frame074.png | 0.67 / 0.47 / 1.72 | 0.67 / 0.47 / 1.72 |
| train/seq_7 frame074.png | 6.97 / 6.87 / 11.48 | 1.18 / 0.70 / 4.36 |
| train/seq_9 frame074.png | 0.87 / 0.59 / 2.28 | 0.87 / 0.59 / 2.28 |
| train/seq_10 frame074.png | 0.71 / 0.40 / 2.48 | 0.71 / 0.40 / 2.48 |
| train/seq_11 frame074.png | 1.60 / 1.44 / 3.07 | 1.85 / 1.70 / 3.27 |
| train/seq_12 frame074.png | 1.03 / 0.78 / 2.55 | 0.89 / 0.65 / 2.64 |
| train/seq_13 frame074.png | 2.94 / 2.86 / 4.07 | 0.79 / 0.57 / 2.36 |
| train/seq_14 frame074.png | 0.69 / 0.44 / 2.07 | 0.69 / 0.44 / 2.07 |
| train/seq_15 frame074.png | 2.77 / 2.41 / 6.77 | 1.32 / 0.68 / 4.58 |
| train/seq_16 frame074.png | 2.48 / 2.45 / 3.66 | 0.65 / 0.35 / 2.12 |
| test/seq_1 frame125.png | 0.66 / 0.46 / 1.76 | 0.66 / 0.46 / 1.76 |
| test/seq_2 frame124.png | 0.93 / 0.57 / 3.37 | 0.93 / 0.57 / 3.37 |
| test/seq_3 frame124.png | 0.66 / 0.50 / 1.87 | 0.62 / 0.48 / 1.72 |
| test/seq_4 frame124.png | 0.93 / 0.50 / 3.35 | 0.93 / 0.50 / 3.35 |
| train/seq_1 frame037.png | 0.81 / 0.38 / 3.38 | 0.81 / 0.38 / 3.38 |

## 2. Per-sequence vertical self-calibration (images only, no labels)

Model: `dy(x, y) = y_left - y_right = a + b*x/1000 + c*y/1000`, estimated by robust (Tukey) regression on matched features of frames [0, 5, 10, 15, 20, 25, 30, 35]. Applied only if it implies more than 1 px somewhere in the image.

| Sequence | matches | a | b | c | residual MAD | max implied abs(dy), px | applied |
|---|---|---|---|---|---|---|---|
| train/seq_1 | 2848 | -0.37 | -0.01 | 0.15 | 0.64 | 0.39 | no |
| train/seq_2 | 3447 | -9.30 | 4.37 | 0.82 | 0.91 | 9.30 | yes |
| train/seq_3 | 2483 | -0.49 | 0.21 | 0.30 | 0.74 | 0.49 | no |
| train/seq_4 | 1701 | -10.50 | 6.08 | 1.24 | 0.80 | 10.50 | yes |
| train/seq_5 | 1994 | 0.32 | 13.91 | 3.63 | 1.51 | 21.85 | yes |
| train/seq_6 | 1826 | -0.40 | 0.46 | 0.02 | 0.71 | 0.40 | no |
| train/seq_7 | 3380 | -9.62 | 5.10 | 0.44 | 1.23 | 9.62 | yes |
| train/seq_9 | 1102 | -0.71 | 0.01 | 0.52 | 0.59 | 0.71 | no |
| train/seq_10 | 1278 | -0.90 | 0.23 | 0.53 | 0.57 | 0.90 | no |
| train/seq_11 | 3313 | -0.47 | -0.68 | 0.61 | 0.61 | 1.34 | yes |
| train/seq_12 | 1080 | -0.28 | -0.87 | 0.80 | 0.62 | 1.40 | yes |
| train/seq_13 | 1934 | -2.12 | -0.00 | -0.79 | 0.61 | 2.92 | yes |
| train/seq_14 | 2335 | 0.23 | -0.39 | 0.21 | 0.71 | 0.45 | no |
| train/seq_15 | 1046 | -3.19 | 0.61 | 0.47 | 0.88 | 3.19 | yes |
| train/seq_16 | 1801 | -2.36 | 0.22 | -0.44 | 0.56 | 2.80 | yes |
| test/seq_1 | 2690 | -0.40 | -0.30 | 0.62 | 0.51 | 0.79 | no |
| test/seq_2 | 3065 | -0.45 | 0.35 | 0.38 | 0.60 | 0.45 | no |
| test/seq_3 | 4170 | 0.30 | 0.80 | -0.94 | 0.55 | 1.33 | yes |
| test/seq_4 | 947 | 0.48 | -0.16 | -0.65 | 1.06 | 0.48 | no |

## 3. SGBM settings

Chosen (by the criterion fixed before any result: fraction of pixels that are LR-consistent AND photometric residual < 10 gray levels, 5 TRAINING tuning frames not among the 20): **block 15, uniqueness 0, speckle 0, numDisparities 192**, P1/P2 = 8/32 x block^2, min disparity 0, 3-way mode, mirror trick for the right-view disparity.

**Caveat:** the optimum is on the edge of the searched grid for block, uniqueness, i.e. the criterion keeps preferring larger windows and fewer uniqueness checks. It cannot see over-smoothing or false matches in textureless regions. A diagnostic declared in advance (`sgbm_diagnostic.txt`) checked this and did not overturn the choice.

Top 8 grid cells:

| block | uniqueness | speckle | verified | LR pass | valid | s/frame |
|---|---|---|---|---|---|---|
| 15 | 0 | 0 | 0.645 | 0.686 | 0.844 | 0.66 |
| 15 | 3 | 0 | 0.642 | 0.682 | 0.837 | 0.68 |
| 11 | 0 | 0 | 0.641 | 0.678 | 0.841 | 0.74 |
| 15 | 0 | 150 | 0.641 | 0.681 | 0.826 | 0.67 |
| 15 | 3 | 150 | 0.638 | 0.678 | 0.819 | 0.71 |
| 11 | 3 | 0 | 0.638 | 0.675 | 0.833 | 0.67 |
| 15 | 5 | 0 | 0.638 | 0.678 | 0.828 | 0.69 |
| 9 | 0 | 0 | 0.637 | 0.672 | 0.839 | 0.65 |

## 4. Disparity validity on the 20 evaluation pairs

Percent of the in-view left image. 'Interior' excludes the left strip where no right-image counterpart can exist.

| Geometry | valid disparity | LR-consistent | LR (interior) | instrument px: valid / LR | tissue px: valid / LR |
|---|---|---|---|---|---|
| Calibration only | 82.9 | 64.1 | 70.0 | 73.8 / 45.6 | 83.7 / 65.2 |
| Calibration + self-calibration | 85.0 | 69.3 | 75.6 | 77.9 / 54.3 | 86.0 / 71.1 |

Instrument pixels: 3,250,237; tissue pixels: 14,628,044 (training frames only; ground truth is used solely to split the pixels).

LR-consistent disparity percentiles (px, rectified geometry): p1 27.6, p5 35.9, median 85.0, p95 139.4, p99 186.5. Fraction at or above 190 px (the numDisparities cap): 0.85 %. The expected 25-160 px range holds for 1st to ~95th percentile; about 1 % of consistent pixels sit at the cap, so close-range structures are slightly truncated.

Per pair (refined):

| Pair | valid % | LR % | instrument LR % | tissue LR % |
|---|---|---|---|---|
| train/seq_1 frame074.png | 88.0 | 76.4 | 64.3 | 79.8 |
| train/seq_2 frame074.png | 80.7 | 63.8 | 61.1 | 64.8 |
| train/seq_3 frame074.png | 85.0 | 67.4 | 33.4 | 69.0 |
| train/seq_4 frame074.png | 77.5 | 55.3 | 44.5 | 60.9 |
| train/seq_5 frame074.png | 75.1 | 41.5 | 36.8 | 43.5 |
| train/seq_6 frame074.png | 86.0 | 73.7 | 54.4 | 81.1 |
| train/seq_7 frame074.png | 86.9 | 72.1 | 61.2 | 76.1 |
| train/seq_9 frame074.png | 90.0 | 80.9 | n/a | 80.9 |
| train/seq_10 frame074.png | 87.4 | 74.8 | 66.6 | 76.7 |
| train/seq_11 frame074.png | 76.2 | 49.3 | 41.3 | 51.6 |
| train/seq_12 frame074.png | 79.6 | 55.9 | 60.2 | 54.9 |
| train/seq_13 frame074.png | 84.5 | 69.3 | 67.8 | 69.5 |
| train/seq_14 frame074.png | 87.3 | 75.8 | 63.6 | 77.9 |
| train/seq_15 frame074.png | 89.6 | 75.4 | 15.1 | 79.1 |
| train/seq_16 frame074.png | 87.9 | 75.1 | 52.5 | 78.5 |
| test/seq_1 frame125.png | 86.8 | 73.6 | n/a | n/a |
| test/seq_2 frame124.png | 84.0 | 70.4 | n/a | n/a |
| test/seq_3 frame124.png | 89.6 | 79.1 | n/a | n/a |
| test/seq_4 frame124.png | 87.6 | 74.6 | n/a | n/a |
| train/seq_1 frame037.png | 90.4 | 81.8 | 74.5 | 82.8 |

## 5. Stop conditions

* [calibration_only] mean vertical disparity after rectification > 3 px: **not tripped**
* [calibration_only] fewer than 40% of pixels survive the LR check: **not tripped**
* [calibration_only] instrument pixels almost no valid disparity (<10% LR-consistent): **not tripped**
* [refined] mean vertical disparity after rectification > 3 px: **not tripped**
* [refined] fewer than 40% of pixels survive the LR check: **not tripped**
* [refined] instrument pixels almost no valid disparity (<10% LR-consistent): **not tripped**

Total run time 270 s. Images (rectified pairs with epipolar lines, disparity maps) are in this folder (git-ignored).
