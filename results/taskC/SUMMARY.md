# Task C: failure-detection baseline table

Full table with intervals, forward passes, segmenters trained and measured ms/frame: `failure_detection_cost.md` (and `.json`). Per-frame scores: `per_frame_scores.csv`
(test) and `work_v2/failure_detection/scores`. Code: `tools/failure_detect_collect.py` (GPU scoring), `tools/failure_detect_table.py` (metrics),
`tools/failure_detect_latency.py` (timing), `tools/failure_detect_merge.py` (merge).

## Definitions, fixed before any score was seen

- Same frames, same risk for every detector: risk = 1 - official frame IoU of the single-pass mask at 1280x1024 (a class counts only if it is in the ground truth). A frame is bad when
  its official IoU is below 0.5 (the repo's standing threshold, not tuned): 13.9% of the 2,235 out-of-fold frames, 29.5% of the 997 test frames.
- "Predicted foreground" is every predicted non-background pixel, the set the official metric scores. Softmax at temperature 1. An empty foreground scores 0 (least reliable).
- The segmenter has no dropout layer, so the "MC" row is MC stochastic depth (its DropPath, rate 0.1, switched on at inference). TTA is flip x scales 0.75, 1.0, 1.25.
- Out of fold, each training sequence is scored only by the cross-fitting model that excluded it. The 4-model ensemble (three fold models plus the final model) is valid on the test
  sequences only, because every training sequence lies inside some member's training set; no out-of-fold ensemble was built.
- Intervals resample whole sequences (2,000 draws; 15 clusters out of fold, 4 on test).

## Result (AURC lower is better; floor = oracle ranking, ceiling = random ranking)

| Detector | passes | ms/frame | OOF AURC | OOF AUROC | test AURC | test AUROC |
|---|---|---|---|---|---|---|
| max-softmax | 1 | 80 | 0.2692 | 0.786 | 0.2852 | 0.792 |
| mean entropy | 1 | 80 | 0.2704 | 0.773 | 0.2853 | 0.789 |
| Soft Dice Confidence | 1 | 85 | 0.2849 | 0.734 | 0.3407 | 0.685 |
| MC stochastic depth | 10 | 782 | 0.2881 | 0.741 | 0.3255 | 0.792 |
| TTA entropy | 6 | 535 | 0.2712 | 0.764 | 0.2727 | 0.850 |
| TTA pairwise Dice | 6 | 557 | 0.2823 | 0.757 | 0.3203 | 0.808 |
| 4-model ensemble pairwise Dice (test only) | 4 | 312 | n/a | n/a | 0.3120 | 0.840 |
| gate, production (MLP, 9 indicators) | 1 | 109 | 0.3034 | 0.740 | 0.3076 | 0.771 |
| gate + 6 cross-view indicators (GBM) | 2 | 288 | 0.2980 | 0.729 | 0.3050 | 0.813 |
| floor / random | | | 0.2303 / 0.3467 | | 0.2330 / 0.3872 | |

## What it says, and what it does not

1. **The cheapest baselines are as good as anything here, and better than the gate on this risk.** Max-softmax and mean entropy (one pass, no extra model, about 80 ms) reach out-of-fold
   AURC 0.269 and 0.270 against 0.303 for the production gate, and 0.285 against 0.308 on test. TTA entropy is best on test (AURC 0.273, AUROC 0.850) at six passes. Every interval
   overlaps every other (the widest are on test, 4 sequences), so no detector is shown to be better than another; what the table does show is that nothing more expensive is shown to
   be better than max-softmax.
2. **The gate was built for a different target.** It regresses INSTRUMENT IoU; this risk is the 12-class official IoU, which includes the anatomy classes where, in the label-noise
   probe (Task L), the largest out-of-fold errors lie. On the label the gate targets (instrument IoU below 0.5, the last-but-one column of `failure_detection_cost.md`) it is the best or
   near-best detector (AUROC 0.788 out of fold, 0.844 on test, against 0.704 and 0.710 for max-softmax). The figure "bad-frame detection AUROC 0.84" quoted on the site is on that label;
   on the official risk the same gate scores 0.771 on test. Both are true; neither was adjusted.
3. **Cost against AURC.** The 4-model ensemble costs four forward passes and four trained models and reaches test AURC 0.312, no better than one-pass max-softmax (0.285). The cross-view
   gate costs two forward passes and one trained model and reaches 0.305 on test (0.298 out of fold): near the ensemble at half the passes, the efficiency pattern the design predicted,
   but it is still behind one-pass max-softmax, and it did not pass its own pre-set test (Task D). Soft Dice Confidence, a one-pass method, is the weakest baseline on test.
4. **What is left to gain.** On this risk the best detectors close about two thirds of the gap between random and oracle ranking out of fold ((0.3467 - 0.2692) / (0.3467 - 0.2303)).
5. **The disparity control** (gate with the three disparity indicators) needs SGBM stereo matching on the CPU, about 1.7 s per frame; its ms/frame is composed from measured parts and marked.

Limits: the risk is a mean over 4-8 classes per frame and the bad-frame threshold is fixed at 0.5; 15 and 4 sequence clusters give wide intervals; the baselines were not tuned, the gate
was tuned on instrument IoU. The latency is measured on one RTX 3050 Ti laptop GPU at batch 1 with the image already on the GPU.
