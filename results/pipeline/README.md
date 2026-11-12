# Pipeline images, one picture per step

Drawn from files the final evaluation already wrote. No model was re-run and no number here is new;
the per-frame IoU in the table is the official per-frame rule at the 512x640 working grid, so it can
differ in the third decimal from the full-resolution figures in the paper.

Each folder holds `01_input.png` ... `08_error_after.png` and `overview.png`.
`dataset_overview.png` shows one frame of each test sequence with its label and the 12 class colours.

Frames were picked by a fixed rule: per sequence the best-scored, the worst-scored, and the routed frame the correction helped most.

These are EndoVis frames: the PNGs stay local (git-ignored) and are not on the website until the frame-image licence is settled.

| folder | kind | gate r | flagged | action | IoU before | IoU after |
|---|---|---|---|---|---|---|
| seq_1_frame079_best | best | 0.734 | False | A0 | 0.9005 | 0.9005 |
| seq_1_frame248_worst | worst | 0.693 | False | A0 | 0.1697 | 0.1697 |
| seq_1_frame023_corrected | corrected | 0.573 | True | A3 | 0.4581 | 0.4681 |
| seq_2_frame006_best | best | 0.782 | False | A0 | 0.8679 | 0.8679 |
| seq_2_frame246_worst | worst | 0.711 | False | A0 | 0.3349 | 0.3349 |
| seq_2_frame116_corrected | corrected | 0.634 | True | A3 | 0.48 | 0.4805 |
| seq_3_frame123_best | best | 0.019 | False | A0 | 0.9595 | 0.9595 |
| seq_3_frame122_worst | worst | 0.256 | True | A3 | 0.3885 | 0.389 |
| seq_3_frame143_corrected | corrected | 0.636 | True | A3 | 0.8375 | 0.8474 |
| seq_4_frame046_best | best | 0.261 | True | A3 | 0.7997 | 0.7996 |
| seq_4_frame013_worst | worst | 0.365 | True | A3 | 0.101 | 0.1012 |
| seq_4_frame210_corrected | corrected | 0.003 | True | FUSE | 0.5219 | 0.5357 |
