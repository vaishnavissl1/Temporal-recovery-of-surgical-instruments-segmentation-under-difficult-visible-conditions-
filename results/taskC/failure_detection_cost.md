### Out of fold (15 training sequences; each frame scored by the cross-fitting model that excluded its sequence)

2235 frames, bad-frame rate 13.9% (official IoU < 0.5). AURC floor 0.2303, random ranking 0.3467.

| Detector | AUROC (95%) | AURC (95%) | passes | segmenters trained | ms/frame |
|---|---|---|---|---|---|
| max-softmax | 0.786 (0.676-0.880) | 0.2692 (0.2226-0.3332) | 1 | 1 | 80 |
| mean entropy | 0.773 (0.662-0.874) | 0.2704 (0.2227-0.3344) | 1 | 1 | 80 |
| Soft Dice Confidence | 0.734 (0.672-0.810) | 0.2849 (0.2384-0.3426) | 1 | 1 | 85 |
| MC stochastic depth (10 passes) | 0.741 (0.670-0.842) | 0.2881 (0.2369-0.3499) | 10 | 1 | 782 |
| TTA entropy (6 passes) | 0.764 (0.649-0.867) | 0.2712 (0.2226-0.3362) | 6 | 1 | 535 |
| TTA pairwise Dice (6 passes) | 0.757 (0.688-0.867) | 0.2823 (0.2317-0.3455) | 6 | 1 | 557 |
| gate: confidence only | 0.684 (0.585-0.760) | 0.3107 (0.2609-0.3641) | 1 | 1 | 109 |
| gate: + temporal (9 indicators) | 0.718 (0.593-0.801) | 0.3075 (0.2593-0.3603) | 1 | 1 | 109 |
| gate (GBM): 4 original indicators | 0.718 (0.571-0.826) | 0.3164 (0.2685-0.3650) | 1 | 1 | 109 |
| gate (GBM): 4 + 3 disparity indicators (control) | 0.747 (0.620-0.852) | 0.3098 (0.2607-0.3601) | 1 | 1 | 1858* |
| gate (GBM): 4 + 6 cross-view indicators | 0.729 (0.633-0.812) | 0.2980 (0.2457-0.3554) | 2 | 1 | 288 |
| gate (GBM): 6 cross-view indicators alone | 0.655 (0.560-0.772) | 0.2972 (0.2421-0.3613) | 2 | 1 | 288 |
| gate: production (MLP, 9 indicators) | 0.740 (0.618-0.821) | 0.3034 (0.2549-0.3571) | 1 | 1 | 109 |

### Official test (4 sequences)

997 frames, bad-frame rate 29.5% (official IoU < 0.5). AURC floor 0.2330, random ranking 0.3872.

| Detector | AUROC (95%) | AURC (95%) | passes | segmenters trained | ms/frame |
|---|---|---|---|---|---|
| max-softmax | 0.792 (0.641-0.867) | 0.2852 (0.1940-0.4755) | 1 | 1 | 80 |
| mean entropy | 0.789 (0.647-0.874) | 0.2853 (0.1939-0.4729) | 1 | 1 | 80 |
| Soft Dice Confidence | 0.685 (0.521-0.800) | 0.3407 (0.2240-0.4549) | 1 | 1 | 85 |
| MC stochastic depth (10 passes) | 0.792 (0.553-0.904) | 0.3255 (0.2145-0.4317) | 10 | 1 | 782 |
| TTA entropy (6 passes) | 0.850 (0.688-0.917) | 0.2727 (0.1919-0.4565) | 6 | 1 | 535 |
| TTA pairwise Dice (6 passes) | 0.808 (0.589-0.899) | 0.3203 (0.2113-0.4310) | 6 | 1 | 557 |
| 4-model ensemble pairwise Dice (test only) | 0.840 (0.583-0.926) | 0.3120 (0.1926-0.4337) | 4 | 4 | 312 |
| gate: confidence only | 0.761 (0.416-0.925) | 0.3191 (0.1980-0.4611) | 1 | 1 | 109 |
| gate: + temporal (9 indicators) | 0.757 (0.379-0.934) | 0.3119 (0.1954-0.4649) | 1 | 1 | 109 |
| gate (GBM): 4 original indicators | 0.772 (0.453-0.944) | 0.3130 (0.2023-0.4601) | 1 | 1 | 109 |
| gate (GBM): 4 + 3 disparity indicators (control) | 0.746 (0.380-0.884) | 0.3257 (0.2097-0.4626) | 1 | 1 | 1858* |
| gate (GBM): 4 + 6 cross-view indicators | 0.813 (0.536-0.952) | 0.3050 (0.1997-0.4546) | 2 | 1 | 288 |
| gate (GBM): 6 cross-view indicators alone | 0.754 (0.555-0.869) | 0.3233 (0.2184-0.4647) | 2 | 1 | 288 |
| gate: production (MLP, 9 indicators) | 0.771 (0.449-0.928) | 0.3076 (0.1982-0.4667) | 1 | 1 | 109 |

`*` composed from measured components, not timed end to end. The intervals resample whole sequences (15 out of fold, 4 on test), so they are wide and overlap heavily.
Hardware: NVIDIA GeForce RTX 3050 Ti Laptop GPU, batch 1. wall clock with the GPU synchronised before and after, mean and P95 over the frames; measured, not composed.
