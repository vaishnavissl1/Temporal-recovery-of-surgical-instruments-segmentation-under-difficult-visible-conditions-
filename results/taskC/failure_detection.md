### Out of fold (15 training sequences, each scored by the fold model that excluded it)

2235 frames with a defined risk; bad-frame rate 13.9%.

| Detector | AUROC | 95% | AURC | 95% | AUROC (instr.-IoU label) | passes | segmenters trained |
|---|---|---|---|---|---|---|---|
| max-softmax | 0.786 | 0.676 to 0.880 | 0.2692 | 0.2226 to 0.3332 | 0.704 | 1 | 1 |
| mean entropy | 0.773 | 0.662 to 0.874 | 0.2704 | 0.2227 to 0.3344 | 0.700 | 1 | 1 |
| Soft Dice Confidence | 0.734 | 0.672 to 0.810 | 0.2849 | 0.2384 to 0.3426 | 0.765 | 1 | 1 |
| MC stochastic depth (10 passes) | 0.741 | 0.670 to 0.842 | 0.2881 | 0.2369 to 0.3499 | 0.753 | 10 | 1 |
| TTA entropy (6 passes) | 0.764 | 0.649 to 0.867 | 0.2712 | 0.2226 to 0.3362 | 0.692 | 6 | 1 |
| TTA pairwise Dice (6 passes) | 0.757 | 0.688 to 0.867 | 0.2823 | 0.2317 to 0.3455 | 0.772 | 6 | 1 |
| gate: confidence only | 0.684 | 0.585 to 0.760 | 0.3107 | 0.2609 to 0.3641 | 0.749 | 1 | 1 |
| gate: + temporal (9 indicators) | 0.718 | 0.593 to 0.801 | 0.3075 | 0.2593 to 0.3603 | 0.770 | 1 | 1 |
| gate (GBM): 4 original indicators | 0.718 | 0.571 to 0.826 | 0.3164 | 0.2685 to 0.3650 | 0.745 | 1 | 1 |
| gate (GBM): 4 + 3 disparity indicators (control) | 0.747 | 0.620 to 0.852 | 0.3098 | 0.2607 to 0.3601 | 0.775 | 1 | 1 |
| gate (GBM): 4 + 6 cross-view indicators | 0.729 | 0.633 to 0.812 | 0.2980 | 0.2457 to 0.3554 | 0.794 | 2 | 1 |
| gate (GBM): 6 cross-view indicators alone | 0.655 | 0.560 to 0.772 | 0.2972 | 0.2421 to 0.3613 | 0.713 | 2 | 1 |
| gate: production (MLP, 9 indicators) | 0.740 | 0.618 to 0.821 | 0.3034 | 0.2549 to 0.3571 | 0.788 | 1 | 1 |

AURC floor (oracle ranking) 0.2303; random ranking 0.3467.

### Official test (4 sequences, final model)

997 frames with a defined risk; bad-frame rate 29.5%.

| Detector | AUROC | 95% | AURC | 95% | AUROC (instr.-IoU label) | passes | segmenters trained |
|---|---|---|---|---|---|---|---|
| max-softmax | 0.792 | 0.641 to 0.867 | 0.2852 | 0.1940 to 0.4755 | 0.710 | 1 | 1 |
| mean entropy | 0.789 | 0.647 to 0.874 | 0.2853 | 0.1939 to 0.4729 | 0.709 | 1 | 1 |
| Soft Dice Confidence | 0.685 | 0.521 to 0.800 | 0.3407 | 0.2240 to 0.4549 | 0.780 | 1 | 1 |
| MC stochastic depth (10 passes) | 0.792 | 0.553 to 0.904 | 0.3255 | 0.2145 to 0.4317 | 0.855 | 10 | 1 |
| TTA entropy (6 passes) | 0.850 | 0.688 to 0.917 | 0.2727 | 0.1919 to 0.4565 | 0.778 | 6 | 1 |
| TTA pairwise Dice (6 passes) | 0.808 | 0.589 to 0.899 | 0.3203 | 0.2113 to 0.4310 | 0.864 | 6 | 1 |
| 4-model ensemble pairwise Dice | 0.840 | 0.583 to 0.926 | 0.3120 | 0.1926 to 0.4337 | 0.870 | 4 | 4 |
| gate: confidence only | 0.761 | 0.416 to 0.925 | 0.3191 | 0.1980 to 0.4611 | 0.847 | 1 | 1 |
| gate: + temporal (9 indicators) | 0.757 | 0.379 to 0.934 | 0.3119 | 0.1954 to 0.4649 | 0.845 | 1 | 1 |
| gate (GBM): 4 original indicators | 0.772 | 0.453 to 0.944 | 0.3130 | 0.2023 to 0.4601 | 0.846 | 1 | 1 |
| gate (GBM): 4 + 3 disparity indicators (control) | 0.746 | 0.380 to 0.884 | 0.3257 | 0.2097 to 0.4626 | 0.830 | 1 | 1 |
| gate (GBM): 4 + 6 cross-view indicators | 0.813 | 0.536 to 0.952 | 0.3050 | 0.1997 to 0.4546 | 0.899 | 2 | 1 |
| gate (GBM): 6 cross-view indicators alone | 0.754 | 0.555 to 0.869 | 0.3233 | 0.2184 to 0.4647 | 0.829 | 2 | 1 |
| gate: production (MLP, 9 indicators) | 0.771 | 0.449 to 0.928 | 0.3076 | 0.1982 to 0.4667 | 0.844 | 1 | 1 |

AURC floor (oracle ranking) 0.2330; random ranking 0.3872.
