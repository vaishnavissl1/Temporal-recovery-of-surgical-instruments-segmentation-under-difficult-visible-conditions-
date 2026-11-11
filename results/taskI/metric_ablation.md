| setting | mIoU | change vs reference (points) | frames | classes scored |
|---|---|---|---|---|
| reference: official protocol | 0.6128 | +0.00 | 997 | per frame |
| background included in the per-frame mean | 0.6452 | +3.23 | 997 | per frame |
| class counted if in ground truth OR prediction | 0.5021 | -11.07 | 997 | per frame |
| per-class mean over frames, then over classes (mcIoU) | 0.4956 | -11.72 | 997 | 10 |
| sequence means averaged unweighted | 0.6128 | -0.01 | 997 | per frame |
| evaluated at 512x640 (ground truth downsampled) | 0.6132 | +0.04 | 997 | per frame |
| background included + ground-truth-or-prediction presence | 0.5447 | -6.81 | 997 | per frame |
| background included, evaluated at 512x640 | 0.6455 | +3.27 | 997 | per frame |
| per-class mean (mcIoU) + background included | 0.5218 | -9.11 | 997 | 11 |
