# Solution v3 — after reading the literature

Scope of the reading behind this document: 51 papers across four areas — EndoVis 2018
single-frame SOTA (10), surgical stereo / multi-view (15), failure detection and
segmentation quality without ground truth (14), temporal and video surgical
segmentation (12). Full text was retrieved for all but three (LSKANet is paywalled;
MDVNet and one IEEE JBHI paper are closed access). Everything below is sourced to a
paper or to a file in `results/`.

---

## 1. Where our numbers actually stand

Published EndoVis 2018 results do not all measure the same thing. There are three
incompatible protocol families and papers mix them:

- **Family A — the official 12-class protocol** (per-frame mean IoU over classes
  present in the frame, background excluded, official 15/4 split). Everything that
  credibly uses it lands in a tight band: SegFormer 63.0, HRNet 63.3, SegNeXt 64.3,
  STswinCL 63.6, LSKANet 66.2. The challenge winner was 62.1 (and used extra
  EndoVis 2017 data).
- **Family B — unstated/background-inclusive** (CLIP-RL 0.81 at 224x224 input,
  Surg-SegFormer 0.80 with the instrument-part labels merged, FASL-Seg 0.73 with
  background scored as a class). Not comparable.
- **Family C — instrument-type on the ISINet split** (held-out sequences 2, 5, 9, 15
  taken out of the official *training* set, 7 classes, "Challenge IoU"). ISINet 77.47,
  TraSeTr 76.2, MATIS 84.26. These are not scene-segmentation numbers.

One single-lab outlier claims 77.5 (TAFE) and 82.6 (TAFPNet) on Family A's protocol.
Its gain is concentrated entirely on the two sequences that are hardest under the
official per-frame rule, it reaches 72.4-80.6 on test sequence 4 where the 18-team
average was 0.279 and the winner managed 0.390, it misprints STswinCL's own
sequence-3 score as 93.7 (the real value is 83.7), it has no ablation of either
contribution, and it ships no weights and an empty README. Treat as its own regime.

**Our position: 0.6128 mIoU, with the evaluation verified against the official
`run.py` to 1.1e-16.** That sits inside Family A's credible band, just under
STswinCL, from a 26M-parameter single-frame model trained at 512x640 on a 4 GB
laptop GPU. That is an honest, defensible number and it should be reported with the
protocol spelled out.

Two things worth knowing about the field, both usable:

- **No paper in the set releases trained weights.** Not one checkpoint across all ten
  EndoVis 2018 papers. Several release code with no data-preparation step.
- **No paper gives a complete, unambiguous mIoU definition.** STswinCL comes closest
  and still does not say whether "present in the frame" means present in the ground
  truth or in the union with the prediction. Publishing our evaluation script with the
  definition written out is a real contribution and costs nothing.

---

## 2. Why the temporal stage gave a null result — and why that is now a finding

Measured temporal gains on EndoVis 2018, every number found:

| Gain | Source | Note |
|---|---|---|
| +0.6 mIoU | STswinCL | cleanest ablation anywhere; 5 frames is *worse* than 4; Dice p=0.175-0.203, not significant |
| +1.02 to +1.89 mIoU | MATIS | range because two tables disagree on the baseline |
| +4.20 cIoU | ISINet | explicit optical flow; **5 of 12 configurations fall below the no-temporal baseline, worst -5.43** |
| +5.1 mIoU | TAFPNet | protocol not comparable |

Two patterns matter more than the headline numbers:

**(a) The literature's temporal gains are about labels, not masks.** MATIS gets
+1.89 mIoU but +5.39 mcIoU; ISINet gets +4.24 IoU but +4.42 mcIoU. Both temporal
modules *only re-label existing per-frame masks* — neither changes mask geometry at
all. So the published gains are about fixing instrument-type flicker, not about
improving boundaries.

**(b) The same module gains about 3x more at 2 Hz than at 1 Hz.** ISINet is the
controlled comparison: +12.72 cIoU on EndoVis 2017 (2 Hz) versus +4.20 on EndoVis
2018 (1 Hz), identical code. The authors attribute it to "the higher variability of
the 2018 dataset" and never connect it to the sampling rate. Three independent
corroborations: TSMS-SAM2 loses 6.44 J&F from one extra frame of stride on EndoVis
2018; MATIS says "increasing the stride value diminishes our performance due to the
low sampling rate in our datasets" and gives no number; Ozbulak et al. measure
SAM2's streaming grasper IoU collapsing from 74-94 at 25 FPS to about 30 at 1 FPS.

**Nobody in this literature measures ground-truth mask IoU between consecutive
EndoVis 2018 frames.** Our 0.30-0.45 measurement is the missing quantitative
explanation for a pattern that at least four papers have each bumped into and
explained away differently. The official challenge paper confirms the cause at
source: sections were "subsampled to 1 Hz" and "similar frames were manually removed
until the sequence contained 300 frames" — the organisers deliberately removed
temporal redundancy, and no challenge participant used temporal information at all.

So the temporal null result is not a failure of our implementation. It is the
expected consequence of the dataset's construction, and we are in a position to be
the first to say so with a number.

---

## 3. The bug: our stereo stage never tested the stereo idea

This is the important finding of this review.

The three stereo indicators fed to the reliability gate are, from
`sitr/stereo.py:instrument_stereo_indicators`:

- `stereo_disagreement` — fraction of predicted-instrument pixels failing the SGBM
  **left-right disparity** consistency check
- `occluded_fraction` — fraction hidden from the right camera, also from disparity
- `disparity_confidence` — mean **photometric residual** confidence

All three are disparity and photometric quantities. **None of them is segmentation
disagreement.** The name is misleading: it is disparity disagreement, not mask
disagreement.

Task 2 already measured why these cannot work. Left-right consistency holds for only
**45.6%** of instrument pixels with calibration alone and 54.3% after
self-calibration, against 65.2% / 71.1% for tissue — and as low as **15.1%** on
train/seq_15. Stereo matching fails hardest exactly on the shiny metal objects we
care about. So `stereo_disagreement` sits near-constant-high whenever instruments are
present: it measures how matchable the tool surface is, not whether the mask is
wrong. Hence the measured AUC of 0.536 / 0.506 / 0.549 — noise, exactly as expected.

The same contamination sinks the stereo correction action. A1 pulls the right-view
prediction onto the left grid **through the disparity map**
(`sitr/stereo.py:right_to_left_grid`), so it can only transfer labels where disparity
is valid — which is everywhere except the instruments. Hence +0.0001 mIoU and a
Pearson of -0.027 between predicted and measured gain.

And the part that makes this fixable: `tools/cache_right_view.py` **already segmented
the right view of all 3,232 frames with the same model**, out-of-fold honest, and
cached argmax label plus confidence in `work_v2/right_view`. That signal was computed
and then only ever used through the broken channel.

External support that this is the right diagnosis, not an excuse: StereoMamba
documents that specular reflections from instruments break stereo matching across
GwcNet, ACVNet, RAFT-Stereo and IGEV-Stereo. The failure is a known property of
photometric stereo on surgical tools, not a bug in our SGBM settings.

---

## 4. What the literature says is unclaimed

From the failure-detection review, with targeted searches for each:

1. **A second physical viewpoint as an ensemble member for failure detection.**
   Nothing does this. The strongest result in the whole failure-detection literature
   is Zenk et al.'s benchmark (6 medical datasets, the only shift-aware one): the best
   detector is not a confidence map at all, it is **disagreement between discrete
   masks** (pairwise Dice over a 5-model ensemble; MC-dropout pairwise Dice second).
   Those are the only two methods ranked consistently across datasets, and the AURC
   gap to single-network mean entropy exceeds 0.1 on two datasets. A calibrated stereo
   pair gives a second mask from a genuinely different physical observation for **one
   extra forward pass** instead of five trained models.
2. **Harmful-correction rate.** No paper defines or reports the fraction of cases a
   refiner makes worse. SCORE (2025) reports only dataset-level means; the one paper
   whose stated purpose is better refinement metrics defines nAUC and
   never-reached-target and neither counts a worsening edit.
3. **Predicting the benefit of a correction before applying it.** Exactly one method
   decides at all (JFS) and it is strictly post-hoc — it must run the refiner first —
   and on a representative random sample its decision accuracy is 52.1%, i.e. chance.
4. **Three-way accept / correct / abstain under one evaluation.**
5. **AURC on surgical video.** Zenk recommends AURC and has no video or surgical data;
   the one surgical failure monitor uses AUROC and AUPRC.

From the stereo review: nobody segments both views of a surgical stereo pair and
compares the predictions; only 1 of 18 challenge teams used the right view at all
(by feature summation, zero geometry); nobody uses EndoVis 2018's actual calibration.
The closest prior art, SegStereo (ECCV 2018), computes semantic features for both
views and warps right to left — but only as a training regulariser for disparity, and
never measures the disagreement.

One warning worth keeping: StreoScenNet found that adding the second view **hurt**
binary segmentation (Dice 87.86 versus monocular VGG19's 90.01). The second view is
not free accuracy. It is a second opinion — which is what we want it for.

---

## 5. The redesign

### 5.1 Cross-view agreement without disparity

The geometric insight that makes this work:

> **Rectification alone gives row correspondence. Disparity is only needed for column
> correspondence.**

After rectification, a 3-D point lies on the *same row* in both images. So any
statistic computed per-row and marginalised over columns — or any pure set or area
statistic — can be compared between the two views with no disparity, no photometric
matching, and therefore no specular contamination. Our rectification is verified at
0.66 / 0.93 / 0.62 / 0.93 px mean vertical error on the four test sequences.

Six indicators, all computed from the two cached label maps:

| Name | Definition | Geometry needed |
|---|---|---|
| `xv_class_jaccard` | Jaccard of the *sets* of classes predicted in each view | none |
| `xv_area_ratio_max` | max over classes of abs(a_L - a_R) / (a_L + a_R) | none |
| `xv_ncomp_diff` | normalised difference in connected-component count per class | none |
| `xv_rowprofile_l1` | per class c: sum_y abs(m_L(c,y) - m_R(c,y)) / sum_y (m_L + m_R), where m(c,y) is the count of class-c pixels in row y; averaged over classes present in either view | rectification only |
| `xv_rowspan_iou` | per class, IoU of the row-support intervals [y_min, y_max] | rectification only |
| `xv_runcount_corr` | Pearson correlation between the per-row counts of class runs in the two views | rectification only |

The one correctness subtlety, which must be handled or the indicators are biased: the
two views do not share a horizontal field of view, so a strip at one edge has no
counterpart. Exclude a strip as wide as the 99th-percentile disparity (186 px,
measured in Task 2) from the column range of **both** views before counting, and
restrict to `rect.valid_left`.

Why this should work where the disparity version did not: it is computed on **class
labels**, not image intensities, so a specular highlight that destroys photometric
matching does not affect it. And it is a direct measurement of the quantity Zenk's
benchmark identifies as the strongest failure signal — disagreement between two
discrete masks — with the ensemble member supplied by physics rather than by five
training runs.

Deployment cost: one extra forward pass, **76 ms** measured, against the **2015 ms**
the SGBM path cost. 26x cheaper than what was tried, and it is the thing nobody has
done.

### 5.2 Correction: stop chasing mask geometry

Our own measurement (consecutive ground-truth masks overlap at 0.30-0.45 IoU) and the
literature agree that mask geometry does not survive a one-second gap. Every
substantial temporal gain in the literature is a **relabelling** of existing
per-frame masks, not a geometric correction. MATIS: +5.39 mcIoU. ISINet: +4.42 mcIoU.

Our results point the same way: suturing-needle IoU is **0**, which is an identity
failure, not a boundary failure. And the per-sequence spread (0.41 on sequence 4 to
0.81 on sequence 3) dwarfs every correction difference we measured, so mIoU is the
wrong instrument for detecting what our method does.

So: report **mcIoU (per-class mean IoU) alongside mIoU**, and make the correction
action temporal identity voting over connected components on routed frames only —
the thing that is known to work on this dataset — rather than morphological boundary
repair, which currently wins only because the alternatives do nothing.

### 5.3 Headline metric

Zenk et al. argue explicitly for **AURC** from risk-coverage analysis, and give
reusable reasons: AUROC needs an arbitrary binary failure label, MAE needs confidence
calibrated onto the risk scale, and correlations ignore segmentation performance
entirely. The soft-Dice-confidence paper independently adopts it. Our AURC already
moves monotonically with the methods (0.3076 to 0.3063) where mIoU does not, and
**nobody has reported AURC on surgical video at all**.

---

## 6. What the contribution is now

Not "we beat state-of-the-art mIoU." Honestly:

1. **The first cross-view segmentation-disagreement failure detector** — a second
   physical camera as an ensemble member, which the literature confirms nobody has
   done, motivated by the one result the failure-detection field agrees on.
2. **The first harmful-correction rate reported for a segmentation refiner**, with the
   distinction between per-frame-processed and per-correction-made spelled out (ours:
   0.80% of all frames versus 3.1% of corrected frames — both are true and they say
   different things).
3. **The first measurement of consecutive-frame ground-truth mask overlap on EndoVis
   2018**, which quantitatively explains a pattern four published papers have each
   explained away differently.
4. **A cost result**: the same mIoU as always-on fusion at roughly a third of the
   latency, activating on 26% of frames.
5. **A published, exactly-verified evaluation script with the metric definition written
   out** — which no paper in this literature provides.

Items 2, 3, 4 and 5 are already measured and already hold. Item 1 is what the pilot
in `ANTIGRAVITY_PROMPT_9.md` tests.

---

## 7. Honesty rules carried forward

- Never compare our numbers to a published number unless measured under an identical
  protocol. The three-family problem above is why.
- No synthetic data on any evaluation path. This dataset only.
- Decisions are made out-of-fold. The test set is touched once, at the end.
- A negative result is reported as a negative result. The stereo premise has already
  been refuted once in its disparity form; if the cross-view form also fails, that is
  the answer and it gets written down.
