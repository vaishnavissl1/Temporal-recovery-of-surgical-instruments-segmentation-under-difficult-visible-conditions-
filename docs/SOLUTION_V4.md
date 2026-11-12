# Solution v4 — the complete design

Supersedes v3's section 5. Read with `REFERENCES.md` (citation keys A1, B1, E3 ... are
used throughout) and `DATASET_FACTS.md` (every measured number).

---

## 1. The problem statement, stated precisely

Given a stereo endoscopic sequence sampled at 1 Hz, with only the left view labelled:

1. produce a 12-class scene segmentation of each left frame;
2. decide, **without ground truth**, which frames the segmenter has failed on;
3. decide per frame whether to **accept**, **correct**, or **abstain**;
4. do (3) without making masks worse, and spend compute only where it is needed.

Part 1 is bounded by the dataset and is a crowded problem: credible protocol-matched
results sit in a 50-66 mIoU band (A1, B1, B2) and have moved 1-3 points in six years.
Parts 2, 3 and 4 are where the problem is genuinely open — no published method on this
dataset abstains, none reports whether its corrections ever hurt, and none measures
quality without ground truth at all (G4, G5, G7). **"Solving the problem statement
completely" therefore means being complete on 2, 3 and 4, and honest and competitive on
1** — not beating a number that different papers compute differently.

---

## 2. What changes from the current pipeline, and why

| Stage | Now | v4 | Reason |
|---|---|---|---|
| S1 segmenter | CAFormer-S18 + SurgeNetXL, FPN + full-res branch, train 512x640, eval 1280x1024 | **unchanged**, plus rare-class work (section 4) | 0.6128 is inside the credible band; the fault is per-class, not global |
| S2 gate inputs | 4 confidence + 3 **disparity** + temporal | **drop the 3 disparity indicators**; add **6 cross-view indicators**; keep confidence + temporal; add within-sequence-normalised blur and specular fraction | The disparity indicators were measured at AUC 0.51-0.55 and cost 2015 ms. Section 3 explains why they cannot work and what replaces them |
| S3 decision | gate threshold -> route | **accept / correct / abstain**, with a **pre-hoc predicted-gain** gate | Three-way under one evaluation is unoccupied (G5); E8 is the only competitor and is post-hoc at chance accuracy (52.1%) |
| S4 actions | A1 stereo-warp, A2 temporal, A3 morphological, fusion | **retire A1**; A2 becomes **identity voting**, not geometry; keep A3; add **A4 hallucination repair** | A1 is disparity-bound (section 3). Every substantial temporal gain in the literature is relabelling, not geometry (C1 +5.39 mcIoU, B8 +4.42 mcIoU) |
| S5 identity | teammate's stage | unchanged interface | Contract already fixed |
| Metrics | mIoU, boundary F1, harmful rate, AURC | add **mcIoU**, **clDice** per thin class, **hallucination rate**; AURC becomes a headline | E3 recommends AURC and nobody reports it on surgical video (G7); mcIoU is where relabelling gains are visible |
| Folds | grouped by sequence | grouped by **camera rig** | Test seq 1-3 share a rig with train seq 9-12; test seq 4 with train 13/15/16. A sequence split still leaks rig |

---

## 3. The central correction: detection and correction are not symmetric

The bug (full diagnosis in `SOLUTION_V3.md` section 3): the gate's three "stereo"
indicators are SGBM disparity and photometric-residual quantities, not segmentation
disagreement. Task 2 measured left-right disparity consistency at 45.6% on instrument
pixels against 65.2% on tissue, and 15.1% on train/seq_15 — so those indicators measure
tool reflectivity, not segmentation error. D7 documents the same failure across four
modern learned stereo matchers. D5 is the paper those indicators actually reproduce, and
it is explicitly about disparity confidence, not segmentation.

The fix rests on one geometric fact:

> **Rectification alone gives row correspondence. Disparity is only needed for column
> correspondence.**

After rectification a 3-D point lies on the *same row* in both images. Our rectification
is verified at 0.66 / 0.93 / 0.62 / 0.93 px mean vertical error on the four test
sequences. So:

- **Detection needs only rows.** Per-row and set statistics compare the two predicted
  masks with no disparity, no photometric matching, and therefore no specular
  contamination — because they are computed on **class labels**, not intensities.
- **Pixel-level correction needs columns.** Fusing two masks pixel-by-pixel requires
  disparity, which is exactly the contaminated channel. So cross-view evidence **cannot**
  be used for pixel-level correction on this data.

That asymmetry is the design, and it is a result worth stating in the write-up: the
second camera is a good *auditor* and a bad *corrector*, and the reason is measurable.
It also explains D2's negative result (StreoScenNet found the second view **hurt** binary
segmentation, Dice 87.86 vs 90.01) — fusion was the wrong use of it.

### 3.1 The six cross-view indicators

All from the two cached label maps. `work_v2/right_view` already holds the right-view
segmentation of all **3,232** frames from the same model, out-of-fold honest, produced
in 10.3 minutes of GPU. No retraining, no re-inference.

| Indicator | Definition | Geometry |
|---|---|---|
| `xv_class_jaccard` | Jaccard of the two *sets* of classes predicted | none |
| `xv_area_ratio_max` | max over classes of abs(a_L - a_R)/(a_L + a_R) | none |
| `xv_ncomp_diff` | mean over classes of abs(n_L - n_R)/(n_L + n_R), n = 8-connected components | none |
| `xv_rowprofile_l1` | per class, sum_y abs(m_L(c,y) - m_R(c,y)) / sum_y (m_L + m_R), m(c,y) = class-c pixel count in row y; averaged over classes present in either view | rectification |
| `xv_rowspan_iou` | per class, IoU of row-support intervals [y_min, y_max] | rectification |
| `xv_runcount_corr` | Pearson correlation of the per-row counts of class runs between views | rectification |

Why this is the right signal and not a guess: E3 — the only shift-aware benchmark in
medical failure detection — finds the strongest detector is **disagreement between
discrete masks**, not any confidence map, and that pixel-confidence aggregation is weak
(E6 measured max-softmax at 0.536, chance; E5 found mean entropy can be *worse than
random*). A calibrated stereo pair supplies that second mask from physics, for **one
extra forward pass** instead of E3's five trained models or E4's 20-40 stochastic passes.
A second viewpoint also decorrelates failure modes that augmentation and dropout cannot
touch: specular highlights, smoke, blood and occlusion by tissue are
**viewpoint-dependent**, whereas a flip or a dropout mask leaves them identical.

**Mandatory field-of-view guard.** The two views do not share a horizontal field of
view, so an edge strip has no counterpart and would bias every count. Drop a strip as
wide as the 99th-percentile disparity (**186 px** at full resolution, measured) from the
left edge of the left view and the right edge of the right view, intersect with
`rect.valid_left`, and report the surviving column fraction. Without this guard all six
indicators are biased.

### 3.2 Decision rule, fixed before any number is seen

The cross-view indicators count as working if, **out of fold**, adding them to the four
confidence indicators improves AUROC by **at least +0.02 with a sequence-clustered 95%
interval excluding zero**. If they fail, the negative result is written up and the gate
keeps the confidence and temporal indicators. No searching for a variant that passes.
Intervals are clustered by sequence, never by frame — E6's cautionary CI of [0.48, 0.99]
from three test sequences is why.

---

## 4. Using the entire dataset

| Component | Measured contents | How v4 uses it |
|---|---|---|
| `left_frames/` | 3,234 x 1280x1024 RGB PNG | S1 input; the only labelled view |
| `right_frames/` | 3,234, 1:1 with left, same size, **never labelled** | **The six cross-view indicators.** This is half the pixel data in the dataset and 1 of 18 challenge teams touched it (D2). Already segmented and cached |
| `labels/` | 3,239 RGBA uint8, exactly 12 palette colours, zero strays | Exact RGB lookup to ids 0-11; nearest-neighbour resize only, never interpolation of class ids |
| `camera_calibration.txt` | **7 distinct files for 19 sequences** (per rig, not per sequence); full stereo extrinsics everywhere | (a) `cv2.stereoRectify` -> the row correspondence the whole design rests on; (b) **rig-grouped folds**; (c) metric-size plausibility: Z = f*B/d with f*B = 4716 px.mm, so a predicted shaft of implausible 3-D width is a geometry-only failure cue independent of the network |
| `repairs/` | 7 masks, changing 2-26% of a frame | Applied as override in training and CV; also used as a **label-noise probe** — frames whose cross-fitted loss is extreme are flagged, to separate label noise from model error |
| Frame order at 1 Hz | 149 per train sequence, ~250 per test | Temporal memory for A2 identity voting; whole-sequence splits |
| `test_data/run.py`, `utils.py` | The official protocol | Metric reproduced to 1.1e-16; and see section 6 for the consequence of its ground-truth-only class rule |
| Release `labels.json` | 3 of 4 releases; release 1 lists 11 classes, release 4 has none | Union class table. `intestine` == `small-intestine`, one colour |
| Per-class statistics | needle 0.0026% of pixels in **12** train frames vs **95** test frames | Fold construction, clipped class weights, and the rare-class programme below |
| Image properties | blur varies ~10x between sequences (Laplacian variance ~40 vs ~460) | Blur **normalised within sequence**; raw blur cannot be thresholded across sequences |

**Not used, deliberately:** test-frame pixels for any training or tuning; the two
unlabelled test frames; monocular depth (real stereo exists); any synthetic data on any
evaluation path.

### 4.1 The rare-class programme — where mcIoU is actually lost

Suturing-needle IoU is **0**, and needle appears in **9.5% of test frames**. This is the
single largest per-class loss and it is a *data* problem, not an architecture problem:
12 training frames against 95 test frames, and every test needle frame is in test seq_1.
Three legitimate levers, in order of expected value:

1. **Copy-paste augmentation of real annotated needle pixels** within the training set
   only — pasting the 12 real needle instances onto other training frames with their
   real masks. These are real annotated pixels relocated, not generated data, and they
   never touch any evaluation path. State it plainly as augmentation.
2. **The right view doubles the rare-class pixels available for augmentation** (24 views
   of real needles instead of 12), by warping the left label to the right frame only
   where the occlusion mask is clean. Risky near occlusion boundaries; keep it as an
   ablation, not the default.
3. **clDice loss on thread and needle** (F3) rather than cross-entropy alone — these are
   thin tubular structures, which is exactly what clDice was built for, and our current
   thin-class clDice is 0.901 / 0.764 / **0.000** for shaft / thread / needle.

Report needle, thread and clamps per-class scores with the caveat that each comes from
**one test sequence**, so their variance is large by construction.

---

## 5. Compute — the optimisation

Retiring the disparity path is the single biggest win. Measured, per frame:

| Path | Now | v4 | Note |
|---|---|---|---|
| Segmentation only | 76 ms | 76 ms | |
| Disparity + stereo evidence | **2015 ms** | **retired** | AUC 0.51-0.55, i.e. noise |
| Cross-view indicators | n/a | **76 ms** (one extra forward pass) + ~5 ms CPU | 26x cheaper than the path it replaces |
| Blind fusion, every frame | 2201 ms | n/a | |
| Selective policy | 792 ms | **~96 ms** mean (one extra pass on the ~26% routed) | ~23x faster than blind fusion |

So v4 is both the more novel design and the cheaper one, and it fits 4 GB of VRAM at
full evaluation resolution with the already-measured headroom.

---

## 6. Metrics, and one insight from the official code

Headline set, all with sequence-clustered intervals:

- **Official mIoU** — exactly as `run.py` computes it, protocol written out in prose.
- **mcIoU (per-class mean IoU)** — the second headline. Every substantial temporal gain
  in the literature shows up here and is invisible in mIoU (C1: +1.89 mIoU but
  +5.39 mcIoU; B8: +4.24 IoU but +4.42 mcIoU). Our per-sequence mIoU spread is 0.41 to
  0.81, which dwarfs every correction effect, so mIoU alone cannot resolve what the
  method does.
- **AURC** from risk-coverage, risk = 1 - IoU. E3's recommended metric, and nobody has
  reported it on surgical video (G7). Ours already moves monotonically with the methods
  (0.3076 -> 0.3063) where mIoU does not.
- **Boundary F1 and clDice** per thin class.
- **Harmful-correction rate, both denominators** — percentage of all frames and
  percentage of corrected frames. Ours are 0.80% and 3.1%: both true, saying different
  things, and reporting only one is how this gets misread. No paper reports either (G4).
- **Hallucination rate** (new, and see below).
- **Latency**, measured, with the hardware stated.

**The insight from `utils.py`.** A class counts for a frame only if it is present in the
**ground truth** (`if (ground_truth_not_overlap + overlap) == 0: return None`). So a
class the model invents but the ground truth lacks is **not penalised at all** by the
official metric. Two consequences:

1. A real failure mode — hallucinating a class — is invisible to the benchmark. We
   report a **hallucination rate** separately, and action A4 targets it.
2. `xv_class_jaccard`, the cheapest of the six cross-view indicators, detects precisely
   that failure, because a class invented in one view is rarely invented identically in
   the other. **The cross-view gate detects a failure the benchmark is blind to** — which
   is a cleaner argument for it than any mIoU delta, and does not depend on the +0.02
   AUROC test passing.

---

## 7. Edge cases — the register

Every one of these is measured, not hypothesised. Each needs a guard in code; this table
is the checklist.

| # | Edge case | Guard |
|---|---|---|
| 1 | Test has **997** labelled frames, not 999: `seq_3/frame249` and `seq_4/frame249` have no label | Iterate the **labels** folder, as `run.py` does. Never iterate `left_frames` |
| 2 | **suction-instrument has zero test pixels** — 11 active classes, not 12 | Per-frame class presence from GT; never divide by a fixed 12 |
| 3 | needle: 12 train frames vs 95 test frames, all in test seq_1 | Folds must never hold out seq_1 and seq_4 together; report needle variance as single-sequence |
| 4 | thread, clamps, needle appear in test **only in seq_1** | Same; state it beside every per-class number |
| 5 | `run.py` maps clasper->3, wrist->2 while `labels.json` says clasper=2, wrist=3 | Harmless for the mean (both sides remapped alike) but it **swaps the names** in per-class output. Report per-class IoU by name from our own table |
| 6 | **7 calibration files for 19 sequences** — per rig. Test seq 1-3 share a rig with train seq 9-12; test seq_4 with train 13/15/16 | Load each sequence's own file; **group folds by rig**. Do not claim unseen-rig generalisation from this test set — it does not measure that |
| 7 | 3 calibration files (seq 13, 15, 16) end in a corrupted dict line; seq_14 is CRLF with no final newline; 8 of 19 omit `Camera-0-Alpha` | Whitelist known keys, skip unparseable lines, default skew to 0 |
| 8 | Shipped calibration leaves a **3-14 px vertical tilt** on train seq 2, 4, 5, 7 (files list Omega_z ~ 0 while cameras are rolled ~0.3-0.8 deg) | Label-free per-sequence self-calibration, estimated on frames 0-35 and evaluated on disjoint frames. All 4 test sequences are already within 0.7-0.9 px. Without this, a cross-view indicator would partly measure each sequence's calibration error |
| 9 | Principal points differ ~100 px in x, so **raw disparity would need negative values** | SGBM only ever on rectified images. Never on raw pairs |
| 10 | The two views do not share a horizontal field of view | Drop a 186 px strip (p99 disparity) from both views before any cross-view count; intersect with `rect.valid_left` |
| 11 | Distortion coefficients labelled "radial" but are consistent with OpenCV `(k1,k2,p1,p2,k3)` | Both orders tested by measuring vertical disparity; difference < 0.15 px, choice recorded |
| 12 | Release 4 has **no** `labels.json`; release 1 lists 11 classes | Union class table |
| 13 | `__MACOSX/` shadow trees, `._*` files, and `.DS_Store` **inside** `labels/` and `left_frames/` of seq_1, 3, 4 | Junk filter must exclude all three, or seq_1/3/4 appear to have 150 frames |
| 14 | Masks are 1024x1280x**4** RGBA; OpenCV returns **BGRA** | Exact RGB lookup after channel reorder; never a nearest-colour search |
| 15 | 7 `repairs/` masks are **large** corrections (2-26% of a frame) | Applied as override everywhere, including CV |
| 16 | `seq_8` is genuinely absent | 15 training sequences, never 16 |
| 17 | `utils.py` imports `skimage`, which is not installed — the official script cannot run as shipped | Re-implemented and cross-checked to 1.1e-16 against the quoted lines |
| 18 | Blur (Laplacian variance) differs ~10x between sequences | Normalise within sequence; never threshold raw blur across sequences |
| 19 | Specular fraction averages 1-2% but spikes to **18.6%** on single frames | Keep it as a continuous covariate, not a threshold |
| 20 | Empty predicted foreground, or a class absent from both views | Stated constant returned, documented; **no NaN may reach the gate** |
| 21 | Frame order of the stereo/right-view caches versus the Stage 2 cache | Verified by count **and** filename, refusing to continue on mismatch — not assumed |
| 22 | Conformal risk on test is 0.077 against a 0.05 budget | Reported as not met. E11: the guarantee is marginal in expectation and assumes exchangeability, which video violates. Say so rather than claiming the guarantee |
| 23 | Frames within a sequence are correlated | All intervals clustered by sequence. Never a frame-level CI |
| 24 | Windows silently spills VRAM to system RAM instead of raising OOM | A step that becomes inexplicably slow is a memory check, not a code bug |
| 25 | A hallucinated class is invisible to the official metric | Hallucination rate reported separately; harmful-correction rate uses its own IoU-change definition against GT, alongside and not replacing the official score |

---

## 8. What is claimed, and what is not

**Claimed, already measured and holding:**

1. The evaluation reproduces the official metric exactly (1.1e-16), with the protocol
   written out — which no paper in section B provides (G9).
2. Consecutive ground-truth instrument masks overlap at **0.30-0.45 IoU**, with 60-77%
   of consecutive pairs below 0.5. The first such measurement on this dataset (G6), and
   the quantitative explanation for a pattern that B1, C1, B8 and C7 each explained away
   differently ("large surgical motion", "low sampling rate", "dataset variability",
   "excessive temporal gaps").
3. A harmful-correction rate, reported with both denominators. The first anywhere (G4).
4. Calibrated stereo rectification is usable on this data, with a label-free correction
   for the sequences where the shipped calibration is not (2.64 -> 1.21 px).
5. Selective correction matches always-on fusion's mIoU while activating on ~26% of
   frames, with v4 bringing that to roughly a twenty-third of the latency.
6. Disparity-based stereo evidence does **not** improve failure detection on this data,
   and the measured reason is that stereo matching fails on specular instruments
   (45.6% vs 65.2% left-right consistency) — independently corroborated by D7.

**Claimed, pending the section 3.2 test:**

7. Cross-view segmentation disagreement improves failure detection. Nobody has used a
   second physical camera this way (G1, G2) and nobody has used this dataset's
   calibration at all (G3).

**Not claimed:**

- That we beat any published mIoU. Protocols differ three ways (REFERENCES.md, protocol
  note) and no comparison is made without an identical protocol.
- That any correction raises mIoU materially. Measured: at most +0.0012.
- That the difficulty index predicts failure. Measured: Spearman ~0 out of fold.
- That the conformal guarantee holds on this test set. It does not (0.077 vs 0.05).
- That the test set measures generalisation to an unseen camera rig. It does not —
  edge case 6.

**If claim 7 fails**, the contribution is items 1-6: a verified evaluation, two
measurements nobody has made, two negative results with measured mechanisms, and a cost
result. That is a complete and honest paper, and it is why the design does not depend on
the novelty landing.
