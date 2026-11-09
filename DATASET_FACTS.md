# DATASET_FACTS.md — what is actually in the data and the environment

Written in Task 1. Every later task re-reads this file. Where it disagrees with
the project brief, **this file wins**. Everything below was measured on the real
files unless it says "not measured". Measurement scripts are reproducible
(`tools/bench_vram.py` is in the repo; the dataset scans were one-off scripts and
their raw outputs are summarised here).

Dataset root: `C:\Users\Chandaluri Vaishnavi\Downloads\cvrsfesp\endovis2018_cvdataset`
Environment: conda env `sitr`, Python from `...\anaconda3\envs\sitr\python.exe`,
torch 2.5.1+cu121, CUDA available. scipy 1.17.1, scikit-learn 1.9.1,
opencv 5.0.0, numpy 2.4.6.

---

## CONTRADICTIONS WITH THE BRIEF

| # | The brief / repo docs say | The data says | Consequence |
|---|---|---|---|
| 1 | GPU "under 8 GB" | RTX 3050 Ti Laptop, **4 GB** total, **3.21 GB free** (apps hold ~0.7 GB) | Everything sized for 3.2 GB. See section 1 |
| 2 | 4 test sequences x 250 frames; "at least one" test frame has no label | Test frames are 250 / **249** / 250 / 250 = 999. Labels are 250 / 249 / **249 / 249** = **997**. `seq_3/frame249` and `seq_4/frame249` have **no label** | Two unlabelled test frames, dropped. Official evaluator only walks the labels folder, so it never sees them either. All test numbers are over **997** frames |
| 3 | "Several labels.json files that disagree" | Release 4 has **no** `labels.json`. Release 1 lists 11 classes (no ultrasound-probe). Releases 2, 3 and the test file list 12. Class ids and colours agree everywhere. The only name conflict is `intestine` / `small-intestine`, one colour (124,155,5) | Class table = union. No colour disagreements to reconcile |
| 4 | Official `run.py` / `labels.json` consistent | `run.py` maps **clasper -> 3 and wrist -> 2**, but `labels.json` says clasper = 2, wrist = 3 | Harmless for the all-classes mean (GT and prediction are both remapped the same way). It swaps the *names* in the official per-class "X vs all" outputs. We report per-class IoU by name from our own table |
| 5 | Test set exercises all 12 classes | **Test has zero suction-instrument pixels** (0 of 997 frames). 11 active classes | Official metric simply skips an absent class per frame |
| 6 | Rare classes "appear in few frames" | Needle: **12 training frames** (1 in seq_1, 11 in seq_4) and 0.0026 % of pixels, but **95 test frames (9.5 %)**, all in test seq_1. Thread, clamps likewise occur in test only in seq_1 | Huge train/test shift on needle. Test per-class scores for needle, thread, clamps come from ONE sequence: expect high variance |
| 7 | README: "~2 % instrument pixels, ~98 % background" | Instrument-type classes are **17.7 %** of training pixels (shaft 9.1, clasper 4.7, wrist 2.1, thread 0.35, clamps 0.50, needle 0.003, suction 0.41, probe 0.52). Background-tissue is 45.3 %, anatomy 37.0 % | The README's class-imbalance reasoning is wrong for the 12-class task |
| 8 | Calibration "per sequence" | There are **7 distinct calibration files for 19 sequences**: they are per camera-rig session. Test seq_1-3 have the same intrinsics as train seq_9-12 (f 1051.7); test seq_4 matches train seq_13/15/16 (f 1081.8) | Load each sequence's own file (correct), but note the test set shares rigs with training sequences. Generalisation to an unseen rig is **not** what the test set measures |
| 9 | Calibration files are well-formed | Three files (`seq_13, 15, 16`) end with a **corrupted, mangled dict line**. `seq_14` has CRLF and no trailing newline. 8 of 19 omit `Camera-0-Alpha` | Parser must whitelist known keys, skip unparseable lines, default skew to 0 |
| 10 | "Translation is 98.9 % pure horizontal" | `|Tx| / |T|` = 0.9999 for every rig (|Ty|,|Tz| are 0.02-0.04 mm against 4.28-4.37 mm). Rotation 0.063-0.075 degrees | Same conclusion (essentially a rectified pair already); my measure gives 99.99 %. Not material |
| 11 | Temporal evidence is weak | **Confirmed, with a nuance.** Consecutive-frame GT instrument IoU is 0.30-0.45 (median 0.19-0.44) against 0.03-0.12 for random frame pairs. So consecutive frames are 3-10x more alike than chance, but still overlap less than half. See section 6 | Temporal propagation has a hard ceiling far below a decent single-frame prediction |
| 12 | HANDOFF.md: tau = 0.76. Brief: tau = 0.75 | Cannot be checked: **no release artefact is in the repo** (weights, caches and the stage2 report are git-ignored and were not provided) | "Not measured". Every Stage 1/2 number in the brief (mIoU 0.5842, AUC 0.825, ...) is unverified until re-produced |
| 13 | README test counts (27 / 19) | Repo suites: `test_pipeline` 30 before the v2 merge, **36 after**; `test_stages345` 20. Total **56** | README test counts are stale |
| 14 | 149 frames x 15 sequences | Confirmed: 2235 training frames, one image/label per frame, `seq_8` absent. (`.DS_Store` files inside `labels/` and `left_frames/` made seq_1, 3, 4 look like 150 files) | Junk filter must also exclude `.DS_Store`, not just `__MACOSX` and `._*` |
| 15 | `repairs/` = 7 corrected masks | Confirmed 7, and **they are large corrections** (section 4) | Must be applied |

Things the brief said that were **confirmed**: right_frames matches left_frames
1:1 for all 19 sequences; all frames 1280x1024 RGB; stereo extrinsics present
for every sequence; label masks are exact-colour, 4-channel (RGBA, alpha always
255), with **zero** colours outside the 12-colour palette; the official extra
colour remaps in `run.py` never fire on this data.

---

## 1. Environment and compute (measured)

* GPU: **NVIDIA GeForce RTX 3050 Ti Laptop**, 4096 MiB, driver 581.57, CUDA 13.0
  shown. Free before the model is built: **3.21 GB** (user figure ~3.3 GB).
  CPU i7-12700H 14 cores / 20 threads, RAM 15.7 GB, 180 GB free disk.
* Sleep: AC idle-sleep is "never" (0). **On battery it is 180 s.** Keep the laptop
  plugged in. Long jobs also call a keep-awake request; no system setting changed.
* Weights: SurgeNetXL (93 MB) and ImageNet CAFormer (105 MB) URLs both reachable
  (HTTP 200). SurgeNetXL downloaded and verified: **150 / 150 encoder tensors
  matched (100 %)**, CAFormer-S18 + FPN = 25.86 M parameters.
* **Windows trap:** when VRAM is full, CUDA on Windows does *not* raise an
  out-of-memory error; it silently spills to system RAM and becomes 2-8x slower.
  "Fits" is therefore judged by measured peak memory vs free memory, and
  `tools/bench_vram.py` stops when peak reserved memory exceeds free VRAM.

### VRAM and speed, CAFormer-S18 + FPN, 12 classes, AMP, channels_last (real train step incl. AdamW)

Headroom = 3.21 GB free minus peak *reserved* memory. The user's rule is >= 0.5 GB.

| Resolution | Checkpointing | Batch | Peak alloc GB | Peak reserved GB | Headroom GB | img/s | Fits (>=0.5) |
|---|---|---|---|---|---|---|---|
| 384x480 | ON | 1 / 2 / 3 / 4 / 5 | 0.81 / 1.26 / 1.70 / 2.15 / 2.58 | 0.92 / 1.42 / 1.88 / 2.69 / 3.00 | 2.29 / 1.79 / 1.33 / 0.52 / 0.21 | 2.3* / 5.1 / 7.5 / 7.6 / 9.6 | b<=4 |
| 448x576 | ON | 1 / 2 / 3 / 4 | 1.10 / 1.80 / 2.50 / 3.19 | 1.23 / 2.18 / 3.05 / 4.02 | 1.98 / 1.03 / 0.16 / spills | 2.59 / 5.23 / 6.3* / 2.4 | b<=2 |
| 448x576 | OFF | 1 / 2 | 1.74 / 3.10 | 1.83 / 3.24 | 1.38 / spills | 3.39 / - | b=1 |
| **512x640** | ON | 1 / 2 / 3 | 1.48 / 2.58 / 3.66 | 1.71 / 2.93 / 4.28 | 1.50 / 0.28 / spills | 2.49 / 3.98 / 1.6 | b=1 |
| **512x640** | **OFF** | 1 / 2 | 2.28 / 4.18 | 2.44 / 4.41 | **0.77** / spills | **3.58** / 0.5 | **b=1** |

`*` = from the short 6-step run, noisy (other apps share the GPU); the 20-step
runs (448x576 and 512x640 rows with two decimals) are the reliable ones.
**512x640 does fit** (the user expected it not to).

**Chosen: 512x640, batch 1, checkpointing OFF, gradient accumulation 8
(effective batch 8), 3.58 img/s, 0.77 GB headroom.** Fallback if memory ever
gets tight: checkpointing ON (1.71 GB, 2.49 img/s). Training scripts must set a
hard memory cap so an overflow raises instead of silently spilling.

### Time estimate (formula from the brief, 2235 frames)

Cross-fitting trains K fold models on (K-1)/K of the data plus one final model on
all of it. For K = 3: 3 x 1490 + 2235 = **6705 image-passes per epoch**.

| Setting | image-steps | hours = steps / (img/s x 3600) x 1.2 |
|---|---|---|
| 512x640, b1 no-ckpt (3.58 img/s), 3 folds, **25** epochs | 167,625 | **15.6 h** (brief's simpler formula 14.9 / 13.9 h) |
| same, 20 epochs | 134,100 | **12.5 h** |
| 448x576, ckpt b2 (5.23 img/s), 3 folds, 25 epochs | 167,625 | 10.7 h |
| 512x640, ckpt b1 (2.49 img/s), 3 folds, 25 epochs | 167,625 | 22.4 h |

25 epochs at 512x640 is just over the brief's ~14 h limit. Cut order is
epochs -> folds -> resolution: **DEV run = 3 folds, 20 epochs, 512x640 -> ~12.5 h**.
Note: the v2 cross-fitting recipe trains a *fixed* number of epochs on purpose
(no epoch is picked with held-out labels), so "early stopping ON" from the brief
conflicts with it; the fixed 20 is used and the loss curve is reported.
Inference passes (OOF + 997 test frames) add well under an hour. The same models
are reused for the final results; there is no separate FINAL training unless the
curves show the run had not converged. Run is resumable (checkpoint every epoch).

---

## 2. Structure

Releases (training, all `seq_*` folders have `left_frames/ right_frames/ labels/
camera_calibration.txt`):

| Release | Sequences |
|---|---|
| miccai_challenge_2018_release_1 | seq_1, 2, 3, 4 |
| miccai_challenge_release_2 | seq_5, 6, 7 |
| miccai_challenge_release_3 | seq_9, 10, 11, 12 |
| miccai_challenge_release_4 | seq_13, 14, 15, 16 |
| test_data/test_data/seq_N-<timestamp>-1-001/seq_N | test seq_1..4 |

`seq_8` is genuinely absent. **15 training + 4 test sequences.**
`__MACOSX/` shadow trees exist; `.DS_Store` also appears inside `labels/` and
`left_frames/` of seq_1, seq_3, seq_4.

Frames: train 149 per sequence (seq_1..seq_16 except 8) = **2235**; test
250 / 249 / 250 / 250 = **999 images, 997 labels**. left, right and label
filenames match 1:1 in every training sequence and in test seq_1, seq_2. Test
seq_3 and seq_4 lack `labels/frame249.png`.

Images: all 3234 left and all 3234 right are **1280x1024, 3-channel RGB, 8-bit
PNG**. Masks: all 3239 (2235 + 997 + 7 repairs) are **1024x1280x4 uint8** (RGBA,
alpha = 255 everywhere). Read masks with exact RGB lookup (OpenCV returns BGRA).

---

## 3. Labels and class statistics

Palette (identical in every `labels.json`; `intestine` == `small-intestine`):

| id | class | RGB |
|---|---|---|
| 0 | background-tissue | 0,0,0 |
| 1 | instrument-shaft | 0,255,0 |
| 2 | instrument-clasper | 0,255,255 |
| 3 | instrument-wrist | 125,255,12 |
| 4 | kidney-parenchyma | 255,55,0 |
| 5 | covered-kidney | 24,55,125 |
| 6 | thread | 187,155,25 |
| 7 | clamps | 0,255,125 |
| 8 | suturing-needle | 255,255,125 |
| 9 | suction-instrument | 123,15,175 |
| 10 | intestine | 124,155,5 |
| 11 | ultrasound-probe | 12,255,141 |

Mask scan (every mask): **12 distinct colours in training, 11 in test, 10 in
repairs; all are in the palette; unmapped pixels = 0.** Palette entries never
used: none in training; `suction-instrument` never in test.

### Pixel share and frame presence, TRAINING (2235 frames, repairs applied)

| id | class | pixel share % | frames containing | % frames |
|---|---|---|---|---|
| 0 | background-tissue | 45.3138 | 2235 | 100.0 |
| 1 | instrument-shaft | 9.0731 | 2077 | 92.9 |
| 2 | instrument-clasper | 4.6597 | 2194 | 98.2 |
| 3 | instrument-wrist | 2.1338 | 1645 | 73.6 |
| 4 | kidney-parenchyma | 17.2003 | 1396 | 62.5 |
| 5 | covered-kidney | 11.6864 | 1391 | 62.2 |
| 6 | thread | 0.3519 | 649 | 29.0 |
| 7 | clamps | 0.4999 | 372 | 16.6 |
| 8 | suturing-needle | **0.0026** | **12** | 0.5 |
| 9 | suction-instrument | 0.4122 | 269 | 12.0 |
| 10 | intestine | 8.1448 | 1015 | 45.4 |
| 11 | ultrasound-probe | 0.5216 | 156 | 7.0 |

### TEST (997 labelled frames)

| id | class | pixel share % | frames | % |
|---|---|---|---|---|
| 0 | background-tissue | 48.1413 | 997 | 100.0 |
| 1 | instrument-shaft | 8.7325 | 843 | 84.6 |
| 2 | instrument-clasper | 1.9518 | 875 | 87.8 |
| 3 | instrument-wrist | 2.1805 | 822 | 82.4 |
| 4 | kidney-parenchyma | 23.1381 | 949 | 95.2 |
| 5 | covered-kidney | 11.7059 | 484 | 48.5 |
| 6 | thread | 0.1439 | 102 | 10.2 |
| 7 | clamps | 0.0251 | 69 | 6.9 |
| 8 | suturing-needle | 0.0204 | 95 | 9.5 |
| 9 | suction-instrument | 0.0000 | **0** | 0 |
| 10 | intestine | 3.0737 | 225 | 22.6 |
| 11 | ultrasound-probe | 0.8866 | 152 | 15.2 |

Where the rare classes live (frames containing the class):

* **needle**: train seq_1 (1), seq_4 (11). Test seq_1 (95).
* **ultrasound-probe**: train seq_5 (20), seq_14 (136). Test seq_1 (57), seq_4 (95).
* **clamps**: train seq_1 (142), 2 (51), 4 (6), 5 (25), 7 (146), 9 (1), 12 (1). Test seq_1 only (69).
* **thread**: train seq_1, 2, 4, 5, 7, 15, 16. Test seq_1 only (102).
* **suction**: train seq_1, 3, 4, 5, 6, 7, 9. Test: none.
* covered-kidney is absent from train seq_1, 2, 4, 5, 7; kidney-parenchyma absent from seq_6, 9, 13, 14.

**Design constraints that follow:** (a) cross-fitting folds must be built so the
two needle sequences (seq_1, seq_4) and the two probe sequences (seq_5, seq_14)
are not all held out together, otherwise a fold model never sees the class;
(b) a frame subsample, if ever needed, must keep every frame containing a class
with < 700 frames; (c) class weights must be clipped (needle is 0.0026 % of pixels).

---

## 4. Repairs

`repairs/repairs/` holds **7** masks: `seq_1` frames 042, 043, 044, 073 and
`seq_4` frames 135, 137, 138. Every original exists; shapes match.

| Repair | pixels changed | % of frame | what changed |
|---|---|---|---|
| seq_1 frame042 | 79,044 | 6.0 | kidney -> clasper 62,783; kidney -> background 13,593 |
| seq_1 frame043 | 84,705 | 6.5 | kidney -> clasper 36,763; kidney -> shaft 36,598 |
| seq_1 frame044 | 86,254 | 6.6 | kidney -> shaft 42,803; kidney -> clasper 14,242 |
| seq_1 frame073 | 25,920 | 2.0 | kidney -> suction 17,391; kidney -> background 7,865 |
| seq_4 frame135 | 326,145 | **24.9** | background -> intestine 326,074 |
| seq_4 frame137 | 342,047 | **26.1** | background -> intestine 338,194; needle -> intestine 3,758 |
| seq_4 frame138 | 295,101 | **22.5** | background -> intestine 294,474 |

In seq_1 the original masks wrongly called instrument pixels "kidney"; in seq_4 a
quarter of the frame was wrongly called background. Ignoring the repairs would
train on, and evaluate against, wrong labels. They are applied as a
non-destructive override (the repo already does this). Only training frames are
repaired; the repair files exist for no test frame.

---

## 5. Calibration (critical)

Raw file, `train seq_1` (verbatim):

```
Width: 1280                                               // width of the images in pixels
Height: 1024                                              // height of the images in pixels
Camera-0-F: 1084.21 1084.05                               // left camera x,y focal dist in pixels
Camera-0-C: 580.02 506.79                                 // left camera x,y center in pixels
Camera-0-Alpha: 0.00000                                   // left camera skew
Camera-0-K: -0.00069 0.00195 0.00018 0.00000 0.00000      // left camera radial distortion
Camera-1-F: 1083.43 1083.22                               // right camera x,y focal dist in pixels
Camera-1-C: 680.91 505.81                                 // right camera x,y center in pixels
Camera-1-Alpha: 0.00000                                   // right camera skew
Camera-1-K: -0.00085 0.00245 0.00004 0.00000 0.00000      // right camera radial distortion
Extrinsic-Omega: -0.0002 -0.0011 -0.0000                  // left to right camera rotation
Extrinsic-T: -4.3499 0.0333 -0.0369                       // left to right camera position
```

**ARE THE STEREO EXTRINSICS PRESENT AND USABLE? Yes.** All 19 sequences have
`Extrinsic-Omega` (a Rodrigues vector, convert with `cv2.Rodrigues`) and
`Extrinsic-T` (millimetres). Left and right intrinsics (focal, principal point,
5 distortion coefficients) are also present.

| Rig (distinct file) | Sequences | f (px) | baseline mm | rotation deg | Camera-0-Alpha |
|---|---|---|---|---|---|
| A | train 1, 2, 3, 4, 6, 7 | 1084.2 | 4.350 | 0.064 | present |
| B | train 5 | 1056.3 | 4.348 | 0.063 | present |
| C | train 9, 10, 11, 12 | 1051.7 | 4.359 | 0.063 | **missing** |
| D | train 13, 15, 16 | 1081.8 | 4.369 | 0.068 | **missing** + corrupted trailing line |
| E | train 14 | 1068.4 | 4.278 | 0.075 | present; CRLF, no final newline |
| F | test 1, 2, 3 | 1051.7 | 4.359 | 0.063 | present |
| G | test 4 | 1081.8 | 4.369 | 0.068 | **missing** |

(7 distinct files; 19 sequences.) Principal points differ left vs right by about
100 px in x (e.g. 580 vs 681) while the rotation is only 0.064 degrees: the axes
are essentially parallel and the **sensors are offset**. The consequence is that
**raw (unrectified) disparity is f x B / Z minus about 100 px**, so far points
would need *negative* disparities. `cv2.stereoRectify` with zero-disparity
alignment removes that offset so that d = f x B / Z, and the 25-160 px range
then means Z of about 30-190 mm (f x B = 1084 x 4.35 = 4716 px.mm). SGBM must run
on rectified images, never on the raw pairs.

**Open question for Task 2:** the 5 distortion numbers are labelled "radial".
Seq_1 is `k1 k2 0.00018 0 0` and seq_13 is `k1 k2 0 0 k5`, which is consistent
with OpenCV order `(k1, k2, p1, p2, k3)`. The distortion is tiny (< 1 px at the
image edge) so the choice barely matters; Task 2 will test both orders by
measuring vertical disparity.

---

## 6. Temporal measurement (tests the "temporal evidence is weak" claim)

Sequences: train seq_2, seq_9, seq_14 and test seq_1, seq_4. Consecutive frames
(1 Hz) vs random pairs from the same sequence at least 20 frames apart.
"Instrument" = shaft, clasper, wrist, thread, clamps, needle, suction, probe
(the repo's `is_instrument_name`). IoU is between ground-truth masks, so it is
the best any "copy the previous mask" method could possibly do.

| Sequence | mean abs pixel diff, consecutive | random pairs | class-mean IoU, consecutive | random | instrument IoU, consecutive (median) | random | % consecutive pairs with instrument IoU < 0.5 |
|---|---|---|---|---|---|---|---|
| train seq_2 | 23.3 | 45.0 | 0.411 | 0.068 | 0.447 (0.435) | 0.117 | 62 % |
| train seq_9 | 22.2 | 51.2 | 0.481 | 0.089 | 0.435 (0.354) | 0.109 | 61 % |
| train seq_14 | 25.5 | 45.5 | 0.394 | 0.070 | 0.393 (0.382) | 0.122 | 64 % |
| test seq_1 | 21.6 | 42.8 | 0.382 | 0.078 | 0.381 (0.376) | 0.107 | 66 % |
| test seq_4 | 23.6 | 42.0 | 0.447 | 0.060 | 0.299 (0.193) | 0.031 | 77 % |

**Verdict: supports the brief, with a nuance.** There is real temporal
information (consecutive masks overlap 3-10x more than random ones), but a
perfectly aligned ground-truth mask from the previous frame would score only
**0.30-0.45 instrument IoU** on the current frame, and 60-77 % of consecutive
pairs overlap by less than half. Any recovery that relies on propagating
the previous frame is capped well below what a decent single-frame model
achieves. This is evidence about the *ceiling*; whether propagation helps a
specific failed frame is Task 3.

---

## 7. Image properties (difficulty-index inputs)

Per-frame statistics on the same 5 sequences (min / mean / max):

| Sequence | brightness (0-255) | specular fraction (V>235 & S<40) | Laplacian variance (blur) |
|---|---|---|---|
| train seq_2 | 80 / 95 / 109 | 0.1 % / 1.2 % / 6.7 % | 52 / **461** / 3621 |
| train seq_9 | 70 / 104 / 136 | 0.2 % / 2.2 % / 9.2 % | 5 / **39** / 76 |
| train seq_14 | 78 / 93 / 106 | 0.2 % / 1.2 % / 6.9 % | 85 / **315** / 996 |
| test seq_1 | 61 / 99 / 123 | 0.3 % / 1.8 % / **18.6 %** | 14 / **49** / 83 |
| test seq_4 | 65 / 93 / 121 | 0.1 % / 1.2 % / 8.4 % | 12 / **33** / 77 |

The Laplacian-variance blur measure differs by **about 10x between sequences**
(roughly 40 vs 460 on average), so a raw blur number cannot be thresholded
across sequences. The difficulty index must normalise blur within each
sequence (or use a relative measure). Specular area is small on average (about
1-2 %) but spikes to 18.6 % on single frames. Exposure is moderate everywhere
(no extreme over/under-exposure sequence among these five).

---

## 8. The OFFICIAL evaluation protocol (`test_data/test_data/run.py`, `utils.py`)

How the official score is computed, quoting the code:

1. Read each submission mask and GT mask as **colour PNGs** and convert colours
   to class ids (`change_frame`, run.py:21-76). Evaluation is at the native
   1280x1024; the submission must be a colour PNG per labelled frame at
   `<submission>/<seq_name>/frameNNN.png`.
2. Only frames listed in the **labels folder** are scored
   (`for frame in sorted(os.listdir(dataset_full_path))`, run.py:174). That is why
   the two unlabelled test frames never count.
3. Per frame, for **each class except background**:
   `if val.classid == 0: continue` (utils.py:172).
   `IoU = TP / (TP + FP + FN)` (utils.py:141-147, 175-180).
4. A class counts for a frame **only if it is present in the GROUND TRUTH**:
   `if (ground_truth_not_overlap + overlap) == 0: return None` (utils.py:143), and
   `# only count IoU for mean for classes actually in this frame` (utils.py:182).
   A class the model hallucinates but the GT lacks is **not penalised at all** by
   this metric. (This settles the "present in GT" vs "GT or prediction" ambiguity
   the repo README flagged: the official code is GT-only.)
5. Frame score = **mean over the classes present** (`np.mean(ious)`, utils.py:193);
   a frame with no non-background class scores `None` and is skipped.
6. Sequence score = mean of its frame scores. Overall score = **average of the
   four sequence scores weighted by their number of scored frames**
   (utils.py:63-76), i.e. a pooled frame mean.
7. The official per-class "X vs all" outputs use `labels.json` ids, which have
   clasper and wrist swapped relative to `run.py` (contradiction 4).
8. `utils.py` imports `skimage`, which is not installed; the official script
   cannot run as shipped here. We re-implement the same arithmetic and check it
   against the quoted lines (Task 6).

Because only classes in the GT count, an extra false-positive class is invisible
to this metric. Our harmful-correction-rate metric therefore uses its own
definition (IoU change against GT), reported alongside, not replacing, the
official score.

---

## 9. "Exploit every part of the dataset" — how each component is used

| Component | Real contents (measured) | How it will be used / why not |
|---|---|---|
| `left_frames/` | 3234 RGB 1280x1024 | Segmenter input; the only labelled view |
| `right_frames/` | 3234, 1:1 with left, same size | (a) disparity and the stereo evidence map; (b) occlusion + specular detection (left-right disagreement); (c) cross-view supervision: only if Task 2 shows clean rectification. **Cost:** a second forward pass for (c) |
| `labels/` | RGBA colour PNGs, 12 colours, no strays | Exact RGB lookup to ids 0-11; nearest-neighbour only |
| `camera_calibration.txt` | 7 distinct rigs, full stereo extrinsics, quirks (section 5) | `cv2.stereoRectify` per sequence with a tolerant parser; rig id also used as a grouping variable for validation (proposal P1) |
| `repairs/` | 7 large corrections | Applied as override for training and CV |
| Sequence/frame order | 149 / ~250 frames at 1 Hz | Memory buffer for the temporal action (A2); **whole-sequence** splits |
| Release folders | labels.json in 3 of 4 releases, 2 schemas | Union class table; missing release-4 file is not an issue |
| `test_data/run.py`, `utils.py`, `labels.json` | Official protocol (section 8) | Evaluator re-implemented to match; the final table reports this exact metric |
| Per-class pixel stats | section 3 | Clipped class weights; fold construction (section 3 constraints); the rare-class list |
| Image properties | section 7 | Difficulty index inputs (specular fraction, within-sequence-normalised blur, brightness, stereo disagreement) |

Not used: the test images' pixels for any training or tuning, and no test label
is touched before Task 6 (the brief forbids it).

---

## 10. Additional uses I propose (not implemented)

| # | Idea | Cost | What it would prove |
|---|---|---|---|
| P1 | **Group the cross-fitting folds by camera rig**, not just by sequence. Test seq_1-3 share a rig with train seq_9-12, so a sequence-level split can still leak rig/procedure-style information | Nothing computational. Fewer valid fold layouts | Whether gains hold when the validation rig is unseen. Honest generalisation claim |
| P2 | **Left-right prediction agreement as a free "2-member ensemble"**: segment the right view too, warp it to the left with the disparity, and use the pairwise Dice as a failure score. This is the stereo analogue of the strongest baseline in the failure-detection benchmark (ensemble + pairwise Dice) | One extra forward pass per frame (about +100 % segmentation compute for the frames where it is used; can be gated) | Whether stereo agreement beats single-model confidence at predicting failure, at half the cost of a real ensemble |
| P3 | **Depth-edge boundary snapping** (action A3): disparity discontinuities coincide with instrument/tissue borders; use them to refine mask boundaries and thin structures | Cheap, classical (reuses cached disparity) | Whether geometry improves boundary F1 and thin-class clDice without a learned model |
| P4 | **Metric-size plausibility**: with Z = f B / d and the known instrument shaft width, flag predicted shafts whose 3-D width is implausible | Needs a reliable disparity on instruments (Task 2 measures this) | A geometry-only failure cue independent of the network's confidence |
| P5 | **Right view as extra training data** (warp left labels to right where the occlusion mask is clean) | Warping error near occlusions; double the data loading | Whether free stereo supervision helps the 12-sample needle class. Risky, optional |
| P6 | **Quantify label noise from the repairs**: the 7 known errors changed 2-26 % of a frame. Flag other frames whose cross-fitted loss is extreme | Free once OOF predictions exist | How much of the "failure" in the gate is label noise, not model error |

Not worth doing: using the unlabelled test frames for self-training or
test-time adaptation (violates the "no test data in training" rule); monocular
depth (real stereo exists).
