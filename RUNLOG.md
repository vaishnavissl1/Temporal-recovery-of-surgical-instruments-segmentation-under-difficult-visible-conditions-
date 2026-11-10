# RUNLOG — errors, fixes and decisions (append-only)

Rule being followed: a technical error is diagnosed, fixed and logged here. A
measurement is never an error — it is recorded as the result, whatever it is.

## Task 1

### Errors hit and how each was fixed
| # | What happened | Cause | Fix |
|---|---|---|---|
| 1 | `tests.test_stages345`: 5 of 20 failed, `No module named 'scipy'` | `scipy` is listed in `requirements.txt` but was not installed in the `sitr` env | `python -m pip install scipy scikit-learn` in the `sitr` env only. 20/20 pass |
| 2 | `conda` not found on PATH in the tool shell | conda is not initialised for non-interactive shells | Use the env interpreter by full path: `C:\Users\Chandaluri Vaishnavi\anaconda3\envs\sitr\python.exe` |
| 3 | Structure scan crashed with `IndexError` | a release folder contains `labels.json`, which is not a `seq_*` folder | filter to `seq_*` directories before sorting (my scan script only) |
| 4 | Calibration parse crashed with `ValueError: could not convert string to float: "{'timestamp':"` | `seq_13/15/16` `camera_calibration.txt` end with a corrupted, mangled dict line | parser now accepts only known keys and skips unparseable lines. **The production parser must do the same** |
| 5 | A 448x576 batch-4 benchmark did not raise OOM, it just ran ~3x slower | Windows WDDM silently spills CUDA memory into system RAM when VRAM is full | "fits" is judged by measured peak memory, not by the absence of an OOM error. Training scripts must enforce a hard cap (see decisions) |

### Decisions
- Merged the v2 zip into branch `stereo-gated` by 3-way merge against base commit `643f9a6`.
  The teammate's `align.py`, `__init__.py`, `identity.py`, `recovery.py`, `tests/test_stages345.py`,
  `tools/run_stages345.py` are untouched. `requirements.txt` was a real conflict: kept both lines
  (scipy from the teammate, scikit-learn from v2).
- Tests: before merge 30 + 20 = 50 pass. After merge 36 + 20 = 56 pass.

### Task 1 results (see DATASET_FACTS.md for everything)
- Benchmark (CAFormer-S18+FPN, AMP, 4 GB GPU, 3.21 GB free): **512x640 fits**, batch 1,
  checkpointing OFF: 2.44 GB reserved, 0.77 GB headroom, 3.58 img/s. Chosen: 512x640,
  batch 1 x accumulation 8. DEV estimate 15.6 h at 25 epochs, **12.5 h at 20 epochs** (chosen).
- SurgeNetXL weights downloaded, 150/150 encoder tensors verified.
- 15 contradictions with the brief recorded (DATASET_FACTS.md, top).
- Raw outputs: results/task1/*.json. Scan scripts: tools/task1_scans/ (one-off, take the dataset root as argv).
- Sample images (left | right | overlay) written to results/task1_samples/ and NOT committed
  (PNG is git-ignored and dataset frames should not go to GitHub).

### Pending fixes noted during Task 1 (done in later tasks)
- tools/check_backbone.py judges "fits" by catching OOM only, which is wrong on Windows (silent spill).
  Training must set a hard memory cap and prefer measured peak reserved vs free VRAM.
- Production calibration parser must ignore unknown/corrupt lines and default Camera-0-Alpha to 0.
- Junk filter must exclude `.DS_Store` as well as `__MACOSX` and `._*`.

## Task 2 (stereo go/no-go) and the training launch

### Errors hit and how each was fixed
| # | What happened | Cause | Fix |
|---|---|---|---|
| 6 | New `tests/test_stereo.py`: 3 of 15 failed on first run | All three were **wrong assumptions in my new tests, not module bugs**: (a) I assumed the rectified focal stays > 1000 px, but `alpha=1` zooms out (focal ~998 px); (b) I assumed rows map to themselves, but the zoom rescales rows, the real property is that left and right maps read the SAME source row; (c) my synthetic pair had the shift sign reversed | Corrected what each test asserts to the true property, with comments. Module code unchanged. 15/15 pass |
| 7 | `make_folds` (the repo default) would train fold 2 with **zero ultrasound-probe frames** (both probe sequences held out together) | It balances only by frame count | Added `tools/make_folds_balanced.py`: exhaustive search over all 126,126 equal 3-fold splits, maximising the worst-case rare-class training coverage; result saved as `work_v2/folds.json` and `results/task1/folds_k3_class_aware.json`. Probe min 20/156 frames (was 0); needle min 1/12 (unavoidable: only seq_1 and seq_4 contain needle) |

### Decisions
- Training launched in the background (resumable): `tools.run_v2 --folds 3 --epochs 20 --batch-size 1
  --accum-steps 8 --gpu-mem-cap-gb 3.0 --workers 6 --arch caformer_surgenet`, output in `work_v2/`.
  Added `--gpu-mem-cap-gb` (hard cap so a Windows memory spill raises instead of silently slowing)
  and a per-process keep-awake request (no system power setting changed).
- Patched `tools/check_backbone.py` so it cannot report a spilling batch size as fitting.
- SGBM tuning criterion was fixed before any result was seen: maximise the fraction of pixels that pass
  the left-right check AND have a photometric residual < 10 gray levels; tuned on 5 training frames that
  are not among the 20 evaluation frames. Test frames are never used for tuning.
- Ground truth is used in Task 2 only to split pixels into instrument vs tissue, on TRAINING frames only.

### Task 2 results (details in results/task2/STEREO_REPORT.md)
- **Calibrated stereo is viable; no stop condition tripped.** Vertical error calibration-only 2.64 px (median 0.68), 4/20 pairs > 3 px;
  with label-free self-calibration 1.21 px, 1/20. LR-consistent 64.1% -> 69.3%. Instruments 54.3% (tissue 71.1%).
- Measured finding: shipped per-rig calibration leaves a 3-14 px tilt on training seq_2/4/5/7 (roll ~0.3-0.8 deg). Test sequences are fine.
- Convention chosen by data: OpenCV distortion order, rotation as given (the differences between conventions are small: 2.64-2.78 px).
- SGBM chosen by the pre-registered criterion: block 15, uniqueness 0. The optimum sits on the grid edge twice; that is disclosed, and a
  diagnostic with a rule declared in advance (results/task2/sgbm_diagnostic.txt) did not overturn it.

### More errors
| # | What happened | Cause | Fix |
|---|---|---|---|
| 8 | The first stereo-cache launch exited immediately with "unrecognized arguments" | `Start-Process` does not quote an argument containing a space (the dataset path has one) | Relaunched with the path wrapped in its own quotes; verified both processes were running before moving on |
| 9 | A difficulty-index test failed ("equal weights") | Test bug: each cue got its own random reference sample, so ranks differed by sampling noise | Test now uses the same reference sample for every cue; module unchanged |
| 10 | Stereo report had garbled characters and broken tables | File written in the Windows default encoding; `|dy|` contains pipe characters | `encoding="utf-8"`, wrote "abs(dy)" |

### Measurement findings recorded as results (not errors)
- The v2 code's "challenge IoU" includes the background class and is NOT the official metric (DATASET_FACTS contradiction 17).
- Task 2's SGBM criterion cannot see over-smoothing; disclosed, tested, not overturned.

## Task 4 (stereo evidence, difficulty index, gate comparison) — in progress

### Done
- Stereo evidence cache for ALL 3232 labelled frames (2235 train + 997 test), 512x640 original-left grid, **1.23 GB** (limit was 20 GB),
  45 min train + 26 min test in parallel on CPU while the GPU trained. Per-sequence self-calibration stored in `stereo_meta.json`.
- Cross-fitting finished (3 class-aware folds x 20 epochs, ~160 min each). Out-of-fold predictions cached for all 15 training sequences.
- Stereo indicators added to the OOF caches (`tools/add_stereo_indicators.py`). Frame names/counts are verified, not assumed.

### Measured result (not an error): the difficulty index does NOT track segmentation error
On 2235 held-out (out-of-fold) frames: index vs (1 - instrument IoU) Pearson -0.017, Spearman -0.029; within-sequence Spearman +0.036
(95% CI -0.057..+0.121, 11/15 sequences positive). Quartiles are not monotone: the hardest quartile has the LOWEST bad-frame rate (0.197 vs
0.283-0.346). Strongest pooled cue is haze but in the WRONG direction (-0.46) and ~0 within sequences, i.e. a sequence effect, not a frame effect.
The cue directions were fixed a priori and nothing was re-fitted. Consequence: novelty claim 6 ("gains concentrate on hard frames, measured by
an objective index") is **not supported on this data**. Full output: results/task4_difficulty_oof.log, results/task4/difficulty_report.json.

### Errors / fixes
| # | What happened | Cause | Fix |
|---|---|---|---|
| 11 | 2 existing tests failed after adding stereo indicator sets | the shared test cache had no stereo fields, and the tests loop over every indicator set | gave the test cache stereo fields; assertions unchanged. 36/36 pass |
| 12 | `python3` not found in the Bash tool | only the conda env's `python.exe` exists | used the Edit tool / the env interpreter |
| 13 | the training log showed nothing for hours | child process stdout is block-buffered when piped | progress is read from `work_v2/crossfit/*/stage1_history.json` instead; no effect on results |

### Statistical note
HistGradientBoosting without subsampling is deterministic, so "run several seeds" would show zero variation. Differences between indicator sets
are therefore tested with a paired cluster bootstrap over SEQUENCES (tools/compare_stereo.py).

### Task 4 result: stereo indicators do NOT beat temporal ones (measured, recorded as the result)
Cross-fitted CV on 15 held-out sequences (GBM, K=5, tau_train=0.5, same for every set), paired bootstrap over sequences:
| set | AUC | AURC (floor 0.211, random 0.375) | Pearson | MSE |
|---|---|---|---|---|
| paper4 | 0.745 | 0.288 | 0.512 | 0.0411 |
| extended (9) | 0.770 | 0.274 | 0.573 | 0.0324 |
| extended_stereo (9+3) | 0.773 | 0.272 | 0.592 | 0.0311 |
| nontemporal (5) | 0.749 | 0.279 | 0.576 | 0.0299 |
| nontemporal_stereo (5+3) | 0.743 | 0.279 | 0.569 | 0.0297 |
| stereo3 alone | **0.432** | 0.376 | 0.060 | 0.0510 |
- 9+3 vs 9: dAUC +0.005 [-0.020, +0.037] -> TIED. nontemporal+stereo vs nontemporal: dAUC -0.004 [-0.033, +0.028] -> TIED.
- stereo3 alone LOSES to nontemporal (dAUC -0.30, interval excludes 0) and is below chance.
- Individual stereo indicators: |Pearson| <= 0.095 with IoU. Leave-one-out AUC drops: stereo_disagreement +0.003, occluded +0.007, confidence +0.001.
- CONTRADICTION WITH THE BRIEF: temporal agreement is NOT weak here. T_temporal Pearson +0.49 (AUC 0.84; within-sequence +0.33); its
  removal costs the most AUC of any indicator (-0.038). The brief's "+0.05 to +0.15" came from the earlier ResNet-UNet/256x320 model.
  extended beats nontemporal on AURC (-0.0056, interval excludes 0): temporal evidence adds a small real amount.
- Consequence: the project's central argument ("temporal evidence is weak, stereo replaces it") is NOT supported on this data and model.
  Novelty claims 1-2 (stereo as evidence) stand only as method contributions, not as measured gains for failure detection.

### Task 4 difficulty index on the official TEST sequences (997 frames), recorded as the result
Index vs (1 - instrument IoU): Pearson +0.180, Spearman +0.081, within-sequence Spearman +0.083 (95% CI -0.145..+0.312; 2/4 sequences positive).
Quartiles: bad-frame rate 0.249 / 0.201 / 0.349 / 0.392 (not monotone; Q4 highest, Q2 lowest). On out-of-fold training frames it was -0.03.
Verdict: **not validated.** Sign differs between the two evaluations and the within-sequence interval includes 0. Do not claim it.

### Task 4 on the OFFICIAL TEST sequences (gate fit on all out-of-fold frames), recorded as the result
AUC: paper4 0.846, extended 0.845, extended+stereo 0.846, nontemporal 0.847, nontemporal+stereo 0.838, stereo3 alone 0.638.
- 9+3 vs 9: dAUC +0.004 [-0.007, +0.033] TIED. nontemporal+stereo vs nontemporal: dAUC -0.012 [-0.025, -0.004] LOSES.
- stereo3 alone vs nontemporal: dAUC -0.187, LOSES. (4 test sequences: coarse intervals.)
Consistent with the out-of-fold result: **stereo indicators do not improve failure detection; confidence/shape indicators carry the signal.**
