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

### Error #14 (Task 3, test run) - panel writer crashed after the numbers were printed
Cause: cv2.putText got a non-contiguous array (a sliced, channel-reversed view) in save_panels, so results/task3/measure_recovery.json was never written; the measured numbers are in the console log, kept as results/task3/measure_recovery_test_log.txt.
Fix: np.ascontiguousarray on the tiles. The already-running out-of-fold job loaded the old code and will hit the same crash at the very end; its log is then the record.
Also reworded interpret()'s top verdict: it said "premise holds" even when 99% of frames were unchanged. Numbers and thresholds untouched.

### Task 3 result on the 4 official test sequences (measurement, not tuned)
Selective (routed 207/997 frames, 20.8%) and always-correct (880/997) both give a statistically positive but tiny change.
Always mode, instrument IoU: 0.6105 -> 0.6140 (+0.0035, 95% CI +0.0002..+0.0081). Official-metric IoU: 0.6397 -> 0.6397 (+0.0001, CI -0.0002..+0.0002), i.e. no change.
Only 0.7% of routed frames improved by more than the margin and 0.3% got worse; 99% were unchanged at that margin (strict sign: 16.9% better, 5.3% worse, 77.7% exactly zero).
Plain reading: temporal correction rarely matters and, in the official metric, does nothing measurable on test. Not evidence of a large gain.

### Task 3 out-of-fold (15 training sequences, per-fold models; measurement, not tuned)
Always-correct mode: instrument IoU 0.6597 -> 0.6627 (+0.0030, CI +0.0010..+0.0055); official IoU 0.6772 -> 0.6774 (+0.0003, CI +0.0001..+0.0004). 1.8% of routed frames improved past the margin, 0.7% worse, 97.5% unchanged (strict sign: 27.7% better, 7.3% worse, 65% exactly 0).
Same picture as test: a real but very small gain; no harm on average. The out-of-fold job crashed in the panel writer at the end as predicted (error #14); numbers kept in results/task3_oof/measure_recovery_oof_log.txt, no JSON.
Selective mode (OOF, routed 623/2235 = 27.9%): instrument IoU 0.5289 -> 0.5374 (+0.0084, CI +0.0028..+0.0161); official IoU 0.6227 -> 0.6234 (+0.0007, CI +0.0003..+0.0011); 5.1% improved / 1.9% worse past the margin.

### Task 5 result: candidate actions + benefit selector (measurement, not tuned; margin 0.01 fixed in advance)
Held-out training (OOF, 2235 frames, gate routes 49.7%): selector activates 26.9%, mean dIoU over all frames +0.0054, harmful 2.0% of corrected; oracle +0.0128; always-A3 (routed) +0.0098 harmful 1.1%; always-A2 +0.0024; always-A1 -0.0000; fuse +0.0024.
Official test (997 frames, routed 30.4%): selector +0.0096, harmful 1.2%, activation 25.9%; oracle +0.0215; always-A3 +0.0189 harmful 1.0%; always-A2 +0.0031; always-A1 +0.0001; fuse +0.0022; fuse+refine on every frame +0.0261.
STOP check: oracle has real headroom over always-A2 (+0.0105 OOF, +0.0184 test), so the selector was worth building.
Plain reading: the selector does NOT beat the simplest baseline. Always applying the class-conditioned morphological refinement (A3, no stereo, no temporal) gets about twice the mean gain of the selector on both sets, with similar harm. Stereo evidence (A1) gives essentially nothing. Selector predictions are weak for A1/A3 (Pearson -0.03 / +0.14), moderate for A2/FUSE (+0.33 / +0.46). The selector's only edge is abstaining on frames where nothing helps.
Compute: selector ~32 ms/frame CPU on test vs 300 ms for fuse+refine on every frame.

### Task 6 result: final M1-M6 on the official test set (997 frames, full resolution, official metric; measurement, not tuned)
mIoU: M1 0.6128, M2 0.6137, M3 0.6129, M4 0.6138, M5 0.6141, M6 0.6128. Our scorer matches the official run.py (|diff| 1.1e-16).
Harmful corrections, % of ALL frames: M4 1.20 vs M6 0.80 (M6 better). % of CORRECTED frames: M4 1.20 vs M6 3.10 (M6 worse). STOP condition (M6 must beat M4 on harmful rate) is met under the first definition and not the second; both recorded.
M6 activation 25.9% vs 99.9%; mean latency 792 ms vs 2201 ms (composed from measured components).
Plain reading: no correction moves mIoU meaningfully; the supportable result is the compute saving, not an accuracy gain. Written up in results/FINDINGS.md.
No errors in this task. (Environment: final_eval's scoring is CPU by design; GPU used only for the latency benchmark.)

### Task 7 progress: results website (from the user's PrebuiltUI Next.js template)
User asked for the supplied template to be modified into a multi-page results/use-case site with buttons. Built as web/ (copy of the template; original left untouched): pages Home, Problem, Results, Stereo, Selector, Limits, Run it. Static export, no analytics, aggregates only (web/data/results.json written by tools/export_web_data.py from results/*.json). Real-estate content, stock photos and template logo removed.
Error #15: first `next build` failed type-check (highlight prop got a number, expects number[]); fixed the call. Error #16: my first Run-page commands used flags from memory that do not exist (e.g. --work on run_v2); corrected against each tool's argparse before building.
Corrections found while checking the pages against the data: M5 gain is +0.0012 not +0.0013; self-calibration was applied to one test sequence (seq_3, 1.3 px implied, negligible effect), so "test needs no correction" was reworded.
Not done: Streamlit demo (Streamlit is installed in the sitr env; the website replaces it as the presentation layer unless asked), Vercel deploy (needs the user to log in: `vercel login`, then `npx vercel --prod` in web/).

### Error #17 (licensing): web/ was pushed to GitHub
The supplied PrebuiltUI template license (section 3) forbids publishing the template source files publicly, including on GitHub. I pushed web/ (template-derived) in the previous commit before reading the license. Fix applied: web/ untracked and git-ignored; only the built site is to be deployed. The earlier commit still exists in branch history on origin; removing it needs a history rewrite and force-push, which is the user's call (and matters only if the repository is public).

### Task 7 deployed
Vercel CLI login completed by the user (device approval); the built static output (web/out only, no template source) was deployed to production: https://stereo-gated-results.vercel.app (project stereo-gated-results). Home and /results/ verified HTTP 200. The localhost error the user saw was because my preview server had been stopped; `npm run dev` in web/ serves http://localhost:3000.
Not done by choice: Streamlit demo (website replaces it); removing the template-derived commit 10fa909 from GitHub history (needs force-push, awaiting the user's decision).

### Website redesign and results bundle
Redesigned the website as a plain research-page style (serif headings, one accent colour, text tabs, thin-rule tables; removed gradient hero, section-label squares, coloured callout boxes, icon cards, scroll animations and repeated copy) and redeployed to https://stereo-gated-results.vercel.app (verified 200 on / and /stereo/).
All results collected into C:\Users\Chandaluri Vaishnavi\Downloads\cvrsfesp\results (README.md index, task1-6 outputs, logs, training histories, FINDINGS/RUNLOG/DATASET_FACTS, website data and built site). Weights, caches and the 2000-file official-check scratch folder were left out.

### Website: back to the template layout, new colours and wording
At the user's request the earlier layout (dark hero, cards, text pills, scroll reveals) was restored and only colours and text changed. Colours unified to a slate base with one teal accent (removed the sky/emerald/rose/amber/violet mix, multi-colour model bars and coloured callouts). Home is now framed as a use-case page (live overlays, review queues, limited hardware, deciding whether stereo is worth it) with the measured results next to it; other pages got plainer headings. The Reproduce page no longer lists website build steps because the site source is not in the repository. Redeployed to https://stereo-gated-results.vercel.app (200 on /, /problem/, /results/); results bundle website/site_build refreshed.

### Website restructured as the SurgiVision platform
Per the user's concept (one platform, six workspaces): Home is a workspace picker; new pages SurgiReview (frame-by-frame viewer from real per-frame reliability, selector choice and IoU before/after for the 997 test frames, CSV report download), MaskCheck (per-class/per-sequence IoU by model), SurgiLearn (class explorer and a quiz built from the results), FrameGuard (flagging, correction mix, processing cost), InstrumentView (instrument categories, thin-structure clDice, hard cases) and SurgiBench (hub linking the existing Method, Results, Stereo, Selector, Limits, Reproduce pages). tools/export_web_data.py now also exports per-frame numbers (web/data/results.json); no images.
Decisions: no EndoVis frame images are published (original brief: no patient frames without approval; EndoVis licence), so the viewers show numbers only. Dice is not shown because it was never computed (official metric is IoU). Supabase not used: everything is precomputed aggregates, so no backend is needed; a demo-request form would be the only feature needing one and was left out rather than faked. Texts reworded to plain wording; claims checked against the data (selector mix A3 194 / A2 40 / fusion 24 / stereo-only 0). Redeployed; all 8 routes checked HTTP 200.
