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


## FINAL_PROMPT tasks (A onward)

### Error #18 (Task A): the temporal action was always built with no motion compensation
tools/make_action_data.py hardcoded build_align("identity") and had no --aligner flag, so every A2 candidate in the action data behind Tasks 5 and 6 was the previous mask pasted on unmoved. Fix: added --aligner {identity,farneback,region} (instrument_ids passed through for region, as run_stages345.py does) and an aligner.json written into each output directory.
Decisions: (1) identity was not regenerated; the new code with --aligner identity reproduced work_v2/actions for test seq_1 with all 35 stored arrays equal (timing columns aside), so the existing directory is the identity data (saves about two hours, same result). (2) Farneback and region were regenerated out of fold (15 training sequences, fold models) and on the 4 test sequences, each into its own directory (work_v2/actions_farneback, work_v2/actions_region). (3) Identity stays the system of record because it is best out of fold. Cost: Farneback about 1.5 h, region about 6 h (up to 3600 s for one sequence).
Not an error: training seq_7 prints A2 mean as nan under all three aligners because its temporal memory is empty for the whole sequence.
Process notes: the Bash tool reported "completed" for background launchers while the Python jobs were still running; I checked the status files before trusting any completion. The selector was re-run on the identity data and reproduces results/task5 exactly (deterministic), so differences under the other aligners are real effects of the data.

### Task A result (results/taskA/SUMMARY.md)
A2 on flagged frames, out of fold: identity +0.0047, Farneback +0.0030, region +0.0007; unchanged share 96.0 / 96.0 / 96.7%. Test: +0.0102 / +0.0067 / +0.0005; unchanged 97.0 / 97.4 / 98.0%. Oracle on flagged test frames +0.0707 / +0.0706 / +0.0641. Paired differences to identity out of fold (sequence-clustered): Farneback -0.0017 [-0.0072, +0.0002]; region -0.0041 [-0.0115, -0.0002]. Answer: the null temporal result was NOT caused by the bug; neither aligner helped and region was worse.
Stop rule, as written before any number existed (all out-of-fold intervals for the difference to identity must contain zero): applied literally it does not report "same answer", because region's A2 interval excludes zero. It excludes zero on the harmful side. The rule is recorded as written (decision_oof.identity_was_not_the_cause = false) and the interpretation (the unaligned candidate was not hiding a benefit) is stated separately, not substituted.
Selector under the other aligners: test flagged-frame gain +0.0412 (Farneback) and +0.0613 (region) against +0.0315 (identity), but out of fold +0.0091, +0.0098 and +0.0109 respectively. Not an improvement: A2 and fusion gain predictions became noise (out-of-fold Pearson for A2 0.331 / 0.014 / -0.069) and the selector shifts to the morphological refinement, whose fixed-policy gain is +0.0621 for all three.

### Published numbers: what changed, and by how much (Task A and B)
Official mIoU for M1 to M6: no change beyond 0.0002 under either aligner (M5 0.6141 vs 0.6140 / 0.6142); the published Task 6 table stands and the identity data is unchanged.
Corrections to earlier wording, not to measurements:
- FINDINGS (Task 5 paragraph) called +0.0189 for always-A3 "on routed frames". It is an all-frame figure; the flagged-frame figure is +0.0621. Likewise the oracle +0.0215 vs always-A2 +0.0031 and the selector +0.0096 are all-frame figures (flagged-frame: +0.0707, +0.0102, +0.0315).
- The brief quotes +0.1040 (oracle) and +0.0370 (selector) as flagged-frame gains. They are per corrected frame (206 and 258 frames respectively); on the 303 flagged frames they are +0.0707 and +0.0315. Both are in results/taskB/policy_table.md with their denominators.
- The statement that the small global number is explained by dilution is incomplete. See below.

### Task B finding (not an error): the +0.062 is hallucination removal, invisible to the official metric
Flagged frames are 303 of 997 (30.4%). Always-A3 on them: instrument IoU +0.0621 when a class counts if it is in the ground truth or the prediction (the definition the action data labels with, present="union"), +0.0001 when it counts only if in the ground truth, and official frame IoU -0.0003. A3 removed 74 predicted instrument classes absent from the ground truth and added none; 191 flagged frames had at least one before, 168 after. Dilution by the flagged share explains 0.0621 -> 0.0189 exactly; the remaining step to about zero on the official metric is the class rule, not class averaging. The brief's instruction to show that +0.062 lands on +0.001 to +0.002 through dilution cannot be satisfied as stated; the measured decomposition is in FINDINGS.md and results/taskB/metric_gap_{A3,A2,FUSE}.json. Same pattern for A2 (union +0.0102, gt-only +0.0003) and fusion (+0.0072, +0.0008).
Also recorded: on flagged frames always-A3 reaches 88% of the oracle (test) and 76% (out of fold), so the oracle's headroom over the best fixed policy is +0.0086 / +0.0062; the learned selector is below always-A3 on both sets.
Tools: tools/policy_table.py (policies side by side from selector_results.json), tools/metric_gap.py (A0 vs action under three scorings, test frames only because the saved masks exist only for test).

## NEEDS OWNER
1. Licensed template in git history: commit 10fa909 on origin/stereo-gated contains PrebuiltUI template source whose licence forbids public publication. It was untracked afterwards but remains in the remote's history. Removing it needs a history rewrite and force-push, which I have not done. Matters only if the repository is public. Your decision.
2. Publishing EndoVis frames or overlays (an original-versus-predicted view): governed by the dataset's own licence and data-use terms, so it is not something that can simply be approved here. The site shows numbers only. Check the terms before any image view is considered.
3. The working tree has an uncommitted .gitignore change (it adds FINAL_PROMPT.md). I did not commit it because it is not my change; say if you want it committed.
4. FINAL_PROMPT.md now ends with a Task P (paper baselines, figures, evaluators). P.3 says no extra database may be downloaded until the owner chooses between A (EndoVis 2018 only, described honestly), B (EndoVis 2017 zero-shot, evaluation only), C (CholecSeg8k zero-shot, evaluation only, monocular). The instruction "you have all the access ... do what it takes" is not a choice among these, and the rule overrides the standing "only this dataset" rule, so nothing is downloaded. Task P runs after N; the parts that need no extra data (P.1 baselines trained on EndoVis 2018, P.2 figures, P.4 definitions, P.5 tables) can proceed.

### Task O (after A and B): website update, deployed
Single source of truth, now enforced. tools/export_web_data.py has two stages: stage 1 turns the sources that are not JSON under results/ into JSON files under results/web_extract/ (per-frame test numbers from choices_test.npz + work_v2/actions, the conformal risk from work_v2/stage2/stage2_report.json, the Task 3 numbers parsed from the two measure_recovery logs, the per-pair stereo table parsed from STEREO_REPORT.md), each with a "source" field; stage 2 reads only results/ JSON and writes web/data/results.json plus a "_sources" map. The Task 3 parser reproduces the previously hand-typed values exactly; the two hand-typed blocks ("recovery", "difficulty") are gone (the difficulty Spearman values now come from results/task4/difficulty_report.json). tools/verify_web_data.py: (1) every numeric field in results.json (12,540) must equal a number in a JSON file that its block declares as its source, with the first match written to web/data/provenance.json as file:key; (2) page and component source may not contain a result-like number literal (decimal, percentage, number with unit, signed, scientific, or 3+ digit integer). Structural facts are an explicit allowlist printed on every run. Limit, stated: the data check matches on value within the declared files, so a small integer can match several keys (the match count is recorded). Result: PASS, 0 untraced fields, 0 source literals; run before the deploy.
Process errors (not repo errors): a multi-line patch failed because the files use CRLF; the helper wrote nothing and was made line-ending aware. A type error (shareOfOracle indexed a record whose headroom entry has a different shape) was fixed before the build passed. My first verifier scan skipped prose lines containing semicolons; found and fixed, which exposed three more literals. Caught before deploy: the "differs by at most" sentence on /selector/ was computed across models as well as aligners (0.0014) when it said "between the three settings"; corrected to the per-model spread across aligners. The first column description on /results/ wrongly included the "Frames changed" column; corrected.
Pages changed (numbers and wording):
- /results/: new section "What correction does on the frames it is meant for": policy table with flagged-frame change, all-frame change and the official change side by side (denominators stated), the three-step decomposition (dilution; the class rule; invented classes removed), oracle share of A3 and selector. Global numbers kept. Literals moved to data.
- /selector/: "Figures corrected" notice (earlier figures used an unaligned temporal candidate); new section comparing the three alignments with sequence-clustered intervals; flagged-frame column in the policy table; the selector's worse-per-correction harm (3.1% vs 1.2% of corrected) and the not-beating-always-A3 caveat; oracle headroom over always-A3.
- /review/ (SurgiReview): "Largest IoU drop" button beside "Biggest improvement"; harmful-correction count for the sequence and the whole test set; instrument-IoU definition stated.
- /frameguard/: flagged-frame IoU change tile; suggestion mix wording computed from the data (it previously said "about three quarters"); conformal numbers from data.
- /stereo/: the sentence "the second camera did not help" is replaced by the mechanism (disparity-based evidence; left-right consistency on instrument vs tissue pixels, both calibration-only and refined); the brief-expectation sentence removed; "Not measured yet" note for the cross-view comparison (Task D not done at this point).
- Home: Numbers section reframed (global figure beside the flagged-frame figure and the class-rule explanation, the corrected stereo sentence); role labels reworded as "Demonstration for ...". /bench/, /limits/, /instruments/, /maskcheck/, /problem/: result literals now computed from the data; /limits/ gains the invented-class and temporal-alignment items. Every page header carries the research-prototype disclaimer.
Not changed on purpose: no EndoVis images; no weights; no analytics; no comparison to published numbers; web/ stays git-ignored; /run/ lists no website build steps.
Deployed to https://stereo-gated-results.vercel.app (production); all 13 routes returned HTTP 200 (/, /review/, /learn/, /frameguard/, /bench/, /maskcheck/, /instruments/, /problem/, /results/, /stereo/, /selector/, /limits/, /run/).
Published numbers that changed on the site, by page: none of the official mIoU values changed (M1 0.6128 ... M6 0.6128). New numbers were added (flagged-frame and decomposition figures, alignment comparison). The wording corrections are listed in "Published numbers: what changed" above.


## Task C (failure-detection baselines): working notes
- Design fixed before any score was seen: risk = 1 - official frame IoU at 1280x1024 of the single-pass mask; bad frame = official IoU < 0.5 (the repo's standing threshold, not tuned); predicted foreground = every predicted non-background pixel (the set the official metric scores); softmax at temperature 1; empty foreground scores 0; instrument-only variants are stored as secondary and never substituted. The segmenter has no Dropout layer, so 'MC-dropout' is MC stochastic depth (DropPath, trained rate 0.1, switched on at inference); this is stated in the table. 4-model ensemble is test only.
- Check: the fresh single-pass masks agree with the saved A0 masks on 99.85 to 99.94% of pixels (frames 0-2 of test seq_1), and full-resolution official IoU matches the working-grid value to within 0.003 on the first eight frames. The residual difference is the fp16/preprocessing path, not a bug; the risk uses the fresh mask so scores and risk come from the same pass.
- Stage 1 (tools/failure_detect_collect.py, GPU, about 2 s per frame) and stage 2 (tools/failure_detect_table.py, CPU) are separate so the table can be rebuilt without inference.

## Task D (cross-view indicators): working notes
- sitr/crossview.py, tests/test_crossview.py (16 checks, all pass), tools/add_crossview_indicators.py. Two test failures on the first run were test bugs (a float == comparison; a stand-in rectifier whose maps did not match its declared resolution); the module was not changed.
- Choices stated: statistics run on the instrument class ids only (the set the gate's other indicators use); the field-of-view strip is 186 px at 1280 wide, scaled to the 320-wide cache grid (47 px), dropped from the left edge of the left view and the right edge of the right view, each intersected with its own valid-pixel mask (valid_left / valid_right: the brief names valid_left only; using the symmetric mask for the right view is the same rule applied to the other image). Surviving column fraction about 0.78 to 0.79 per view after both the guard and the rectification border. Both views empty returns the constants (1,0,0,0,1,1) (perfect agreement); Pearson with a constant vector is 1.0 if both are equal constants, else 0.0. The cached left masks are on the 256x320 grid, the right-view labels (512x640) are brought to it by nearest neighbour.
- Indicator sets registered in sitr/reliability.py: crossview6, paper4_stereo3, paper4_crossview6. The decision rule is in the docstring and code of tools/compare_crossview.py, written before any number was seen: out of fold, paper4_crossview6 must beat paper4 by at least +0.02 AUROC with a sequence-clustered interval excluding zero. The test set is scored once, by a separate stage that refuses to run until the out-of-fold decision file exists.

## Task K (consecutive-frame ground-truth IoU): done
results/CONSECUTIVE_FRAME_IOU.md, results/taskK/consecutive_frame_iou.json, tools/consecutive_frame_iou.py (+ _md.py writes the document from the JSON). 15 training sequences, 2220 consecutive pairs, pooled instrument IoU mean 0.323 (95% 0.286 to 0.361), median 0.272, 76% of pairs below 0.5; random pairs 20+ apart 0.056. Per sequence 0.211 to 0.521. Matches DATASET_FACTS section 6 (0.30-0.45 on three sequences) and extends it to all 15.
Brief vs measurement: the brief asks the write-up to note that "no challenge participant used temporal information". I checked the paper's HTML rendering: the two quoted sentences about 1 Hz subsampling and removing similar frames are verbatim; the paper does not state that no team used temporal information, and one method (Method 4) takes a left/right frame pair as input. The write-up says exactly that and does not repeat the unverified claim. docs/SOLUTION_V3.md line 87 contains the unverified claim; it should not be cited until checked against the full method descriptions.

## Task I (metric ablation): done
results/taskI/metric_ablation.{json,md}, tools/metric_ablation.py. Same M1 test predictions, one choice changed at a time from the exact official protocol. The reference row reproduces the published M1 mIoU 0.612823 to 1e-16. Effects in mIoU points: background included +3.23; class counted if in ground truth OR prediction -11.07; per-class mean over frames then classes (mcIoU) -11.72; sequence means unweighted -0.01; evaluated at 512x640 +0.04. The two large effects (presence rule, mcIoU aggregation) are each far larger than any correction effect measured in this project.
Brief vs measurement: the brief says 11 of 12 classes are scored under mcIoU. With background excluded and suction-instrument absent from the test ground truth, 10 classes are scored (shaft, clasper, wrist, kidney, covered kidney, thread, clamps, needle, intestine, probe). 11 appears only if background is counted. Task J's "11 active test classes" is 10 for the same reason. Reported as 10.

## Task J (per-class, per-sequence, rig leakage): done
results/taskJ/per_class_and_rigs.json, tools/per_class_report.py. Test mIoU / mcIoU per sequence: seq_1 0.668 / 0.535, seq_2 0.564 / 0.512, seq_3 0.806 / 0.776, seq_4 0.413 / 0.433; pooled 0.6128 / 0.4956. On test seq_4 the class that collapses is kidney-parenchyma: IoU 0.153 on 227 frames against 0.800 on the other three sequences (drop 0.646); then ultrasound-probe 0.224 vs 0.446, shaft 0.672 vs 0.887, wrist 0.454 vs 0.661, clasper 0.457 vs 0.585. Why is not established here. Seq_4 shares a camera rig with training sequences 13, 15 and 16, so an unseen rig is not the explanation.
Rigs identified from the calibration files' parameter values (tolerance 0.05; largest within-group difference 0.01, smallest between-group difference 12.08): 7 distinct calibration files, 5 rigs. A = train 1,2,3,4,6,7; B = train 5; C = train 9,10,11,12 plus TEST 1,2,3; D = train 13,15,16 plus TEST 4; E = train 14. This confirms the brief's grouping. A first attempt that hashed parameters rounded to five decimals split rig C into two groups because the test file stores fewer digits (focal length 1051.66 vs 1051.67); replaced by the tolerance grouping above.
Leave-one-rig-out is infeasible: holding out rig A leaves suturing-needle with zero training frames (needle appears in train seq_1 with 1 frame and seq_4 with 11, both rig A). Holding out any other rig leaves 12 needle frames. The official test set shares rigs with training (test 1-3 with C, test 4 with D), so it does not measure generalisation to an unseen camera rig. No fold was trained.
clDice for all classes was not computed (the flag exists, --cldice); the thin-class clDice (shaft/thread/needle) is already in results/task6/final_results.json.

## Task H (per-pixel occlusion as a localisation signal): done, negative
results/taskH/occlusion_localisation.json, tools/occlusion_localisation.py. Pixel-level AUROC for "this predicted-instrument pixel is wrong", out of fold (15 sequences, sequence-clustered intervals): occluded 0.501 (0.497 to 0.504); left-right check failure 0.522 (0.506 to 0.544); low disparity confidence 0.529 (0.511 to 0.555). The non-stereo reference, proximity to a mask boundary, scores 0.741 (0.679 to 0.808). Within boundary-distance strata (removing the boundary confound) the stereo signals stay at 0.504, 0.544 and 0.541. Test (computed once): 0.497, 0.505, 0.516. Per-pixel occlusion is at chance as a localiser, and the other two disparity signals are only slightly above it. A frame-level failure did not predict a pixel-level success. Recorded as a negative result.

## Task G (stereo versus temporal information bound): done
results/taskG/information_bound.json (+ per_frame.json), tools/information_bound.py. Limitation first: the right view has no labels, so the stereo axis is a BOUND (a ground-truth left mask carried into the right view and back), not a score against right-view ground truth. 445 frames (every 5th frame of the 15 training sequences with instrument ground truth in the stereo-valid set), rectified-left geometry, a 320x256 grid, instrument IoU.
Results (mean, 95% sequence-clustered bootstrap): stereo round trip with two independently computed disparities 0.941 (0.927 to 0.954), median 0.967; the same-map round trip, scored on its own (larger) defined set, 0.991 (0.988 to 0.993), so the inflation from reusing one disparity map is about 5 points; temporal ground-truth mask t-1 warped to t, on the same pixels: identity 0.310 (0.265 to 0.353), Farneback 0.426 (0.378 to 0.476). Stereo minus best temporal +0.516 (0.467 to 0.566); stereo is higher on 98.7% of frames. The round trip is defined for 56.3% of ground-truth instrument pixels (two-map; 76.3% for the same-map variant), against the 45.6-54.3% left-right consistency quoted for instrument pixels in Task 2.
Reading, and what this does NOT show: (1) on the pixels where defined, the stereo axis preserves far more of the mask than the temporal axis does, so by the pre-stated rule cross-view propagation is not shown dead by this measurement. (2) The comparison is not like for like: the stereo target is the same instant seen from a second camera, the temporal target is a different instant, and the round trip carries a mask through geometry rather than predicting anything, so it is an upper bound on what stereo could offer, and it says nothing about prediction errors. (3) The defined set is the pixels with consistent disparities, which selects the easy pixels. (4) It does not contradict the measured failure of action A1 (Task 5: +0.0001 mIoU, Pearson -0.027, never chosen): A1 pulls the right view's PREDICTED labels, not ground truth. Why A1 failed given this bound is not established here; one testable candidate is that the right-view prediction shares the left view's errors. Detection and propagation remain separate claims and the cross-view detection result (Task D) is reported separately.
The region aligner returns the unmoved mask in this experiment because region matching needs the target frame's predicted mask, which does not exist for a ground-truth-only measurement; its row equals identity and is not an independent third aligner here.

## Task L (label-noise probe): done
tools/label_noise_probe.py, results/taskL/{oof_loss_ranking.csv, top_frames.json, label_noise_summary.json}. All 2235 training frames ranked by out-of-fold cross-entropy (raw softmax, the fold model that excluded the sequence). Top 20 inspected by eye as photograph | ground truth | prediction (local images in work_v2/label_probe, never committed or published). Judged by me, not a domain expert.
Result: 0 of 20 look like label error; 20 of 20 look like model error on tissue classes (the model hedges between kidney-parenchyma, covered-kidney and untinted background tissue in large patches while the instruments are fine; the ground truth is a plausible, consistent labelling). 14 of the 20 are from train/seq_16 and 6 from train/seq_5. Only 1 of the 20 is a gate failure (official IoU < 0.5), out of 310 such frames (13.9% of 2235; instrument-IoU < 0.5 gives 619, 27.7%). So this probe finds no label error that could account for the gate's failures; for top-loss frames like these the exact one-sided 95% upper bound on the share that are label errors is 13.9% (0 of 20). Limits: a loss ranking surfaces anatomy disagreement, not instrument-label error; the 7 repaired masks in repairs/ are already applied; the probe cannot rule out a few wrong instrument labels outside the top 20.

## Task E (segmenter TTA and the 4-model ensemble): done, null result
tools/tta_ensemble_eval.py, results/taskE/tta_ensemble.json. Three separate systems on the 997 test frames, scored from the masks the Task C run saved. Single model (re-scored here): mIoU 0.6128, mcIoU 0.4956 (10 classes), boundary F1 0.424, clDice shaft/thread/needle 0.901/0.764/0.000; it equals the published M1 mIoU to five decimals. TTA (6 passes): mIoU 0.6139, mcIoU 0.4971, boundary F1 0.428, clDice 0.908/0.771/0.000. 4-model ensemble (a DIFFERENT SYSTEM: four trained models; test sequences only because every training sequence lies inside some member's training set): mIoU 0.6105, mcIoU 0.4977, boundary F1 0.398, clDice 0.898/0.757/0.000.
Differences from the single model in mIoU points (95% sequence-clustered bootstrap, 4 sequences so coarse): TTA +0.10 (-0.20 to +0.28); ensemble -0.23 (-0.91 to +0.45). mcIoU points: TTA +0.15, ensemble +0.21. Neither moves needle off 0.000. The ensemble gains on clamps (0.653 to 0.698) and ultrasound-probe (0.308 to 0.317) and loses on covered-kidney (0.401 to 0.378) and boundary F1; clamps come from a single test sequence. Reported as a negative result: neither TTA nor the ensemble gives a gain larger than its interval, and the ensemble costs four forward passes and four trained models. They stay separate rows and do not replace the single-model number anywhere.
Process error: the first run failed with a memory error because indexing an NpzFile inside the frame loop re-reads the whole array on every access; arrays are now loaded once per sequence.

## Task D (cross-view indicators): DONE, pre-set test FAILED (stop condition applied to this task only)
sitr/crossview.py, tests/test_crossview.py (16 pass), tools/add_crossview_indicators.py, tools/compare_crossview.py, results/taskD/ (SUMMARY.md has the full write-up). Rule fixed in code before any number: out of fold, paper4 + crossview6 must beat paper4 by >= +0.02 AUROC with a sequence-clustered 95% interval excluding zero.
Out of fold (GBM, K=5, bad = instrument IoU < 0.5): paper4 0.7449 (reproduces the committed 0.7449), paper4+disparity3 0.7747, paper4+crossview6 0.7937, crossview6 alone 0.7133. Decision: gain +0.0488, interval -0.0396 to +0.1601 -> FAIL. Not searched for a variant; no indicator dropped. Leave-one-out: q_confidence carries the weight (+0.050); of the cross-view six, xv_area_ratio_max +0.006, xv_ncomp_diff +0.004, xv_runcount_corr +0.003, while removing xv_rowspan_iou would have raised AUROC by 0.021.
Test (once, after the decision): paper4 0.8464 (reproduces committed), +disparity3 0.8301, +crossview6 0.8988, crossview6 alone 0.8287; cross-view over paper4 +0.052 (+0.026 to +0.217, 4 sequences, coarse).
Not a pass, recorded as an observation: AURC (not the rule's metric) improves with an interval excluding zero both out of fold (-0.027, -0.051 to -0.008) and on test (-0.022, -0.042 to -0.009). Choosing that metric after seeing the AUROC result would be exactly the forking-path the rule exists to prevent, so the decision stays FAIL. The disparity control moved +0.030 out of fold and -0.016 on test (noise-like).
Choices stated and recorded elsewhere in this log: instrument class ids only; 186 px guard scaled to 47 px on the 320-wide grid; symmetric valid-pixel mask for the right view (the brief names valid_left only); empty-in-both-views constants (1,0,0,0,1,1); Pearson with a constant vector is 1.0 if both are equal constants else 0.0.
Process notes: the out-of-fold stage took 2 h 04 min because the machine was running five other jobs at once (the stage is CPU-bound GBM fitting); the committed disparity comparison took 36 minutes alone.
Process error (Task D): registering the cross-view indicator sets made tests/test_pipeline.py fail two checks (the synthetic cache fixture had no cross-view indicators, and two tests loop over every registered set). A fixture gap, not a product bug: the fixture now carries them, and a new test asserts that a cache without cross-view indicators refuses the cross-view sets. All suites pass (pipeline 37, stages345 20, selector 9, actions 12, stereo 27, eval 9, measure 7, difficulty 9, crossview 16).
Process error (Task C): the table builder's frame-alignment check refused to continue because the cache's instrument IoU and the action data's differ by up to 1.4e-3 on train/seq_1. They come from two separate inference runs (fp16/cuDNN noise, about 0.1% of pixels), not from different frames; the check now requires max difference < 0.02 and correlation > 0.999 per sequence and records the observed values in the output. A shifted frame list would differ by tenths.

## Task C (failure-detection baselines): DONE
results/taskC/SUMMARY.md (write-up), failure_detection_cost.md/.json (full table with intervals, passes, models, ms/frame), failure_detection.json, per_frame_scores.csv, latency.json. 3,232 frames scored (2,235 out of fold, 997 test), every number in the table traces to those files.
Result, stated plainly: under the official-IoU risk the cheapest baselines (max-softmax, mean entropy: one pass, ~80 ms) match or beat the production gate: out-of-fold AURC 0.269 / 0.270 vs 0.303, test 0.285 vs 0.308 (AUROC 0.786 / 0.773 vs 0.740 out of fold). TTA entropy is best on test (AURC 0.273, AUROC 0.850, six passes). All intervals overlap (15 and 4 sequence clusters). The gate regresses instrument IoU while the risk is the 12-class official IoU; on its own label (instrument IoU < 0.5) the gate is the best or near-best detector (AUROC 0.788 out of fold, 0.844 test vs 0.704 / 0.710 for max-softmax), which is where the site's "0.84" comes from. Ensemble (4 passes, 4 trained models, test only) AURC 0.312, no better than one-pass max-softmax; cross-view gate (2 passes) 0.305 test / 0.298 out of fold, near the ensemble at half the passes, behind one-pass max-softmax, and it failed its own pre-set test (Task D).
Brief vs measurement: the brief expects cross-view at one extra pass to land near the ensemble on AURC (it does, 0.305 vs 0.312 on test) and treats that as the efficiency result; the table also shows that neither is better than max-softmax, which the brief did not anticipate. Reported as found.
Choices stated in the code and SUMMARY: foreground = predicted non-background; temperature 1; empty foreground scores 0; MC "dropout" is MC stochastic depth because the segmenter has no dropout layer; the bad-frame threshold 0.5; instrument-only variants stored as secondary, never substituted.
Process notes: (1) the fresh single-pass masks agree with the saved A0 masks on 99.85-99.94% of pixels, and the table's frame-alignment check was relaxed twice after measurement (first because max-abs differences of up to 0.3 occur on single frames when an instrument class flips between two inference runs; the final rule is mean abs difference < 0.005 and correlation > 0.99, observed worst case 0.0022 and 0.994, while a one-frame shift gives correlation <= 0.82). (2) the gate fits are cached in work_v2/failure_detection/gate_scores_gbm.pkl because they take about 20 minutes. (3) pip-installed transformers 4.46.3 into the sitr environment for SegFormer (Task P.1).

## Task F (mcIoU, hallucination rate, retire A1, add A4): CPU parts done; A4 and its GPU pass pending
tools/mciou_hallucination.py, tools/train_selector.py (new flag --retire-a1), tools/f_cost.py; results/taskF/ (mciou_hallucination.json, no_a1/selector_results.json + choices_test.npz, cost.json).
F.1 mcIoU: M1 0.4956, M2 0.4962, M3 0.4956, M4 0.4962, M5 0.4954, M6 0.4956 (10 of 11 foreground classes; suction-instrument has no test pixels; 11 only if background were counted). Per-class breakdown is in the JSON; mIoU and mcIoU move together within 0.001, so no correction is visible in either.
F.2 Hallucination (test, floor 0.05% of the frame = 655 px, a stated choice; floors 0 and 0.5% stored as a check): 58.4% of frames contain at least one invented class under M1 (0.78 invented classes per frame; 15.2% of all predicted (frame, class) pairs are invented). M2-M6 are within 1.5 points of that (M5, the only variant that applies the refinement to every frame, 57.0%). The official score cannot see any of this.
F.3 Retire A1: candidate set {A2, A3} (the stereo-temporal fusion contains A1 and collapses to A2, so it goes too; the always-A1 and fusion policies are no longer reported, M2/M4/M5 are retired with it; code and the earlier measurement stay in the record). Selector re-run out of fold and on test (results/taskF/no_a1): out-of-fold mean change over all frames +0.0055 (oracle +0.0125, always-A3 +0.0098); test +0.0176 (oracle +0.0213, always-A3 +0.0189, always-A2 +0.0031). The selector still does not beat always-A3; it harms 1.2% of what it corrects on test against 1.0% for always-A3. Official test score with the retired-A1 selector: mIoU 0.6128, mcIoU 0.4956, identical to M6. Choice mix on test: A3 21.0%, A2 4.6%, keep 74.4%.
Cost: composed from measured components, the retired-A1 selector costs 153 ms per frame against 792 ms for the old M6, because flagged frames no longer need stereo matching (1.7 s on the CPU) or the right-view pass. The old tile "2.8x faster than correct-all" compared 792 ms with 2201 ms for the fusion applied to every frame (M4, now retired). Like for like with the retired set (temporal then refinement on every frame, 360 ms) the retired-A1 selector is 2.35x faster; against the old M4 baseline it would be 14.4x. Both are stated; the like-for-like figure is the honest tile.
Pending (needs the GPU, after P.1 training): action A4 (hallucination repair) and its effect on the hallucination rate and on mIoU reported separately; the out-of-fold regeneration is CPU-heavy (about 6 h for all actions earlier) so A4 will be run on its own, without recomputing A1/A3/fusion.

## Task Q (Stage 5 identity verification): code audit and ceiling done; verifier run pending the GPU
tools/identity_ceiling.py, results/taskQ/identity_headroom.json.
Audit of the teammate's sitr/identity.py (read, not changed): HungarianIdentityVerifier returns IdentityResult(mask=mask, ...) with the INPUT mask, by its own stated design ("identity uncertainty does NOT modify the segmentation mask"); and `switches` is initialised to 0 and never incremented, so the identity-switch count it declares (and eq. 37 relies on) is always 0. By construction, M7 = M6 + Stage 5 has exactly the masks of M6, so mIoU, mcIoU, per-class IoU and class sets are identical to M6 whatever the data. This is reported, not worked around, and the code is untouched. It is not a contract mismatch (the Protocol in stage34_interface.py allows a mask in the result); it is a design in which Stage 5 cannot move any of the metrics asked about in Q.2.
What identity relabelling could be worth (a GROUND-TRUTH ceiling, not a method): relabelling every connected predicted instrument region to the instrument class that covers most of it in the ground truth, on the M6 masks (512x640 grid, 997 test frames): mIoU +2.07 points (0.6132 to 0.6339), mcIoU +5.37 points (0.4964 to 0.5501). The mcIoU gain is 2.6 times the mIoU gain, the pattern the literature reports for relabelling temporal modules (MATIS +1.89 mIoU / +5.39 mcIoU), and here it comes without changing any mask geometry. The ceiling relabels 7.4% of predicted instrument pixels in 63% of frames. Per class: shaft 0.859 to 0.879, wrist 0.630 to 0.665, clasper 0.567 to 0.574, thread 0.432 to 0.438, clamps 0.656 to 0.659, suturing-needle 0.000 to 0.134 (a share of the needle is already segmented but labelled as another instrument). Restricted to the clasper/wrist pair only: +0.47 mIoU, +0.30 mcIoU.
Clasper versus wrist (M6, pixels): 20.3% of ground-truth wrist pixels are predicted clasper, 2.6% of ground-truth clasper pixels are predicted wrist; 93.8% of clasper and 76.2% of wrist pixels are right. The confusion matrices before and after the ceiling relabelling are in the JSON.
Reading: the literature's relabelling gain exists on this data as a ceiling, and the shipped Stage 5 captures none of it because it never relabels. A relabelling stage would be a new component, not a wiring change; per the task's stop condition and the owner's rule not to redesign the teammate's stage, none was built.
Pending (needs the GPU): running the verifier on the test sequences with real frame states to report the share of frames with an uncertain region (c_max 0.5, the teammate's default, not tuned) and to assert numerically that the output masks equal the input masks.
