# Selective Temporal Recovery — Stages 1 & 2

Reference implementation of the **first two components** of *"Temporal Recovery
of Robotic Surgical Instrument Segmentation Under Difficult Visibility
Conditions"*, on the **EndoVis 2018 Robotic Scene Segmentation** dataset.

```
I_t ──► [ Stage 1 · base segmentation f_seg ] ──► P_t , S_t
                                                    │
                                                    ▼
                  [ Stage 2 · reliability  r_t = σ(f_θ([q_t,T_t,A_t,F_t])) ]
                                                    │
                                     r_t ≥ τ  ──────┴────── r_t < τ
                                    accept S_t            route to recovery
                                                          (Stages 3–4: teammate)
```

| report § | component | status |
|---|---|---|
| IV‑B / V‑C | base segmentation `f_seg` (TernausNet‑11) | **implemented** |
| IV‑C / V‑D | reliability estimator, eqs. (6)–(13), (30) | **implemented** |
| IV‑D / V‑E | reliable memory `B_t`, eq. (15) | **implemented** (the buffer; the prediction memory for recovery is part of Stage 4) |
| IV‑E | alignment `W_{s→t}`, eq. (16) | identity + Farneback provided; region-level default is Stage 3 |
| IV‑F / IV‑G | recovery `R_φ`, identity verification | **not implemented — see `HANDOFF.md`** |
| V‑H / V‑I | event construction, event-based metrics | **not implemented** |

Nothing here fakes a later stage. `NOVELTY.md` separates what is standard, what
is the paper's contribution, and what this implementation adds. `HANDOFF.md` is
the contract your teammate builds Stages 3–5 against.

---

## Install

```bash
pip install -r requirements.txt
```
`torch`, `torchvision`, `opencv-python-headless`, `numpy`; `matplotlib` only for
one diagnostic figure. No albumentations / scipy / sklearn — the augmentations,
rank statistics and AUC are implemented directly, so there is nothing to pin.

---

## Your dataset

Point `--root` at `endovis2018_cvdataset/`. **Everything except `test_data/` is
training data**; the loader classifies splits automatically.

```
endovis2018_cvdataset/
  miccai_challenge_2018_release_1/
    __MACOSX/…                                   ← junk, excluded automatically
    miccai_challenge_2018_release_1/
      labels.json
      seq_1 … seq_4/{camera_calibration.txt, left_frames, right_frames, labels}
  miccai_challenge_release_2/…   seq_5 … seq_7
  miccai_challenge_release_3/…   seq_9 … seq_12
  miccai_challenge_release_4/…   seq_13 … seq_16
  repairs/repairs/seq_1_frame042.png …           ← corrected labels, applied
  test_data/test_data/
    labels.json  run.py  utils.py
    seq_1-<ts>-001/seq_1/…                       ← 4 × 250 frames, held out
```

**15 training sequences × 149 frames = 2235 annotated frames** at 1280×1024.
There is no `seq_8`; the loader expects its absence and flags anything else.

### Three things in this download that silently corrupt results

The audit checks all three and fails loudly rather than training on bad data.

1. **`__MACOSX/` shadow trees.** They contain *real* `labels/`, `left_frames/`
   and `right_frames/` folders full of 1 KB AppleDouble `._frameXXX.png` files.
   Any loader that searches for `left_frames` finds four phantom copies of
   seq_1–seq_4. Excluded everywhere.
2. **`repairs/` = the challenge's 7 corrected masks** for release 1 (seq_1
   frames 042/043/044/073, seq_4 frames 135/137/138). Training without them
   means training on seven masks the organisers state are wrong. Applied as a
   **non-destructive override — your files are never modified**.
3. **Two `labels.json` schemas.** `test_data` uses `{"classes": […]}` with
   **RGBA** `[0,0,0,128]` plus an `"active"` flag; the releases use a bare list
   with RGB. Each release lists only its own classes, so the true class table is
   their union. Alpha is dropped; conflicts raise instead of being guessed.

A fourth, handled quietly: training `seq_1` and test `seq_1` are different
procedures, so sequences are keyed `<split>/<name>` and can never collide.

---

## Running it — five commands

### 0 · Audit (always run this first)

```bash
python -m tools.audit_dataset \
    --root /path/to/endovis2018_cvdataset \
    --out runs/class_map.json --scan-colors --check-readable
```

Reports sequences per release, frame/label pairing, resolutions, the unioned
class table, and — the important one — **whether any colour in your label PNGs
is missing from the class table**. An unmapped colour becomes ignore-index and
silently deletes supervision; this is the check that catches it. Exits non-zero
if anything is wrong. *(~2 min with `--check-readable`.)*

This also answers "is preprocessing needed?": **no separate preprocessing pass
is required** — colour decoding, the repairs override and resizing all happen in
the pipeline. Step 0b is a speed optimisation, not a correctness requirement.

### 0b · Bake a resized cache (optional, ~10× faster epochs)

```bash
python -m tools.prepare_dataset \
    --root /path/to/endovis2018_cvdataset --class-map runs/class_map.json \
    --task instruments --height 256 --width 320 \
    --splits train test --out data/prep_256x320_instruments
```

Otherwise every epoch re-decodes ~3 GB of 1280×1024 PNG and re-runs a 24-bit
colour LUT over 1.3 M pixels per frame, which makes the **data loader**, not the
GPU, the bottleneck. This does the same resize and the same nearest-neighbour
label handling once. Numbers are unchanged. *(~4 min, ~350 MB.)*

### 1 · Stage 1 — base segmentation

```bash
python -m tools.train_stage1 \
    --prepared data/prep_256x320_instruments --class-map runs/class_map.json \
    --task instruments --val-seqs 2 5 9 15 \
    --epochs 60 --batch-size 8 --out runs/stage1
```

*(~2.5 h on one RTX 3090 / A5000, ~6 GB VRAM at batch 8.)* Drop to
`--batch-size 4` for 8 GB cards. `--resume runs/stage1/last.pt` continues.

### 2a · Cache what Stage 2 needs (one frozen pass)

```bash
python -m tools.cache_features \
    --prepared data/prep_256x320_instruments \
    --checkpoint runs/stage1/best.pt --splits train test --out runs/cache
```
*(~3 min. Produces ~250 MB.)*

### 2b · Stage 2 — reliability estimator

```bash
python -m tools.train_stage2 \
    --cache runs/cache --val-seqs 2 5 9 15 \
    --memory-size 5 --tau-train 0.5 --epochs 300 --out runs/stage2
```
*(~2 min on CPU — the whole point of the cache.)*

Outputs `runs/stage2/stage2_report.json`: MSE/MAE/bias, Pearson and Spearman
correlation between `r_t` and true IoU, AUC for detecting failed frames,
per-indicator correlations (which of `q,T,A,F` actually carries signal), the
cold-start and degenerate-frame rates, the exposure-bias gap, and the τ sweep.

---

## Task presets

`--task` decides what the model predicts and what `Ω^fg` means.

| preset | classes | use it for |
|---|---|---|
| `instruments` | background + each instrument class | **default.** Anatomy merged into background: identity matters for Stage 5, kidney does not. |
| `binary` | instrument vs rest | the simplest baseline; easiest to compare externally |
| `parts` | background + shaft/wrist/clasper | the articulated-part setting the report uses |
| `full` | all 12 | the challenge's own task, for a leaderboard-comparable number |

`Ω^fg` in eqs. (6)–(10) is always the **instrument** foreground. EndoVis 2018
annotates anatomy too, so "not background" is *not* "instrument" — see
`NOVELTY.md` §3.1.

---

## Hyperparameters, and why each one

Tuned for this dataset's shape (2235 frames, 15 sequences, ~2 % instrument
pixels), not copied from a generic recipe.

| setting | value | reason |
|---|---|---|
| `--arch` | `resnet34_unet` | five clean skip levels **including /2**, which is what keeps 2–4 px clasper tips alive. Measured 2.4× faster per epoch than `unet11` at equal IoU. Use `--arch unet11` for the report-comparable Baseline A; `convnext_unet` is strongest but ~1.7× the cost |
| `--lambda-tversky` | 0.5 | Focal-Tversky (α=0.3, β=0.7, γ=0.75). Dice weights FP and FN equally, so dropping a 3 px clasper entirely costs almost nothing and the model learns to. β>α penalises exactly that. **Set 0 for the report's literal eq. (29)** |
| `--indicators` | `extended` | the report's four plus entropy, margin, compactness, fragmentation and drift — all free from the same forward pass. **Run `--indicators paper4` too and report both**; on synthetic data they tied |
| input size | 256×320 | 1280×1024 is 1.25:1; 256×320 preserves it exactly and is a multiple of 32 (the encoder downsamples by 32). Halving to 128×160 loses thin claspers. |
| batch size | 8 | fits 8 GB at 256×320; the dataset is small enough that larger batches reduce the number of updates more than they help |
| lr | 3e‑4 decoder | AdamW on a randomly-initialised decoder; 1e‑3 overshoots on 2235 frames |
| `--encoder-lr-scale` | 0.1 | **the one that matters most.** A single lr destroys the pretrained VGG features in the first epochs and costs several IoU points. |
| warmup | 3 % + cosine | stabilises the first epochs when the Dice term is large and noisy |
| loss | CE + Dice, λ=1 each | eq. (29). Dice counters the ~98 % background imbalance; CE alone collapses to all-background |
| CE weights | inverse-sqrt, clipped ×12 | plain inverse frequency puts weight ~500 on suturing-needle and diverges. Applied only for >2 classes |
| epochs / early stop | 60 / 15 | converges by ~40; early stopping on val mIoU avoids overfitting 11 training sequences |
| augmentation | h-flip, ±20 % scale, ±15°, ±6 % shift, brightness/contrast/gamma, motion blur p=0.15, glare p=0.10 | the photometric ones deliberately mimic the report's difficult-visibility causes, so Stage 2 sees a realistic spread of achieved IoU. **No vertical flip** — endoscopic scenes have a fixed camera convention. |
| rotation border | ignore-index | pixels rotated in from outside have no label; filling them with background trains the model to predict background on invented content |
| memory `K` | 5 | eq. (15). At 1 Hz that is ~5 s of context; the report notes windows ≫8 frames inject noise from distant frames |
| `--tau-train` | 0.5 | the quality threshold defining a "reliable" past frame while building history. **Distinct from the deployed decision threshold**, which is swept afterwards |
| scheduled sampling | ramp to 0.5 over the second half | Sec. V‑D. The residual gap is reported, not assumed away |

Validation `--val-seqs 2 5 9 15` is the split commonly used in the EndoVis 2018
literature. **Splits are always at sequence level** — even at 1 Hz adjacent frames are
near-identical and a random frame split leaks (report V‑B.5).

---

## Design decisions you should be able to defend

Each is flagged in the code at the point it applies.

1. **τ is provisional.** The report fixes τ by maximising *event-level recovery
   success*, which needs Stage 4. Stage 2 returns the τ maximising Youden's J
   for detecting frames with `r* < θ`, alongside the routing rate ρ of eq. (2)
   at every candidate. **Re-sweep once recovery exists.**
2. **Alignment defaults to identity, not optical flow.** Sec. IV‑E argues dense
   flow is unreliable at the 1 Hz release rate. Farneback is the ablation:
   `--align farneback`.
3. **`IoU(∅,∅)` is a real conflict.** Eq. (10) drives `r_t → 0` on an empty
   foreground; `IoU(∅,∅)` is conventionally 1. So a correct "no instrument here"
   frame scores as maximally unreliable. `--empty-both-iou` exposes the choice;
   the hit rate is reported. Decide it deliberately.
4. **Challenge IoU is reported under both "present" conventions** (GT-only and
   GT ∪ prediction), because the challenge page is ambiguous and they differ
   when a model hallucinates an absent class. Say which you used.
5. **No latency claims.** The report's budget is <40 ms/frame. Nothing here
   measures it, because timing on a machine that is not the deployment target is
   meaningless. Measure `c_seg` and `c_rel` on your hardware before quoting
   eq. (2). The estimator is 4,545 parameters over four scalars, so `c_rel` is
   negligible by construction — that part is safe to state.

---

## Choosing the hyperparameters from data, not by assertion

The report specifies `tau` and `K` as *validation-set hyperparameters*
(Sec. IV-C, IV-D), so leaving them at a default does not implement the method —
it implements a guess at it. Stage 1 is frozen and cached, so the whole grid
costs ~10 minutes on CPU:

```bash
python -m tools.sweep_stage2 --cache work/cache --out work/stage2_sweep \
    --val-seqs 2 5 9 15
```

Selection criterion is printed before the numbers, so it cannot be chosen after
the fact: AUC for detecting `r* < theta` on held-out sequences under
self-predicted history, with MSE as the tie-break. A *flat* grid is itself a
reportable finding, and the tool says so when it sees one.

## Is the extended indicator set actually better?

The nine-indicator set is this implementation's addition; the report uses four.
A single run of each cannot settle that, because the estimator is randomly
initialised and trained on ~1,600 frames — run-to-run variation is easily as
large as the difference being measured. Comparing one run against one run and
reporting the winner is how a null result gets published as a positive one.

```bash
python -m tools.compare_indicators --cache work/cache --val-seqs 2 5 9 15 \
    --seeds 5 --out work/indicator_ablation
```

It repeats both configurations over several seeds and reports mean ± std on two
metrics that answer different questions — **AUC** (can the gate *rank* bad
frames? this is what the gate is for) and **MSE** (is `r_t` a calibrated
estimate of the quality *value*? this matters only if Stage 3 weights by `r_t`
rather than thresholding it). The decision rule is stated before you see the
numbers: **if the difference in means is smaller than the pooled standard
deviation, the sets are tied**, and a split decision between the two metrics is
reported as a split decision rather than resolved in favour of whichever looks
better.

A tie is a legitimate result. It is exactly the confidence-only ablation the
report proposes in Sec. V-J.4, answered with evidence instead of assertion.

---

## Facts checked against primary sources, not against the project report

The project report is a student write-up and is not itself evidence. These are
the dataset and benchmark facts this implementation relies on, checked against
the official challenge paper ([arXiv:2001.11190](https://arxiv.org/abs/2001.11190))
and the dataset's public description.

| claim | status |
|---|---|
| Frames extracted at **1 Hz**, then visually similar frames **manually removed** | **Verified.** Earlier drafts of these notes said 2 Hz — that was wrong. 1 Hz makes the temporal assumption *weaker*, not stronger. |
| 15 training sequences, 4 test sequences, 1280×1024 stereo | Verified. |
| **12 annotation classes** (background-tissue, instrument shaft / clasper / wrist, kidney-parenchyma, covered-kidney, thread, clamps, suturing-needle, suction-instrument, intestine, ultrasound-probe) | Verified, and matches the class map derived from this dataset's own `labels.json` files. |
| Metric is `IoU = TP/(TP+FP+FN)`, averaged over the classes **present** in a frame, then over frames | Verified — this is why `tools/` reports both "present" conventions rather than picking one silently. |
| 2018 challenge team scores ranged **0.192 to 0.621** (best: OTH Regensberg 0.621; then UNC 0.607, NCT 2 0.585, Digital Surgery 0.579, IRCAD 0.573) | Verified. |

**How to use that last row honestly.** A Stage 1 mIoU around 0.58 on this
codebase is *not* a challenge score and must not be written as one. It is
measured on a held-out split of the **training** sequences, on the
`instruments` class subset, at 256×320 — not on the sequestered test set, not
over all 12 classes, and not at full resolution. The leaderboard range is
useful only as evidence that this order of magnitude is normal for the task,
which it is. Any sentence comparing the two directly is a false claim.

Not verified from a primary source: the **149 frames per sequence** in the
public training release. The challenge paper describes sequences of 300 frames;
the redistributed training set commonly used in the literature has fewer. This
implementation reads whatever is on disk and reports the count, so nothing
depends on the figure — but do not assert it in a write-up without checking
your own download.

---

## Before handing the release to anyone

```bash
python -m tools.verify_release --release work/release
```

The unit tests prove the CODE is right. This proves the ARTEFACTS YOU PRODUCED
are right: it loads the release exactly as your teammate will, replays real
validation sequences through it, and checks that `P_t` is a true distribution,
that `B_t` is complete and bounded by K, that memory never crosses a sequence,
that recovery is offered only on unreliable non-cold-start frames, that ground
truth cannot reach an inference-time indicator, and — the strongest check —
that the routing rate observed on replay matches the rho in
`stage2_report.json`. If those disagree, the shipped estimator, the shipped tau
and the reported numbers are not the same system, and everything downstream
would be built on a mismatch.

Exit code 0 means safe to hand over.

## Tests

```bash
python -m tests.test_pipeline
```
27 invariant tests, no dataset and no GPU needed, ~20 s. They cover the failures
that corrupt results *silently* rather than crashing: junk-path exclusion, IoU
edge cases, ignore-index handling, every architecture's shapes and gradients,
`|B_t| <= K` and the memory window, recovery firing only on the right frames,
and a check that ground truth cannot leak into inference-time features. Run this
after any change and before you trust a number.

---

## Verify the install in ~1 minute, without the real data

`tools/make_dummy_dataset.py` writes a miniature replica of the **exact** layout
above — `__MACOSX` junk, the seven repairs, both `labels.json` schemas, the
missing seq_8, and the train/test name collision — so the loader is exercised
against every trap before you commit GPU hours.

```bash
python -m tools.make_dummy_dataset --out /tmp/fake18 --frames 149
python -m tools.audit_dataset  --root /tmp/fake18 --out /tmp/cm.json --scan-colors
python -m tools.prepare_dataset --root /tmp/fake18 --class-map /tmp/cm.json \
       --task instruments --height 128 --width 160 --out /tmp/prep
python -m tools.train_stage1   --prepared /tmp/prep --class-map /tmp/cm.json \
       --task instruments --val-seqs 2 5 9 15 --height 128 --width 160 \
       --epochs 6 --no-pretrained --out /tmp/runs/s1
python -m tools.cache_features --prepared /tmp/prep \
       --checkpoint /tmp/runs/s1/best.pt --out /tmp/runs/cache
python -m tools.train_stage2   --cache /tmp/runs/cache --val-seqs 2 5 9 15 \
       --epochs 300 --out /tmp/runs/s2
```

It is a **plumbing test, not science.** Never report numbers from it.

---

## Layout

```
sitr/
  labels.py              labels.json union, RGBA, colour LUT, task presets, junk filter
  dataset.py             sequence discovery, repairs override, splits, augmentation
  models.py              TernausNet-11 (logits + decoder features), ReliabilityMLP
  losses.py              CE + soft Dice (27–29), reliability MSE (30)
  metrics.py             confusion matrix, eq. (33) per-frame IoU, Pearson/Spearman/AUC
  align.py               W_{s→t}: identity (default), Farneback (ablation)
  reliability.py         eqs. (6)–(11), memory B_t, streaming, τ sweep
  stage34_interface.py   ►► THE HANDOFF: Protocols + Algorithm 1 driver
  utils.py               seeding, AMP, checkpoints, cosine LR
tools/
  audit_dataset.py       step 0   — integrity audit + class table
  prepare_dataset.py     step 0b  — resized cache
  train_stage1.py        step 1
  cache_features.py      step 2a
  train_stage2.py        step 2b
  make_dummy_dataset.py  replica of the real layout, for install verification
tests/
  test_pipeline.py       19 invariant tests (no dataset, no GPU)
NOVELTY.md               what is new, what is not, what would sink the claim
HANDOFF.md               contract for Stages 3–5
```

---

## One command

```bash
python -m tools.run_all --root /path/to/endovis2018_cvdataset --out work
```

Runs audit → prepared cache → Stage 1 → temperature calibration → feature cache
→ Stage 2 → release assembly. **Every step is skipped if its output exists**, so
re-running after a crash resumes rather than restarts.

The result is `work/release/` — the only thing your teammate needs:

```
release/
  manifest.json           task, classes, tau, K, temperature, indicators, results
  class_map.json          class table derived from YOUR labels.json
  stage1_best.pt          frozen segmentation model
  stage2_reliability.pt   trained reliability estimator
  caches/*.npz            per-frame indicators, predicted masks, r*
  reports/                audit, stage1 history, stage2 report, calibration, figure
  README_FOR_STAGE3.md    how to plug in
```

Their entry point is three lines:

```python
from sitr.runtime import ReleaseBundle
bundle = ReleaseBundle.load("release")
runner = bundle.runner(recovery=MyRecovery(), identity=MyIdentity())

for seq in bundle.sequences("train"):
    runner.reset()                        # memory must not cross procedures
    for state in bundle.stream(seq):      # FrameState: I_t, P_t, S_t, Phi_t, E_t, r_t, B_t
        final_mask, _ = runner.step(state)
```

`bundle.stream()` recomputes the dense posterior `P_t` and features `Phi_t` from
the frozen backbone rather than caching them — 2235 frames of both would be tens
of GB, and a frozen model makes recomputation exact, not approximate.

Omit `recovery=` and you get the honest Stage 1+2 system: reliable frames
accepted, unreliable ones flagged, nothing invented. That is the baseline to
diff every later change against.

### Temperature calibration

Fitted automatically on the validation split after Stage 1 and stored in the
checkpoint. It cannot change `argmax`, so **every Stage 1 metric is unaffected**
— it only makes the posterior honest, which matters because Stage 2's `q`, `H`
and `M` indicators are read straight off it. `reports/calibration.json` shows
NLL and ECE before and after, so the effect is reported rather than assumed.
