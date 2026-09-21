# Novelty — what is actually new in Stages 1–2

This file separates three things that get conflated in write-ups: what is
standard practice, what is the paper's claimed contribution, and what this
implementation adds on top. A reviewer will do this separation anyway; doing it
first is cheaper.

---

## 1. Not novel, and should not be claimed as such

* **The backbone.** Three are provided — `unet11` (TernausNet, the 2018 EndoVis
  baseline cited as [22]), `resnet34_unet` (default) and `convnext_unet`. None
  is novel; they are all standard encoder-decoder segmentation. Keep the
  distinction sharp in the write-up: **`unet11` is the backbone for the
  comparison against Baseline A**, because the report's question is "does
  selective temporal recovery improve robustness", not "does a better backbone".
  Report the headline framework result on `unet11` and the best absolute numbers
  separately on `resnet34_unet`, or a reviewer will say the gain came from the
  encoder. (Measured on the synthetic replica, `resnet34_unet` was **2.4x faster
  per epoch** than `unet11` at equal or better IoU, which is a legitimate
  engineering note — not a contribution.)
* **CE + soft Dice, AdamW, cosine schedule, ImageNet init.** Standard.
* **Sequence-level splits.** Correct practice, not a contribution — though it is
  worth one sentence in the paper, because random frame-level splits on 1 Hz
  video inflate results and some published EndoVis numbers are affected.

## 2. The paper's claimed contribution, which Stage 2 implements

**Temporal context is applied conditionally, on an explicit learned per-frame
decision, rather than unconditionally to every frame.**

The literature review in the report establishes that every reviewed temporal
method (MATIS, VMSIS, TD-SAM, Wang et al., Zhang et al.) runs its temporal
machinery on every frame regardless of whether the current prediction needs
help. The closest prior work — YOLOv8+ByteTrack (Myo et al.) — does recover
low-confidence detections from tracklets, but its trigger is a **fixed detector
confidence threshold baked into a tracking pipeline**, not a learned model of
segmentation quality.

Stage 2 is the concrete difference: a estimator `r_t = σ(f_θ([q_t, T_t, A_t, F_t]))`
trained to regress the IoU the base model actually achieved, using **only
quantities observable at inference time**. That is the gate. It is what makes
the system *selective* rather than always-on, and it is Gap 1 in the report.

What makes this defensible as a contribution rather than a threshold in a
trench coat:

* It regresses a **continuous quality estimate**, not a binary "trust/don't".
  That matters downstream: eq. (21) makes the fusion weight `α_t = min(1, r_t/τ)`
  a continuous function of `r_t`, so the recovered output degrades gracefully
  toward the base prediction as `r_t → τ⁻` instead of jumping at the boundary.
* It fuses **four complementary and individually-insufficient signals**.
  Confidence alone is the ablation the report proposes (Sec. V-J.4, ablation 1),
  and this implementation reports the per-indicator correlation with true IoU so
  that ablation is answered with evidence rather than asserted.
* Its cost is quantified, not waved away: the estimator is 4,545 parameters and
  runs on four scalars, so `c_rel` in eq. (2) is genuinely negligible against
  `c_seg`, which is the entire efficiency argument for selectivity.

## 3. What this implementation contributes beyond the report

These are engineering and methodological contributions. They are the parts I
would defend in a rebuttal, and each is a sentence or two in the paper.

### 3.1 Ω^fg is the *instrument* foreground, not "non-background"

EndoVis 2018 annotates kidney parenchyma, small intestine, covered kidney and
background tissue alongside the instruments. Defining the predicted foreground
in eqs. (6)–(10) as "not class 0" therefore mixes anatomy into `q_t`, `a_t` and
`E_t`, and the reliability score ends up partly measuring how well the model
segments a kidney — which the recovery module will never act on. The class table
is split into instrument and anatomy classes, and `Ω^fg` uses only the former.
This is a small definition that changes what the estimator learns; it is also
the kind of thing a reviewer finds and a paper loses credibility over.

### 3.2 The frozen-backbone feature cache makes Stage 2 trainable at all

There is a circular dependency in the method as written: `T_t` and `A_t` are
defined against the most recent *reliable* frame `t'`; which frame is reliable
depends on the estimator; the estimator is what is being trained. Training
naively means re-running the backbone inside every epoch of the reliability
loop.

Since the base model is frozen at this stage, **its outputs cannot change**, so
one pass caches `q_t`, `a_t`, `E_t`, `S_t` and `r*_t` per frame and Stage 2
becomes arithmetic over 2,235 cached rows. This turns an O(epochs × backbone)
problem into O(1 × backbone) and is what makes hundreds of scheduled-sampling
epochs affordable. It is not a trick to go faster; without it the scheduled
sampling regime below is not practical to run.

### 3.3 Exposure bias is measured, not assumed away

The report identifies the train/inference mismatch (Sec. V-D): training defines
`t'` from ground-truth IoU, deployment defines it from the estimator's own
predictions, and errors in past reliability judgements perturb the current
frame's features. It proposes teacher forcing plus a scheduled-sampling phase.

This implementation runs both regimes **and reports the gap between them as a
number** (`exposure_bias_mse_gap`). A method that claims to handle exposure bias
should show the residual, not just describe the mechanism. If the gap is large,
that is a finding; if it is near zero, that is the evidence the mechanism worked.

### 3.4 Falsification criterion 5 is instrumented

The report commits, admirably, to five conditions under which its hypothesis is
disconfirmed. Criterion 5 — "the estimator's predicted `r_t` correlates weakly
with the true IoU on held-out sequences under self-predicted history" — is
exactly a Stage-2 quantity. `stage2_report.json` reports Pearson and Spearman
correlation and the AUC for detecting genuinely failed frames, under
self-predicted history, on held-out sequences. The criterion can be checked
before any of Stages 3–7 exist, which means the project can fail fast rather
than after building the recovery module.

### 3.5 An extended indicator set, free from the same forward pass

The report uses four indicators. Five more are added, every one computed from
tensors the segmentation forward pass has already produced, so `c_rel` in
eq. (2) stays negligible — which is the whole efficiency argument for a gate:

* **`H` entropy confidence.** `q` is the mean *maximum* posterior. A frame split
  0.50/0.45 between shaft and wrist has a high maximum and is not confident at
  all. Normalised entropy sees that; `q` structurally cannot.
* **`M` top-1 minus top-2 margin.** Separates "confident and correct" from
  "confidently torn between two classes" — the failure mode that dominates
  shaft/wrist/clasper confusion, which is precisely what S3Net [3] was built to
  address.
* **`Bnd` compactness** (1 − perimeter/area over the foreground) and
  **`Frag` largest-component share.** A correct instrument mask is compact and
  largely one piece; under occlusion and blur it shatters. **No indicator in the
  report measures mask shape at all**, yet fragmentation is the visible
  signature of exactly the failures the framework targets.
* **`Drift`** cosine against the immediately previous frame. `F` compares only
  against *reliable* memory, so during a run of unreliable frames it goes stale;
  `Drift` still registers frame-to-frame appearance shock.

**Honest result so far.** On the synthetic replica used for verification the
extended set did **not** beat the paper's four on headline metrics (MSE 0.0025
vs 0.0025, Pearson 0.855 vs 0.856) — the synthetic failure modes are simple
enough that `q` alone nearly saturates. What the per-indicator correlations do
show is that `M` (+0.80), `H` (+0.76) and `Bnd` (+0.69) each track true IoU as
strongly as `q` (+0.79) and far more strongly than `T` (+0.37), `A` (+0.32) and
`F` (+0.04). **Whether this converts into a better gate on real EndoVis data is
an open question that `--indicators paper4|extended` answers with one flag.**
Run it; report whichever wins, and report that you checked. Do not claim the
extension helps until that number exists.

### 3.6 Two ambiguities are surfaced rather than silently resolved

* **`IoU(∅, ∅)`.** Eq. (10) drives `r_t → 0` when the predicted foreground is
  empty, but the IoU of an empty prediction against an empty ground truth is
  conventionally 1. A frame with no instrument present and none predicted is
  therefore a *correct* prediction that the indicators score as maximally
  unreliable. This is a real inconsistency in the method. It is exposed as a
  parameter, and the fraction of frames hitting it is reported, so the choice is
  made deliberately and stated in the paper.
* **"Classes present in a frame"** in the challenge's IoU definition is
  ambiguous between "present in ground truth" and "present in ground truth or
  prediction". They differ materially when a model hallucinates an absent class.
  Both are computed and reported.

### 3.7 Dataset integrity as a stated precondition

Three properties of the actual download will silently corrupt results and none
are mentioned in the report: the `__MACOSX` shadow tree produces four phantom
copies of seq_1–seq_4 whose "labels" are 1 KB AppleDouble junk; the seven
corrected masks in `repairs/` supersede release-1 ground truth; and the four
`labels.json` files use two different schemas, one of which carries an alpha
channel. `tools/audit_dataset.py` checks all three and fails loudly. Reproducible
results start here, and a paper that reports numbers from a corpus containing
phantom sequences is reporting noise.

---

## What would make the *whole framework* novel — and what would sink it

Stages 1–2 establish that a reliable/unreliable decision can be made. They do
not establish that acting on it helps. The claim only lands once Stages 3–4
exist and the event-based protocol of Sec. V-H/V-I shows:

1. **Recovery success rises on recoverable events** (eq. 34), **while the false
   restoration rate does not** (eq. 36). Reporting only the first would reward
   hallucinating instrument pixels into genuinely occluded regions, and the
   report is right to insist both are reported together. This is the single
   most important experiment in the project.
2. **No regression on reliable frames** — aggregate IoU on frames with `r_t ≥ τ`
   must not fall below Baseline A. The framework guarantees this by
   construction; it still needs to be shown empirically.
3. **Baseline B (always-on temporal, gate removed) does not match the proposed
   method.** If it does, selectivity contributes nothing and Gap 1 is not
   substantiated — the report states this explicitly as falsification criterion 1,
   and it is the comparison a reviewer will ask for first.

One honest caveat to carry into the write-up: EndoVis 2018 is sampled at 1 Hz -- verified against the official challenge paper (arXiv:2001.11190), which states frames were extracted at 1 Hz and visually similar frames then manually removed --
after the organisers **manually removed similar frames**. The curation actively
suppresses the temporal redundancy a recovery mechanism feeds on, so results on
this dataset are a **conservative lower bound** on the benefit of temporal
recovery, and recovery latency measured in released frames cannot be converted
to milliseconds. Say this in the paper before a reviewer says it for you.
