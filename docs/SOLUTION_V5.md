# Solution v5 — maximising real results and novelty

Supersedes v4's sections 1, 2 and 6. v4's dataset table (section 4), edge-case register
(section 7) and compute analysis (section 5) still stand.

One sentence up front, because it is what makes the rest work: **your results are not
flat, they are diluted.** The method is measured on a population it was not designed for
and with a metric that cannot see it. Fixing both is honest, costs nothing, and changes
the headline numbers by an order of magnitude.

---

## 1. The diagnosis: why +0.0012 is the wrong number

What you report now: official mIoU 0.6128 -> 0.6141 across M1-M6. Looks like nothing
happened.

What is actually in `results/task5/selector_results.json`, on the official test set:

| On the frames the gate flags (30.4% of test) | mean IoU change |
|---|---|
| Oracle choice | **+0.1040** |
| Always apply A3 (morphological) | **+0.0621** |
| Learned selector | **+0.0370** |
| Always apply A2 (temporal) | +0.0151 |

So on flagged frames, correction is worth **+0.037 to +0.062 instrument IoU**, and a
perfect chooser would be worth **+0.104**. That is 30-85x the global number you are
reporting. The global number is small for three arithmetic reasons, none of which is a
failure of the method:

1. Only **30.4%** of frames are flagged. 70% are left untouched by design.
2. The official metric averages over **all 997 frames**, so a gain on 300 of them is
   diluted by a factor of three before anything else.
3. The official metric is a **per-frame mean over classes present in the ground truth**,
   so an improvement concentrated in one instrument class is divided by the 4-8 classes
   in that frame.

Multiply those out: +0.062 on 30.4% of frames, divided across the classes in a frame,
lands almost exactly on the +0.001 to +0.002 you measured. **Your global result is the
predictable arithmetic consequence of a selective method being scored globally.**

So report both, and say why they differ. "On flagged frames, correction improves
instrument IoU by +0.062; the official 12-class score is unchanged because flagged
frames are 30% of the test set and the official metric averages per-frame over all
classes present" is a true sentence, a strong one, and it pre-empts the obvious
reviewer question instead of waiting for it.

**The oracle gap is your best single piece of evidence.** Oracle +0.1040 against
always-A2's +0.0151 means the signal is real and large and the chooser is what is
missing — not the actions, and not the premise. That is a legitimate, quantified "there
is something here, and here is exactly how much" result.

---

## 2. Four free or cheap wins, ranked by expected gain per hour

### Win 1 — the ensemble you already trained (biggest mIoU gain available)

`work_v2/crossfit/fold{0,1,2}/best.pt` plus `work_v2/final/best.pt` are **four trained
segmenters on disk**. None of them has ever seen a test sequence. So on the test set, a
4-member deep ensemble is available for **zero training cost** — about 30 minutes of
inference.

Deep ensembles essentially always gain 1-2 mIoU points in segmentation. If that holds,
0.6128 becomes roughly **0.62-0.63 under the exactly-reproduced official protocol**,
which is the level of the best protocol-matched published result (B1's 63.6) from a
26M-parameter model on a 4 GB laptop GPU. That is a headline worth having.

**The honesty constraint, which matters and must be stated.** Each fold model trained on
10 of the 15 training sequences; the final model trained on all 15. For the **test set**
all four are valid, because no test sequence was in any training set. For
**out-of-fold evaluation on training sequences** only one model per sequence is honest
(the one whose fold excluded it), so a clean out-of-fold ensemble does not exist. Report
the ensemble as its own row on test, keep every out-of-fold decision on the single
model, and say so explicitly. An ensemble row and a single-model row are two different
systems and must be labelled as two systems.

### Win 2 — the baseline table you are missing (biggest presentational gain)

Right now you compare M1-M6 against each other, which are all within 0.0013 of one
another, so every comparison looks like noise. **You have a strong gate — test AUROC
about 0.84 — and nothing to show it against.** The literature's numbers on surgical
video are max-softmax 0.536 (chance) and entropy 0.764 (E6). Add the standard failure
detectors as baselines and the table fills with real wins:

| Baseline | Cost | Availability | Why it belongs |
|---|---|---|---|
| Max-softmax | free | OOF + test | Scores 0.536 in E6 — the floor |
| Mean predictive entropy | free | OOF + test | The "naive but commonly used" baseline E3 names |
| **Soft Dice Confidence** (E5) | **free, closed form, 1 pass, no tuning data** | OOF + test | Lowest AURC in all 8 columns across 11 medical datasets including polyp endoscopy. **Never tested on surgical instruments** — closes gap G14 |
| **TTA, 6 passes** (E7) | ~25 min inference | OOF + test | E3 never benchmarked TTA at all. E7: TTA 0.881 beats 30-pass MC-dropout 0.722. Closes gap G13 |
| **Pairwise Dice over the 4-model ensemble** (E3's winner) | free (models exist), 4 passes | **test only** | E3's best-in-class detector across 6 medical datasets. Gold standard |
| MC-dropout, 10 passes | ~40 min | OOF + test | E3's second-best; include for completeness |
| **Cross-view disagreement (ours)** | **1 extra pass, already cached** | OOF + test | The claim |

This is the most important experiment left in the project. It converts "our gate scores
0.84" into "our gate beats every standard detector, and matches the 4-model ensemble at
a quarter of its cost." If cross-view lands near ensemble pairwise-Dice, that is the
efficiency result the whole design predicts, measured against the field's gold standard.

### Win 3 — mcIoU and the rare classes (biggest headline-metric gain)

Suturing-needle IoU is **0.000** and needle is in 9.5% of test frames. mcIoU averages
over 11 active classes equally, so moving needle from 0 to even 0.25 adds
**0.25 / 11 = +0.023 mcIoU** — about **19x the entire correction effect you measured on
mIoU**. Thread (0.144% of test pixels) and clamps (0.025%) are the same story. The
numbers in this project are in the rare classes, and they are reachable:

1. **clDice loss** on thread and needle (F3) — built for exactly these thin tubular
   structures. Current thin-class clDice: shaft 0.901, thread 0.764, needle **0.000**.
2. **Copy-paste augmentation of the 12 real annotated needle instances** onto other
   training frames, with their real masks. Real annotated pixels relocated, training
   only, never on any evaluation path. Describe it as augmentation, never as synthetic
   data.
3. Needle also appears in the **right view** of those 12 frames, so warping the left
   label across where the occlusion mask is clean doubles the available instances. Keep
   as an ablation, not the default.

### Win 4 — segmenter-side test-time augmentation

Multi-scale and horizontal-flip TTA on the segmenter itself is standard practice and
reliably worth +0.5 to +1.5 mIoU. You are currently single-scale. It also produces the
TTA uncertainty baseline in Win 2 from the same inference run — one job, two gaps.

---

## 3. Which research gaps this closes

Keys from `RESEARCH_GAPS.md`.

| Gap | Closed by | Status after v5 |
|---|---|---|
| G1 consecutive-frame GT IoU | already measured | **Closed**, first measurement on this dataset |
| G2 the 2 Hz -> 1 Hz attribution | already measured | **Closed**, with three corroborating sources |
| G3 harmful-correction rate | already measured, both denominators | **Closed**, first anywhere |
| G4 complete mIoU definition | evaluator + prose definition + G12 ablation | **Closed** |
| G5 sequence IDs | state ours | **Closed** |
| G6 released weights | release the four checkpoints | **Closed** — and you would be the only one |
| G7 why disparity stereo fails | already measured, corroborated by D7 | **Closed**, negative result with mechanism |
| **G8-G11 cross-view / stereo** | Prompt 9 tasks 9.1-9.2 | **The novelty claim**, pass/fail rule fixed |
| G12 metric-implementation ablation | you hold both implementations | **Closeable, nearly free** |
| G13 TTA as a surgical-video baseline | Win 2 + Win 4 | **Closeable** |
| G14 Soft Dice Confidence in this domain | Win 2 | **Closeable, free** |
| G15 SDC vs ensemble pairwise-Dice | Win 2 — you can run both | **Closeable**, first head-to-head |
| G16 AURC on surgical video | already computed | **Closed** |
| G17 variance / repeated runs | re-seed the gate learner; 4 segmenters give spread on test | **Partially closeable** |
| G18 per-sequence / per-class difficulty | you have both breakdowns | **Closeable, free** |
| G19 bridging the two task definitions | report 12-class and the B8-split type task | **Closeable** if you run the second protocol |
| G20 label noise from `repairs/` | Prompt 9 task 9.8 | **Closeable, free** |
| G21-G23 predicted-benefit gate, 3-way decision, selective temporal | measured and bounded, not closed | **Report as bounded** — the oracle gap quantifies the headroom |
| G24-G28 cross-centre, native-resolution training, conformal under correlation | out of reach on this dataset and GPU | **Stated as limitations** |

That is **11 gaps closed, 7 more closeable cheaply, 3 bounded with numbers, 5 stated as
limitations.** Eighteen contributions, of which one needs the novelty test to pass.

---

## 4. The novelty, stated for a reviewer

**Working title.** *Cross-View Agreement for Ground-Truth-Free Failure Detection in
Robotic Surgical Scene Segmentation*

**The one-line claim.** The second camera of a surgical stereo pair is a free ensemble
member for detecting segmentation failure — but only for detection, never for
correction, and the reason is geometric and measurable.

**Four first-of-kind claims, each with the clean negative behind it:**

1. **First use of a second physical viewpoint as an ensemble member for failure
   detection** (G8). Surgical stereo work fuses the views to improve the mask (D1, D2,
   D3); "cross-view consistency" in the wider literature means augmentations of one
   image (D11) or reslices of one volume (D12); stereo confidence work targets the
   disparity map (D5, D6).
2. **First use of left-versus-right segmentation disagreement as an uncertainty signal**
   (G9). D4 computes semantic features for both views and warps them, but only as a
   training regulariser for disparity, and never measures the disagreement.
3. **First use of EndoVis 2018's actual stereo calibration** (G10). 1 of 18 challenge
   teams touched the right view at all, by feature summation with zero geometry; one
   recent method randomises pseudo-baselines instead; another discards the right view
   and estimates monocular depth on a stereo dataset.
4. **First harmful-correction rate reported for a segmentation refiner** (G3), with both
   denominators distinguished.

**The mechanism, which is the actual intellectual contribution.** Rectification alone
gives row correspondence; disparity is only needed for column correspondence. So
per-row and set statistics on the two predicted masks compare across views with no
photometric matching, and are immune to the specular failure that destroys disparity on
instruments. Detection needs only rows. Pixel-level correction needs columns. **The
second camera is therefore a good auditor and a bad corrector, and that asymmetry is
measured, not asserted** — 45.6% left-right disparity consistency on instrument pixels
against 65.2% on tissue, independently corroborated by D7 across four modern learned
stereo matchers, and it explains D2's otherwise puzzling negative result.

**Pre-empt the three attacks a reviewer will make:**

- *"Cross-view consistency already exists."* It means augmented views of one image, at
  training time, for semi-supervised learning (D11). Name it and distinguish it in the
  related work, do not wait to be asked.
- *"StreoScenNet showed the second view hurts."* It does, for fusion — which is the
  corrector role. Your asymmetry result explains their number rather than contradicting
  it. This strengthens you.
- *"Your mIoU does not improve."* Answer with section 1's arithmetic, the flagged-frame
  numbers, and the oracle gap. A selective method scored globally is diluted by design,
  and you can show the multiplication.

---

## 5. The main results table, as it should look

Three tables, not one. Most of your current difficulty is that one table is being asked
to carry three different claims.

**Table 1 — segmentation, official protocol, stated in full.** Rows: single model;
+ segmenter TTA; + 4-model ensemble. Columns: mIoU, mcIoU, per-class IoU, boundary F1,
clDice on shaft/thread/needle. Then the protocol-matched published band (B1 63.6, B2
66.2 second-hand, SegNeXt 64.3) in a **separate block with a protocol column**, never
interleaved.

**Table 2 — failure detection (the novelty).** Rows: max-softmax, mean entropy, Soft
Dice Confidence, MC-dropout pairwise Dice, TTA, ensemble pairwise Dice, confidence
indicators only, + temporal, **+ cross-view (ours)**. Columns: AUROC, **AURC**, forward
passes, trained models, ms/frame. Sequence-clustered intervals throughout.

**Table 3 — selective correction.** Rows: keep-all, always-A2, always-A3, always-fuse,
blind fusion every frame, selector, oracle. Columns: activation %, mean IoU change **on
flagged frames**, mean IoU change over all frames, harmful % of corrected, harmful % of
all, hallucination rate, latency. The oracle row is the headroom and belongs in the
paper.

---

## 6. One guardrail, because it protects the result

Everything above makes the numbers look better by measuring the right population with
the right metric and adding the baselines you are entitled to beat. None of it requires
touching a number. That matters practically, not just ethically: the fastest way to lose
this paper is a reviewer noticing a protocol mismatch, and the three-family problem in
`REFERENCES.md` means that is the single most likely attack. So:

- Report mIoU **and** mcIoU **and** the flagged-frame numbers together, always. Choosing
  AURC and mcIoU as headlines is legitimate because E3 argues for AURC explicitly and
  mcIoU is where the effect lives — but drop neither of the others, or the choice stops
  being legitimate.
- Never put a number of yours in the same column as a published number without a
  protocol column. B3/B4's 77.5 and 82.6 are not comparable to anything.
- Keep every out-of-fold decision out-of-fold, and label the ensemble row as a different
  system from the single-model row.
- Report the negative results — disparity stereo, the difficulty index, the selector
  losing to always-A3. They cost nothing and they are what make a reviewer trust the
  positive ones.

The strongest version of this paper is the honest one, because the honest version has
eighteen contributions and a mechanism, and the dishonest version has one fragile number.
