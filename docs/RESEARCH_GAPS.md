# Research gaps across the 51 papers

Citation keys are from `REFERENCES.md`. Each gap says how it was established — a clean
negative across a set of papers, a targeted search that returned nothing, or a quote.

**The structural gap, first, because the rest sit inside it.**

Two literatures exist and they do not touch.

- The **EndoVis 2018 literature** (A1, B1-B10, C1-C7) is about architecture. Across all
  ten EndoVis 2018 papers, **not one** quantifies uncertainty, calibration, abstention
  or selective prediction. No predictive entropy, no ensembles, no MC-dropout, no
  reliability diagram, no risk-coverage curve. B6 uses reinforcement learning for mask
  refinement, not for uncertainty.
- The **failure-detection literature** (E1-E18) is about knowing when you failed, and
  has **almost no surgical video**. E3, the field's reference benchmark, covers six
  medical datasets and **none is video and none is surgical**. E5 reaches endoscopy, but
  polyps, and binary. E6 is the only surgical-video failure monitor found, it is on
  EndoVis **2017** with synthetic corruptions, it has no stated venue, its external
  validation produced no valid number, and its confidence interval is [0.48, 0.99] from
  three test sequences.

So: the dataset with the stereo pair, the calibration and the hard visibility conditions
has no failure-detection work on it, and the failure-detection field has no surgical
video. Your project sits exactly in that hole. Everything below is a consequence.

---

## Tier 1 — gaps your work already fills, measured

**G1. Nobody measures ground-truth mask overlap between consecutive EndoVis 2018 frames.**
C8 is the only controlled frame-rate study in surgical video and explicitly does not
analyse adjacent-frame overlap (and is on CholecSeg8k, not EndoVis). Nobody else
measures it at all. You have 0.30-0.45 instrument IoU, 60-77% of consecutive pairs below
0.5. This is the missing quantitative explanation for a pattern four papers each
explained away differently: B1 "distant frames showing different appearances from large
surgical motion", C1 "the low sampling rate in our datasets" (no number given), B8 "the
higher variability of the 2018 dataset", C7 "excessive temporal gaps harm feature
alignment".

**G2. The 3x drop in temporal benefit from 2 Hz to 1 Hz is in the literature and nobody
attributes it to frame rate.** B8 is the controlled comparison — identical module,
+12.72 cIoU on EndoVis 2017 versus +4.20 on EndoVis 2018 — and blames dataset
variability. C2 states its rate is 2 Hz; A1 states 2018 is 1 Hz; nobody connects the
two. Corroborated from three independent directions (C7's -6.44 J&F from one extra
stride, C1's untabulated stride claim, C8's 1-FPS collapse from 74-94 to ~30 IoU).

**G3. No paper defines or reports a harmful-correction rate.** Targeted searches for
"degradation rate", "worsened", "negative transfer" and harmful-refinement phrasings
returned nothing. E9 (a 2025 medical refiner) reports only dataset-level means and has
no degradation analysis. E10 reports only net gains. E15 — the one paper whose stated
purpose is better refinement metrics — defines nAUC and never-reached-target, and
**neither counts an edit that makes the mask worse**; it describes "degradation after
early gains" qualitatively, with no number. E8 comes closest and its worst subset is
selected by ground-truth outcome, so it is an illustration, not a rate. The whole
refinement literature reports means, and a mean improvement is compatible with a large
minority of cases getting worse.

**G4. No paper in the EndoVis 2018 set gives a complete, unambiguous mIoU definition.**
B1 comes closest ("only the classes present in the given frame are counted", "the
background class is excluded") and still never says whether "present" means present in
the ground truth or in the union with the prediction. Everyone else says nothing, says
"the official protocol" in one unverifiable sentence, or silently scores background
(B7). A1 itself does not resolve it. **The official code does** — ground-truth-only — and
no paper states that. The one quantity the whole field ranks on is not pinned down
anywhere.

**G5. No paper states the EndoVis 2018 train/test sequence IDs for the official 15/4
split.** Not A1, not B1, not B3, not B4, not B6. The only paper naming sequence IDs
(B7: 2, 5, 9, 15) names them for the *non-official* B8 split and leaves its 12-class
split entirely undescribed. Nobody in this literature can be exactly reproduced on the
split dimension.

**G6. No paper in the EndoVis 2018 set releases trained weights.** Not one checkpoint
across all ten. B1 and B2 release real code with no weights and no data preparation;
B3 releases an undocumented fork with an empty README; B4, B5, B6 and B9 release
nothing; B7 releases a 13.5 kB zip.

**G7. Disparity-based stereo evidence does not help failure detection on this data, and
nobody had measured why.** Your 45.6% versus 65.2% left-right consistency on
instrument versus tissue pixels, independently corroborated by D7's finding that
specular instrument reflections break GwcNet, ACVNet, RAFT-Stereo and IGEV-Stereo. A
negative result with a measured mechanism, which is the useful kind.

---

## Tier 2 — the gap you are testing

**G8. Nobody has used a second physical viewpoint as an ensemble member for
ground-truth-free failure detection.** Established three ways:

- Surgical stereo papers use the second view to **improve** the mask, never to **audit**
  it. D1, D2, D3 all fuse. None computes left-right disagreement; none reports any
  failure-detection metric.
- "Cross-view consistency" in segmentation means augmented views or feature branches of
  the **same** image, as a semi-supervised training regulariser (D11). Not cameras, not
  test time, not failure detection. The name collision needs addressing explicitly in
  your related work or a reviewer will raise it.
- The closest analogue is reslicing, not stereo: D12 uses axial/coronal/sagittal
  reslices of one volume as "views" and weights them by uncertainty — synthetic views of
  a single acquisition, at training time.
- Stereo confidence work (D5, D6) estimates confidence in the **disparity map**, not in a
  segmentation mask. D13 treats view consistency as a metric computed **with** ground
  truth.

**G9. Nobody anywhere uses left-versus-right segmentation prediction disagreement as an
uncertainty, quality or failure signal.** Clean negative. D4 (SegStereo, ECCV 2018) is
the nearest miss: it computes semantic features for **both** views and warps right to
left — but only as a training regulariser for disparity, and never measures the
disagreement.

**G10. Nobody uses EndoVis 2018's actual stereo calibration.** 1 of 18 challenge teams
used the right view at all (D2, by feature summation, zero geometry). D8 substitutes a
learned flow network and **randomises** pseudo-baselines. D9 discards the right view and
estimates monocular depth instead, on a stereo dataset. The extrinsics are present for
all 19 sequences and go unused.

**G11. No paper in the EndoVis 2018 set uses the stereo pair at all** beyond D2. Half
the pixel data in the dataset is unused by the entire literature.

Why G8-G10 are well-motivated rather than merely empty: E3's central finding is that the
strongest failure detector is **disagreement between discrete masks**, not any
confidence map — pixel-confidence aggregation is weak (E6 measured max-softmax at 0.536,
chance; E5 found mean entropy can be worse than random ordering). A calibrated stereo
pair supplies that second mask for one extra forward pass instead of five trained
models, from a genuinely different physical observation, and it decorrelates
viewpoint-dependent failures (specularity, smoke, blood, tissue occlusion) that flips
and dropout leave identical.

---

## Tier 3 — open gaps you could claim cheaply, and probably should

**G12. Nobody has ablated a metric implementation**, which, given G4, is the single most
informative experiment anyone in this subfield could run. You are uniquely positioned:
you have the official implementation reproduced to 1.1e-16 **and** the background-
inclusive variant that was in your own v2 code. Showing how much each protocol choice
moves the number (background in or out, ground-truth-only or union, per-frame or
per-sequence averaging, weighted or unweighted across sequences) would quantify why one
lab reports 77.5 where the credible band is 60-66. Nearly free, and directly citable.

**G13. Test-time augmentation has never been benchmarked as a surgical-video failure
baseline.** E3 covers seven detectors and **does not benchmark TTA at all**. E7 found
TTA at 6 passes beats MC-dropout at 30 passes decisively on uncertainty ranking (0.881
versus 0.722) — cheaper *and* better. You have the model; this is the fair comparison
for cross-view, since both are cheap substitutes for a real ensemble. Running it makes
your claim much harder to attack.

**G14. Soft Dice Confidence has never been tested on surgical instruments.** E5 has the
lowest AURC in all 8 columns across 11 medical datasets including polyp endoscopy, from
one forward pass, closed form, **no tuning data needed**. It is the cheapest strong
baseline in existence and nobody has run it in your domain. Add it.

**G15. E5 versus E3's winner has never been run.** The leading cheap detector (soft Dice
confidence) and the leading expensive one (pairwise Dice over an ensemble) have never
been compared — E5 deliberately excluded ensembles. Your stereo pair is a two-member
ensemble, so you are in a natural position to run the three-way comparison.

**G16. Nobody reports AURC on surgical video.** E3 recommends AURC and has no video or
surgical data; E6 is surgical video and uses AUROC, AUPRC, miss-rate and false-alarm
rate. You already compute AURC (0.3076 to 0.3063), so this is a sentence away.

**G17. No paper in the EndoVis 2018 set reports variance or repeated runs**, while
claiming 1-3 point improvements on a four-sequence test set. Every number in the
literature is a single run with no seeds, error bars or confidence intervals. Your gate
learner is cheap to re-seed even if the segmenter is not — partial coverage of this gap
is better than none, and sequence-clustered intervals already put you ahead.

**G18. No paper does per-sequence or per-class difficulty analysis of its own failures.**
Test sequence 4 is catastrophically hard — the 18-team average is 0.279 against 0.478
overall — and no paper investigates why or reports which classes break down there. The
organisers' only note is that most of the kidney surface is covered. You have per-class
and per-sequence numbers already.

**G19. Nobody reconciles the two task definitions.** No paper reports both the 12-class
scene result on the official test sequences *and* the 7-class instrument-type result on
the B8 split, so the two literatures have never been bridged — yet several papers
compare across them anyway. Reporting both, clearly separated, would be a service.

**G20. Nobody quantifies label noise.** The 7 masks in `repairs/` changed 2-26% of their
frames, so annotation error of that magnitude is documented in this dataset, and no
paper in the reference list examines how much of a model's apparent "failure" is really
annotation error. Free once out-of-fold predictions exist, which you have.

---

## Tier 4 — real gaps that need more than you have

**G21. Predicting the *benefit* of a correction before applying it.** E8 is the only
method that decides whether to keep a correction and it is strictly post-hoc — it must
run the refiner first — with 52.1% decision accuracy on a representative sample
(chance), buying +0.7 mIoU over refining blindly, and needing a labelled support pair at
inference. E10 gates **where** to correct, never **whether**. E9 corrects everything.
E14 predicts a per-image threshold but for accept-versus-discard, and it models
inter-observer variability rather than model error. Nobody regresses expected IoU gain
and gates on it. **You attempted this** — your selector reaches Pearson +0.46 on fusion
and +0.33 on A2, but -0.03 on A1 and +0.14 on A3, and it loses to simply always applying
A3. So this gap is real, you have evidence about it, and you have not closed it. Report
it that way: the gap stays open and you have bounded how hard it is.

**G22. A three-way accept / correct / abstain decision under one joint evaluation.**
E3 and E5 do accept-or-abstain. E9 and E10 correct always. Nobody does all three under a
single risk-coverage-style evaluation. You are closest of anyone, and whether you can
claim it depends on the selective policy actually beating its baselines — which it
currently does not.

**G23. Selective temporal correction for automatic 12-class scene segmentation.** The
principle is proven to pay: C5's credible-initial-frame selection +6.14 J&F and
diversity memory +3.03, C6's reliability-gated memory +3.61 mcIoU, C3's entropy-gated
global memory +5.13 — all larger than any uniform temporal gain on EndoVis 2018. And
C5 shows that *more* memory without selection buys exactly **-0.02**. But every one of
these gates the memory **write**, never the output **correction**, and all are on
promptable or referring tracking with a ground-truth first-frame prompt or a text
prompt. B8 is the only non-promptable method whose temporal correction is conditional,
and there the gate is a byproduct of instance matching rather than a designed decision.
**Nobody decides per frame whether to apply temporal correction based on a measured
inter-frame similarity signal.** Open, and your 1 Hz measurement explains why it is
harder here than the promptable-tracking results suggest.

**G24. Real cross-centre surgical generalisation of a learned failure monitor.** E6's
only external check (CholecSeg8k) produced no valid number, attributed to a label
binarisation bug and deferred; its corruptions are synthetic. Needs a second dataset,
which your "this dataset only" constraint rules out.

**G25. No paper does cross-dataset or cross-procedure generalisation.** B7 is the only
one to name this as a limitation (both its datasets are nephrectomy). Nobody tests
transfer to cholecystectomy, CholecSeg8k, SurgVisDom or real clinical video. Same
constraint.

**G26. No paper evaluates at native 1280x1024.** Resolutions run 224x224, 512x512,
640x512, 512x1024, and three papers never state resolution at all — a first-order
confound, since thread, needle and clips are a few pixels wide at native resolution.
You *evaluate* at full resolution, which is better than most, but you *train* at
512x640 because of 4 GB of VRAM, so you cannot close this one.

**G27. Correct statistics for conformal or selective guarantees under frame
correlation.** E6 concedes its conformal coverage is empirical, not certified, under
temporal dependence. E11's conformal-risk guarantee is **marginal in expectation over
test images and assumes exchangeability**, which video violates. Your Stage 2 risk is
0.077 against a 0.05 budget, i.e. the guarantee is not met on your test set — which is
the honest datapoint for this gap rather than a fix for it.

**G28. Per-image adaptive thresholds for AI failure in video.** E14 shows adaptive beats
global (~2x precision, MAE 5.6%) but on static CT and for inter-observer variability.
E6 shows global thresholds give a clinically unusable 40.4% false-alarm rate on correct
frames, and that per-stratum calibration cuts miss-rate from 50.5% to 9.3% at the price
of more alarms. Nobody has done per-image adaptive thresholding for model failure in
video.

---

## Tier 5 — "gaps" that are warnings, not opportunities

These look empty but the literature already says why. Do not walk into them.

**W1. Stereo fusion to improve accuracy.** D2 found adding the second view **hurt**
binary segmentation (Dice 87.86 versus monocular VGG19's 90.01). Your own A1 measured
+0.0001. The second camera is an auditor, not a corrector, and D7 explains the mechanism.

**W2. Longer temporal windows.** B1: 5 frames worse than 4. C1: "larger window sizes
slightly diminish performance as they introduce too much noise from far frames". C6:
memory 20 worse than 15. C5: longer short-term memory gives -0.02. B8's supplementary
sweep: **5 of 12 temporal configurations fall below the no-temporal baseline, worst
-5.43 mcIoU**, degrading monotonically with window length when the matching gate is
removed. More time is not the answer on this dataset.

**W3. A pure video transformer.** B1's VideoSwin baseline scores 57.9 mIoU against the
single-frame CNN baseline's 61.2 — losing by 3.3 points. B1 states pure transformers
overfit this small dataset.

**W4. MC-dropout.** E7: 30 passes, uncertainty AUROC 0.722, beaten by 6-pass TTA at
0.881. More expensive and worse.

**W5. A reference-database quality predictor (RCA-style).** E13 matches or beats E12 at
~40 ms versus ~660 s — roughly 16,500x faster. Empirically dominated.

**W6. A hand-crafted difficulty index.** You already tried it: Spearman ~0 out of fold,
+0.08 on test with an interval including zero. E6 independently found image-quality
features contribute 20.6% and temporal features only 5.0%, with confidence carrying
57.9%. Difficulty cues are weak on their own.

**W7. Monocular depth on a stereo dataset.** D9 does this. Real calibrated stereo exists
in the files.

**W8. Treating uncertainty as out-of-distribution detection.** E3's Mahalanobis result —
first for OOD on most datasets, **not** first for failure detection — is direct evidence
the two tasks are different.

**W9. Chasing the 77-82 mIoU numbers.** B3/B4 are a single-lab measurement regime: the
gain concentrates entirely on the two sequences hardest under the official per-frame
rule, test sequence 4 reaches 1.9-2.1x the challenge winner, B3's own table misprints
B1's sequence-3 score as 93.7 against the real 83.7, neither ablates its contributions,
and neither ships weights. Reproducing their evaluation code would be required before
any comparison means anything.

---

## Summary

- **7 gaps your work already fills** (G1-G7), all measured.
- **4 gaps it is testing** (G8-G11), with a pass/fail rule fixed in advance.
- **9 cheap additions** (G12-G20) — G12 (metric ablation), G13 (TTA baseline) and
  G14 (soft Dice confidence baseline) are the three highest-value-per-hour items left
  in this project.
- **8 gaps beyond your resources** (G21-G28), of which G21, G22 and G23 you have
  *bounded* rather than closed, which is itself reportable.
- **9 warnings** (W1-W9) where the literature already shows the path is bad.
