# References

Every source behind `SOLUTION_V3.md` and `SOLUTION_V4.md`. 51 works, grouped by the
question they were read to answer. Retrieval status is marked honestly:

- **[full]** full text retrieved and read (arXiv HTML/PDF, ar5iv, CVF, PMC, publisher HTML)
- **[partial]** abstract, metadata and repository only — paywalled or blocked
- **[secondary]** numbers taken from another paper's table, not first-party verified

Where a venue is not stated on the paper itself, that is said rather than guessed.

---

## A. The benchmark and its official protocol

**A1 [full]** Allan, M., Kondo, S., Bodenstedt, S., Speidel, S., et al. (40+ authors).
*2018 Robotic Scene Segmentation Challenge.* arXiv:2001.11190 [cs.CV], 30 Jan 2020
(rev. 3 Aug 2020). https://arxiv.org/abs/2001.11190
The source of record for: 19 sequences / 15 train + 4 test; stereo SXGA 1280x1024 with
only the left eye labelled; "subsampled to 1 Hz"; "similar frames were manually removed
until the sequence contained 300 frames"; the metric ("arithmetic mean of the IoU for
all classes that are present in a given frame", frames with no class discounted); the
18-team leaderboard (winner OTH Regensburg 0.621; test-set-4 team average 0.279).

**A2 [full]** Allan, M., et al. *2017 Robotic Instrument Segmentation Challenge.*
arXiv:1902.06426. https://arxiv.org/abs/1902.06426
Used only to establish that EndoVis 2017 is sampled at 2 Hz, which makes the
2017-vs-2018 comparison in section C a controlled one.

**A3 [full]** The shipped official evaluator, `test_data/test_data/run.py` and
`utils.py`, read line by line. Establishes that background is skipped
(`if val.classid == 0: continue`), that a class counts only if present in the
**ground truth** (`if (ground_truth_not_overlap + overlap) == 0: return None`), and
therefore that a hallucinated class is not penalised at all.

---

## B. EndoVis 2018 segmentation — single-frame and general

**B1 [full]** Jin, Y., Yu, Y., Chen, C., Zhao, Z., Heng, P.-A., Stoyanov, D.
*Exploring Intra- and Inter-Video Relation for Surgical Semantic Scene Segmentation.*
IEEE TMI, 2022. arXiv:2203.15251. https://arxiv.org/abs/2203.15251
Code: https://github.com/YuemingJin/STswinCL (no weights)
The only paper that states its metric explicitly: "only the classes present in the
given frame are counted", "the background class is excluded". 63.6 mIoU / 72.0 Dice.
The de-facto protocol anchor.

**B2 [partial]** Liu, M., Han, Y., Wang, J., Wang, C., Wang, Y., Meijering, E.H.W.
*LSKANet: Long Strip Kernel Attention Network for Robotic Surgical Scene Segmentation.*
IEEE TMI 43(4):1308-1322, April 2024. DOI 10.1109/TMI.2023.3335406
Code: https://github.com/YubinHan73/LSKANet (no weights; documented config path absent)
Closed access; no open copy found via Unpaywall, OpenAlex or the senior author's lab
page. Its 66.2 mIoU / 75.3 mDice is **[secondary]**, from two independent third-party
tables that agree exactly.

**B3 [full]** Yuan, C., Ban, Y. *Surgical Scene Segmentation by Transformer With
Asymmetric Feature Enhancement.* arXiv:2410.17642, Oct 2024. Venue not stated.
https://arxiv.org/abs/2410.17642 — Code: https://github.com/cyuan-sjtu/ViT-asym
Claims 77.5 mIoU. Not comparable; see the protocol note below.

**B4 [full]** Yuan, C., Ban, Y. *Temporal Propagation of Asymmetric Feature Pyramid
for Surgical Scene Segmentation.* arXiv:2504.13440, Apr 2025. Venue not stated;
partially anonymised header. https://arxiv.org/abs/2504.13440 — no code.
Claims 82.6 mIoU. Its "AFPNet" ablation row reproduces B3's four per-sequence values
exactly, so B3 is this group's internal baseline.

**B5 [full]** Ahmed, F., Abdel-Ghani, M., Arsalan, M., Ali, M., Al-Ali, A.,
Balakrishnan, S. *Surg-SegFormer: A Dual Transformer-Based Model for Holistic Surgical
Scene Segmentation.* arXiv:2507.04304, Jul 2025. NeurIPS 2025 affinity-event poster
(not a main track or workshop). https://arxiv.org/abs/2507.04304 — no code.
Merges the three instrument-part labels; input resolution never stated.

**B6 [full]** Ahmed, F.A., Arsalan, M., Al-Ali, A., Al-Jalham, K., Balakrishnan, S.
*CLIP-RL: Surgical Scene Segmentation Using Contrastive Language-Vision Pretraining
and Reinforcement Learning.* arXiv:2507.04317. IEEE CBMS 2025,
DOI 10.1109/CBMS65348.2025.00175. https://arxiv.org/abs/2507.04317 — no code.
Claims 0.81 mIoU at **224x224** input.

**B7 [full]** Abdel-Ghani, M., Ali, M., Ali, M., Ahmed, F., Arsalan, M., Al-Ali, A.,
Balakrishnan, S. *FASL-Seg: Anatomy and Tool Segmentation of Surgical Scenes.*
arXiv:2509.06159v3. DOI 10.3233/FAIA250908 (IOS Press, FAIA / ECAI series).
https://arxiv.org/abs/2509.06159 — Code: https://zenodo.org/records/16920427 (13.5 kB)
The only paper reporting both parameters (81.99 M) and GFLOPs (223.42). Uses the
correct 12-class list but **scores background as a class**.

**B8 [full]** González, C., Bravo-Sánchez, L., Arbeláez, P. *ISINet: An Instance-Based
Approach for Surgical Instrument Segmentation.* MICCAI 2020. arXiv:2007.05533.
https://arxiv.org/abs/2007.05533 — Code + weights: https://github.com/BCV-Uniandes/ISINet
Defines the split that much of the field reuses: validation = sequences **2, 5, 9, 15**,
carved out of the official *training* set.

**B9 [full]** Zhang, T., Yuan, X., Xu, H. *Surgical Instrument Segmentation via
Segment-Then-Classify Framework with Instance-Level Spatiotemporal Consistency
Modeling.* Journal of Imaging 11(10):364, 2025 (MDPI, "Communication").
DOI 10.3390/jimaging11100364. https://www.mdpi.com/2313-433X/11/10/364 — no code.
States EndoVis 2018 is "sampled at 2 Hz", which is wrong (A1 says 1 Hz).

**B10 [partial]** Liu, M., Han, Y., Wang, Y. *Multiscale dynamic visual network for the
scene segmentation of surgical robots.* Journal of Image and Graphics 30(7):2542-2557,
2025. DOI 10.11834/jig.240385. English HTML and PDF both blocked by robots.txt;
only a truncated abstract obtained. The most likely accessible route to B2's tables.

> **Protocol note.** B1/B2 and the A1 leaderboard form one comparable family
> (50-66 mIoU). B5/B6/B7 form a second (background included or unstated, resolution
> unstated or 224x224, and the same SegFormer baseline reported as 0.57, 0.57 and 0.75
> across three papers from one lab). B8/B9 and MATIS/SurgicalSAM/S3Net/TraSeTr form a
> third (7-class instrument type on the ISINet split). B3/B4 are a single-lab regime of
> their own. Rows must not be mixed across these.

---

## C. Temporal and video methods

**C1 [full]** Ayobi, N., Pérez-Rondón, A., Rodríguez, S., Arbeláez, P. *MATIS:
Masked-Attention Transformers for Surgical Instrument Segmentation.* arXiv:2303.09514v4.
Venue not stated in the text (commonly cited as IEEE ISBI 2023).
https://arxiv.org/abs/2303.09514 — Code + weights: https://github.com/BCV-Uniandes/MATIS
Temporal gain +1.89 mIoU / +5.39 mcIoU, and the gain is on *classification only* —
the temporal module never changes mask shape. Contains the field's clearest admission:
"increasing the stride value diminishes our performance due to the low sampling rate in
our datasets" (no number given).

**C2 [full]** Jin, Y., Cheng, K., Dou, Q., Heng, P.-A. *Incorporating Temporal Prior
from Motion Flow for Instrument Segmentation in Minimally Invasive Surgery Video.*
MICCAI 2019. arXiv:1907.07899. https://arxiv.org/abs/1907.07899
Code: https://github.com/keyuncheng/MF-TAPNet
States its sampling rate explicitly ("2 Hz"). Largest flow-warping gain in the
literature (+5.70 binary IoU) — on EndoVis 2017, and it transfers poorly to 2018.

**C3 [full]** Wang, J., Jin, Y., Wang, L., Cai, S., Heng, P.-A., Qin, J. *Efficient
Global-Local Memory for Real-time Instrument Segmentation of Robotic Surgical Video*
(DMNet). MICCAI 2021. arXiv:2109.13593. https://arxiv.org/abs/2109.13593
Code: https://github.com/jcwang123/DMNet
Entropy- and distance-gated global memory: an early instance of *selective* temporal
use. No EndoVis 2018 ablation; its 2018 number is 3-class part segmentation.

**C4 [full]** Wu, Z., Schmidt, A., Kazanzides, P., Salcudean, S.E. *Augmenting
Efficient Real-time Surgical Instrument Segmentation in Video with Point Tracking and
Segment Anything.* arXiv:2403.08003. Venue not stated.
https://arxiv.org/abs/2403.08003 — Code: https://github.com/wuzijian1997/SIS-PT-SAM
Names the frame rate as the cause of its one loss (6.7 FPS UCL dVRK, where
memory-based XMem beats it). No temporal ablation at all.

**C5 [full]** Liu, H., Gao, M., Luo, X., Wang, Z., Qin, G., Wu, J., Jin, Y.
*ReSurgSAM2: Referring Segment Anything in Surgical Video via Credible Long-term
Tracking.* MICCAI 2025. arXiv:2505.08581. https://arxiv.org/abs/2505.08581
Code + checkpoints: https://github.com/jinlab-imvr/ReSurgSAM2
The strongest evidence that *selection* beats *more memory*: credible-initial-frame
selection +6.14 J&F and diversity memory +3.03, while simply lengthening the
short-term memory gives **-0.02**.

**C6 [full]** Bundele, V., Hosseinzadeh, M., Lensch, H.P.A. *ReMeDI: Refined Memory for
Disambiguation of Identities with SAM3 in Surgical Segmentation.* arXiv:2512.16880
(v1 Dec 2025, v2 Mar 2026). No venue, no code.
https://arxiv.org/abs/2512.16880
**Reference correction:** this arXiv ID is *not* "Memory-Enhanced SAM3 for
Occlusion-Robust Surgical Instrument Segmentation" — check wherever that title came
from. Reliability-gated memory (tau 0.95), occlusion-triggered refill (tau 0.75).
Memory size 20 is worse than 15.

**C7 [full]** Xu, G., Shao, H.-C., Zhang, Y. *Multi-scale Temporal Sampling
Augmentation and Memory-Splitting Pruning for Promptable Video Object Segmentation and
Tracking in Surgical Scenarios* (TSMS-SAM2). arXiv:2508.05829. Venue not stated.
https://arxiv.org/abs/2508.05829 — Code: https://github.com/apple1986/TSMS-SAM2 (no weights)
The only hard number on temporal gap for EndoVis 2018: stride (1,3) costs **-6.44 J&F**
versus (1,2). "Excessive temporal gaps harm feature alignment."

**C8 [full]** Ozbulak, U., Mousavi, S.A., Tozzi, F., Rashidian, N., Willaert, W.,
De Neve, W., Vankerschaver, J. *Less is More? Revisiting the Importance of Frame Rate
in Real-Time Zero-Shot Surgical Video Segmentation.* arXiv:2502.20934. Venue not
stated. https://arxiv.org/abs/2502.20934 — no code.
**Reference correction:** retitled between versions; the search index also lists
"Revisiting the Evaluation Bias Introduced by Frame Sampling Strategies in Surgical
Video Segmentation Using SAM2". Cite the version read.
The controlled frame-rate study: streaming grasper IoU 30.8 at 1 FPS versus 74.4 at
25 FPS. Critically, it shows that *sampled-frame* evaluation flatters low frame rates
while *streaming* evaluation exposes them. It does not measure consecutive-frame GT IoU.

(B1, B4, B8, B9 are also temporal and are not repeated here.)

---

## D. Stereo, multi-view and surgical depth

**D1 [full]** Psychogyios, D., Mazomenos, E., Vasconcelos, F., Stoyanov, D. *MSDESIS:
Multi-task stereo disparity estimation and surgical instrument segmentation.*
IEEE TMI. https://pmc.ncbi.nlm.nih.gov/articles/PMC7613770/
Fuses the two views to *improve* the mask. Never measures left-right disagreement.

**D2 [full]** Mohammed, A., Yildirim, S., Farup, I., Pedersen, M., Hovde, Ø.
*StreoScenNet: surgical stereo robotic scene segmentation.* SPIE Medical Imaging 2019.
The one challenge-adjacent method that uses the right view — by feature summation, with
zero geometry. Found that adding the second view **hurt**: Dice 87.86 versus monocular
VGG19's 90.01.

**D3 [full]** *SEGSRNet for Stereo-Endoscopic Image Super-Resolution and Surgical
Instrument Segmentation.* arXiv:2404.13330. https://arxiv.org/abs/2404.13330
Again fusion, not auditing.

**D4 [full]** Yang, G., Zhao, H., Shi, J., Deng, Z., Jia, J. *SegStereo: Exploiting
Semantic Information for Disparity Estimation.* ECCV 2018.
The closest prior art to the idea in `SOLUTION_V4.md`: it computes semantic features
for **both** views and warps right to left — but uses it as a training regulariser for
disparity and never measures the disagreement.

**D5 [full]** Hu, X., Mordohai, P. *A Quantitative Evaluation of Confidence Measures
for Stereo Vision.* IEEE TPAMI 34(11), 2012.
Left-right consistency as a **disparity** confidence measure. Explicitly about
disparity, not segmentation — the distinction that our bug collapsed.

**D6 [full]** Haeusler, R., Nair, R., Kondermann, D. *Ensemble Learning for Confidence
Measures in Stereo Vision.* CVPR 2013. https://openaccess.thecvf.com/content_cvpr_2013/papers/Haeusler_Ensemble_Learning_for_2013_CVPR_paper.pdf
Confidence in the disparity map, not in a segmentation mask.

**D7 [full]** *StereoMamba* (surgical stereo matching).
The load-bearing warning for our design: specular reflections from instruments break
stereo matching across GwcNet, ACVNet, RAFT-Stereo and IGEV-Stereo. Any disagreement
computed through *learned or photometric* disparity is contaminated exactly where tools
are shiniest — which is what our Task 2 measured independently (instrument left-right
consistency 45.6% versus tissue 65.2%).

**D8 [full]** *LACOSTE* (surgical instrument segmentation with pseudo-stereo).
Substitutes a learned flow network for the real calibration and **randomises**
pseudo-baselines. Does not use EndoVis 2018's calibration.

**D9 [full]** *MT3DNet* (surgical scene understanding with monocular depth).
Discards the right view and estimates monocular depth instead, on a stereo dataset.

**D10 [partial]** *EISegNet.* IEEE JBHI, 2025. DOI 10.1109/JBHI.2025.3600291
Closed access. Flagged by the stereo review as the work most likely to overlap the
cross-view idea; worth institutional access before submission.

**D11 [full]** Wang, Y., Chen, Y., Wang, W., et al. *Conflict-Based Cross-View
Consistency for Semi-Supervised Semantic Segmentation.* CVPR 2023.
https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Conflict-Based_Cross-View_Consistency_for_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf
"Cross-view" here means augmented views of one image, at training time. Not cameras,
not test time, not failure detection. The name collision is worth addressing explicitly
in our related-work section.

**D12 [partial]** Xia, Y., Yang, D., Yu, Z., et al. *Uncertainty-aware multi-view
co-training for semi-supervised medical image segmentation.* WACV 2020 / MedIA 2020.
arXiv:1811.12506. https://arxiv.org/abs/1811.12506
CVF PDF returned 403. "Views" are axial/coronal/sagittal reslices of one volume — the
closest analogue to a second viewpoint in the literature, and still not a second sensor.

**D13 [full]** *Every View Counts: View-Consistent Panoptic Quality.*
Treats view consistency as an evaluation metric computed **with** ground truth, not as
a ground-truth-free test-time signal.

---

## E. Failure detection and segmentation quality without ground truth

**E1 [full]** Rottmann, M., Colling, P., Hack, T.-P., Chan, R., Hüger, F., Schlicht, P.,
Gottschalk, H. *Prediction Error Meta Classification in Semantic Segmentation: Detection
via Aggregated Dispersion Measures of Softmax Probabilities* (MetaSeg).
arXiv:1811.00648. https://arxiv.org/abs/1811.00648
The origin of hand-crafted segment-wise quality features (15 scalars + class vector).
AUROC up to 91.9%, R^2 84.2%, one forward pass. Flag only, no abstention.

**E2 [full]** Maag, K., Rottmann, M., Gottschalk, H. *Time-Dynamic Estimates of the
Reliability of Deep Semantic Segmentation Networks.* IEEE ICTAI 2020. arXiv:1911.05075.
https://arxiv.org/abs/1911.05075
The video version of E1. Gradient boosting won both classification and regression on
both datasets, and also overfitted with limited data. Longer time series do **not**
monotonically help.

**E3 [full]** Zenk, M., Zimmerer, D., Isensee, F., Traub, J., Norajitra, T., Jäger, P.,
Maier-Hein, K. (DKFZ). *Comparative Benchmarking of Failure Detection Methods in Medical
Image Segmentation: Unveiling the Role of Confidence Aggregation.* arXiv:2406.03323v2,
2024. https://arxiv.org/abs/2406.03323
**The key reference for our design.** Six datasets, shift-aware. Two findings we build
on: (i) the best detector is **disagreement between discrete masks** (pairwise Dice over
a 5-model ensemble; MC-dropout pairwise Dice second) — the only two ranked consistently,
with an AURC gap over single-network mean entropy exceeding 0.1 on two datasets;
(ii) **AURC from risk-coverage analysis is the right headline metric**, because AUROC
needs an arbitrary binary failure label, MAE needs confidence calibrated onto the risk
scale, and correlations ignore segmentation performance. Also: Mahalanobis ranks first
for OOD detection but **not** for failure detection — the two tasks are different.

**E4 [full]** Wang, G., Li, W., Aertsen, M., Deprest, J., Ourselin, S., Vercauteren, T.
*Aleatoric uncertainty estimation with test-time augmentation for medical image
segmentation with convolutional neural networks.* Neurocomputing, 2019.
arXiv:1807.07356. DOI 10.1016/j.neucom.2019.01.103. https://arxiv.org/abs/1807.07356
Test-time augmentation as aleatoric uncertainty; structure-level volume variation
coefficient. 20 passes (2D) / 40 (3D). Notably **not benchmarked by E3**.

**E5 [full]** Borges, B.L.C., Pacheco, B.M., Silva, D. *A Near-Optimal Confidence
Estimator for Selective Prediction in Semantic Segmentation* (Soft Dice Confidence).
Machine Learning (Springer), DOI 10.1007/s10994-026-07096-w. arXiv:2402.10665v4.
https://arxiv.org/abs/2402.10665
Lowest AURC in all 8 columns across 11 medical datasets **including polyp endoscopy**,
from one forward pass, closed form, no tuning data. Binary segmentation only. The
authors deliberately exclude ensembles and MC-dropout, so E5-vs-E3 has never been run.

**E6 [full]** Pham, H.D., Cao, D.P.M., Huynh, T.T. *Beyond Uncertainty: Generalizable
Failure Monitoring for Surgical Segmentation under Acquisition Degradation*
(TCSR-Monitor). arXiv:2608.16748. No venue stated.
https://arxiv.org/abs/2608.16748 — Code: https://github.com/dinhieufam/tcsr-monitor
The closest prior work to our Stage 2 and the only surgical-video failure monitor
found: 22 hand-crafted features (confidence 57.9% of gain importance, image quality
20.6%, shape 16.5%, **temporal only 5.0%**) into XGBoost with conformal thresholds,
0.48 ms/frame, zero extra segmenter passes. Clean AUROC 0.877 versus entropy 0.764 and
max-softmax 0.536. Treat cautiously: EndoVis **2017**, synthetic corruptions, the
sequence-level 95% CI is [0.48, 0.99] from 3 test sequences, the external CholecSeg8k
audit produced no valid number, and entropy **beat** the monitor on SAM2 (0.984 vs 0.896).

**E7 [full]** Maganti, S. *Rethinking Uncertainty in Segmentation: From Estimation to
Decision.* arXiv:2604.13262v1, Apr 2026. Preprint, sole author.
https://arxiv.org/abs/2604.13262
TTA (6 passes) beats MC-dropout (30 passes) decisively on uncertainty AUROC,
0.881 versus 0.722. Ranking generalises across datasets better than accuracy does
(uncertainty AUROC stays > 0.83 while Dice drops 6.5 points) — the single most
encouraging result for a gate under distribution shift.

**E8 [full]** Moon, S., Liu, Q., Kong, H., Khan, M.H. *Judging from Support-set: A New
Way to Utilize Few-Shot Segmentation for Segmentation Refinement Process* (JFS).
arXiv:2407.04519v3. https://arxiv.org/abs/2407.04519
The **only** method that decides whether to keep a correction — and it is strictly
post-hoc (must run the refiner first), its decision accuracy on a representative random
sample is **52.1%** (chance), it buys +0.7 mIoU over refining blindly, and it needs a
labelled support pair at inference time.

**E9 [full]** de Langlais, A., Billot, B., Aguilar Vidal, T., Gauci, M.-O.,
Delingette, H. *SCORE: Resource-Efficient Automatic Refinement of Segmentations via
Weak Supervision from Light Feedback.* arXiv:2511.02576v1.
https://arxiv.org/abs/2511.02576
A current medical refiner that corrects **every** input with no gate, no confidence
estimate and no abstention, and reports only dataset-level means with no degradation
or failure-case analysis. The counter-example our gate is argued against.

**E10 [full]** Kujur, A. *Beyond Accuracy: Uncertainty-Guided Boundary Refinement for
Reliable Biomedical Image Segmentation.* arXiv:2609.12892v1, Sep 2026. Preprint.
https://arxiv.org/abs/2609.12892
Gates **where** to correct inside an image, never **whether** to correct the image.
Also reports that calibration does not automatically improve after refinement.

**E11 [full]** Shen, C., Liu, Y. *Conformal Segmentation in Industrial Surface Defect
Detection with Statistical Guarantees.* arXiv:2504.17721.
https://arxiv.org/abs/2504.17721
Conformal risk control on a per-image loss. The guarantee is **marginal — in expectation
over test images, not per image — and requires exchangeability**, which video violates.
Directly relevant to our Stage 2 conformal claim.

**E12 [full]** Valindria, V.V., Lavdas, I., Bai, W., Kamnitsas, K., Aboagye, E.O.,
Rockall, A.G., Rueckert, D., Glocker, B. *Reverse Classification Accuracy: Predicting
Segmentation Performance in the Absence of Ground Truth.* IEEE TMI, 2017.
arXiv:1702.03407. http://www.doc.ic.ac.uk/~bglocker/pdfs/valindria2017tmi.pdf
The foundational no-ground-truth quality predictor. ~660 s per image.

**E13 [full]** Robinson, R., Oktay, O., Bai, W., Valindria, V., et al., Rueckert, D.,
Glocker, B. *Real-time Prediction of Segmentation Quality.* MICCAI 2018.
arXiv:1806.06244. https://arxiv.org/abs/1806.06244
A small CNN on (image + one-hot mask) regressing Dice: MAE 0.031, 96.5% accuracy, at
~40 ms GPU versus E12's ~660 s. A ~16,500x speedup for equal-or-better accuracy, so
the reference-database route is empirically dominated.

**E14 [full]** Fournel, J., Bartoli, A., Marchi, F., Maurin, A., Arjomand Bigdeli, A.,
Jacquier, A., Feragen, A. *Difficulty Estimation for Image-Specific Medical Image
Segmentation Quality Control.* MICCAI 2025, LNCS 15972, pp. 118-127.
DOI 10.1007/978-3-032-05169-1_12. https://papers.miccai.org/miccai-2025/0232-Paper4042.html
The clearest existing argument that a **per-image adaptive threshold beats a single
global threshold** (~2x precision, MAE 5.6%). Caveat its own reviewers raised: it models
inter-observer variability, not AI error.

**E15 [full]** Esmaeili, M., Fernandez, M., Borges, P., Gibson, E., Ourselin, S.,
Cardoso, M.J. *A methodology for clinically driven interactive segmentation evaluation.*
arXiv:2510.09499, Oct 2025. https://www.alphaxiv.org/abs/2510.09499
The one paper whose explicit purpose is better refinement metrics. Defines nAUC and
never-reached-target; **neither counts an edit that makes the mask worse**. Describes
"degradation after early gains" qualitatively, with no number.

**E16 [partial]** Jungo, A., Balsiger, F., Reyes, M. *Analyzing the Quality and
Challenges of Uncertainty Estimations for Brain Tumor Segmentation.*
https://pmc.ncbi.nlm.nih.gov/articles/PMC7156850/
Blocked by a CAPTCHA; known only through E3's description of its radiomics
regression-forest aggregator. Worth a direct read.

**E17 [secondary]** Lakshminarayanan, B., Pritzel, A., Blundell, C. *Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017.
**E18 [secondary]** Gal, Y., Ghahramani, Z. *Dropout as a Bayesian Approximation.*
ICML 2016.
Not fetched as primary sources. Their behaviour *in segmentation* is reported through
E3 and E7, which is the more relevant evidence: ensembles = M trained models and M
passes (M=5 in E3); MC-dropout = N stochastic passes (N=10 in E3, 30 in E7).

---

## F. Components used in our own pipeline

**F1 [full]** Angelopoulos, A.N., Bates, S., Fisch, A., Lei, L., Schuster, T.
*Conformal Risk Control.* ICLR 2024. The routing threshold with a stated budget.
**F2 [full]** Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. *On Calibration of Modern
Neural Networks.* ICML 2017. Temperature scaling.
**F3 [full]** Shit, S., Paetzold, J.C., Sekuboyina, A., et al. *clDice — a Novel
Topology-Preserving Loss Function for Tubular Structure Segmentation.* CVPR 2021.
arXiv:2003.07311. https://arxiv.org/abs/2003.07311 — the thin-structure metric for
thread and needle.
**F4 [full]** *SurgeNetXL* — surgical self-supervised pretraining on 4.7 M frames.
arXiv:2501.09436. https://arxiv.org/abs/2501.09436 — the encoder initialisation.
**F5 [full]** Yu, W., Si, C., Zhou, P., et al. *MetaFormer Baselines for Vision*
(CAFormer). The backbone family vendored in `sitr/third_party/metaformer.py`.

---

## G. What none of these do

Established by targeted search, each a separate query, all negative:

1. **No paper segments both views of a surgical stereo pair and compares the
   predictions.** Clean negative across D1-D13.
2. **No paper anywhere uses left-versus-right segmentation disagreement as an
   uncertainty, quality or failure signal.** D5/D6 do it for disparity; D11/D12 use
   "views" that are augmentations or reslices; E3's winning signal is mask disagreement
   but between *ensemble members*, never between *cameras*.
3. **No paper uses EndoVis 2018's actual stereo calibration.** 1 of 18 challenge teams
   used the right view at all (D2, by feature summation). D8 randomises pseudo-baselines;
   D9 discards the right view entirely.
4. **No paper defines or reports a harmful-correction rate** — the fraction of cases a
   refiner makes worse. E9, E10 and E15 each had the opportunity and did not.
5. **No paper predicts the benefit of a correction before applying it.** E8 decides
   post-hoc at chance accuracy; E10 gates where, not whether; E9 corrects always.
6. **No paper measures ground-truth mask IoU between consecutive EndoVis 2018 frames.**
   C8 is the only controlled frame-rate study and explicitly does not analyse adjacent-
   frame overlap; it is also on CholecSeg8k, not EndoVis.
7. **No paper reports AURC on surgical video.** E3 recommends it and has no video or
   surgical data; E6 is surgical video and uses AUROC/AUPRC.
8. **No paper in section B releases trained weights.** Not one checkpoint.
9. **No paper in section B gives a complete, unambiguous mIoU definition.** B1 comes
   closest and still does not resolve "present in the frame"; the official code (A3)
   settles it as ground-truth-only, and no paper states that.
10. **No paper in section B reports variance or repeated runs**, while claiming 1-3
    point improvements on a four-sequence test set.
11. **No paper in section B evaluates at native 1280x1024**, and three never state
    resolution at all — a first-order confound for thread, needle and clips.
