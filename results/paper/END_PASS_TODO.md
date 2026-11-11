# Collected changes for the single end-of-run pass (Tasks O, R, S)

Tasks O (site) and R (deck, paper) run once at the end. Changes found since the drafts were built are collected here and applied in one pass; Task S checks each is done.

1. **Gate calibration finding** (results/paper/gate_calibration_offset.json, tools/gate_calibration_offset.py). One finding with one candidate cause: the production gate's absolute scores are optimistic on test
   (mean score 0.648 against 0.561 achieved instrument IoU, offset +0.087; out of fold the offset is -0.020) and the conformal budget met out of fold (0.048 against 0.05) is missed on test (0.077). Both are
   consistent with the test sequences being harder than the training sequences the gate and threshold were fitted on; that cause is not tested. Ranking is moderate, not strong: AUROC on the official-IoU risk
   0.740 out of fold and 0.771 on test (interval 0.449 to 0.928), 0.788 and 0.844 on the gate's own label. State in the paper (Discussion and limitations) and on /limits/: ranking usable, absolute values not
   trustworthy. Figure 6 shows it. Do not quote 0.899 as the gate's AUROC (that figure belongs to a different detector row).
2. **Baselines** (P.1): results/taskP/baselines.json -> figures 3, 4, table baseline_comparison, deck slide 9 bullet, paper Results subsection, /bench/. Include the training-safeguard note (P1_PROTOCOL.md).
3. **EndoVis 2017 zero-shot** (P.3 B): results/taskP3/endovis17_zeroshot.json -> paper dataset section (2017 is an evaluation-only test, never trained on), Results, deck, site, NUMBERS.
4. **Promote** cvavish/main_new.tex over cvavish/main.tex (the owner's original is already saved as cvavish/main_original.tex; verify before overwriting). Typo pass, cite check, tectonic twice, page count.
5. **Deck**: slide 8 line "Frame-level intervals" slightly past the border; slide 9 baselines bullet; leave the three slide-1 placeholders.
6. **Figure 6a** (segmenter pixel reliability) needs tools/calibration_pixels.py on the GPU after training.
7. **DeepLabV3+ is absent** (four unstable runs, see RUNLOG and P1_PROTOCOL.md). Every place that mentions "U-Net, DeepLabV3+, SegFormer-B0" (deck slide 9 and any methodology text, paper, site /bench/) must say two baselines, name DeepLabV3+ as not finished and give the reason. The deck's "baselines pending" bullet must go.
8. The zero-shot smoke tests (first 6 and first 2 frames of each 2017 sequence) are disclosed in RUNLOG; say so in the paper's EndoVis 2017 subsection.
