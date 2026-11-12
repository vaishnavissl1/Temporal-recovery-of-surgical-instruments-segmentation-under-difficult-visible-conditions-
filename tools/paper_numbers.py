#!/usr/bin/env python3
"""
Numbers used in the paper's PROSE, as LaTeX macros read from the result files (nothing is typed into main.tex).

    python -m tools.paper_numbers --out results/paper --tex cvavish/numbers.tex

Writes results/paper/NUMBERS_TEXT.md (every macro with its file and key path, one line each) and a `numbers.tex` of \\newcommand lines (letters-only names, as LaTeX requires). A number whose
file does not exist yet is defined as "--" and listed under "missing", so the paper still builds and says so instead of carrying a stale value.
"""
from __future__ import annotations

import argparse
import json
import os

D = {}


def J(p):
    if p not in D:
        D[p] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    return D[p]


def dig(d, path):
    for k in path:
        d = d[k]
    return d


T6, F6 = "results/task6/final_results.json", "results/taskF/no_a1/final/final_results.json"
PT, GAPF = "results/taskB/policy_table.json", "results/taskB/metric_gap_A3.json"
AL = "results/taskA/aligner_comparison.json"
TC, TD, TDT = "results/taskC/failure_detection_cost.json", "results/taskD/decision.json", "results/taskD/gate_comparison_test.json"
TE, TF, TFS, TFC, TFA = "results/taskE/tta_ensemble.json", "results/taskF/mciou_hallucination.json", "results/taskF/no_a1/selector_results.json", "results/taskF/cost.json", "results/taskF/a4/a4.json"
TG, TH, TI, TJ, TK, TL, TQ = ("results/taskG/information_bound.json", "results/taskH/occlusion_localisation.json", "results/taskI/metric_ablation.json", "results/taskJ/per_class_and_rigs.json",
                              "results/taskK/consecutive_frame_iou.json", "results/taskL/label_noise_summary.json", "results/taskQ/identity_headroom.json")
TP = "results/taskP/baselines.json"
PD = "results/taskR/pixel_dice.json"
DF = "results/task4/difficulty_report.json"
ZS, CALP, GCO = "results/taskP3/endovis17_zeroshot.json", "results/paper/calibration_pixels.json", "results/paper/gate_calibration_offset.json"
GB, S2, T5 = "results/task4/gate_comparison_gbm.json", "results/web_extract/stage2_gate.json", "results/task5/selector_results.json"
DET = lambda g, n, k: ["groups", g, "detectors", n, k]  # noqa: E731

# (macro, file, key path, format, scale)
SPEC = [
    # ---- data and base model
    ("TestFrames", T6, ["n_frames"], "{:d}", 1), ("OofFrames", T5, ["n_oof"], "{:d}", 1), ("RoutedFrames", T6, ["n_routed"], "{:d}", 1),
    ("BaseMiou", T6, ["models", "M1", "mIoU"], "{:.4f}", 1), ("OfficialDiff", T6, ["crosscheck_official_run_py", "abs_difference"], "{:.1e}", 1),
    ("SeqOneMiou", T6, ["models", "M1", "per_sequence", "test/seq_1"], "{:.3f}", 1), ("SeqTwoMiou", T6, ["models", "M1", "per_sequence", "test/seq_2"], "{:.3f}", 1),
    ("SeqThreeMiou", T6, ["models", "M1", "per_sequence", "test/seq_3"], "{:.3f}", 1), ("SeqFourMiou", T6, ["models", "M1", "per_sequence", "test/seq_4"], "{:.3f}", 1),
    ("MiouMSix", F6, ["models", "M6", "mIoU"], "{:.4f}", 1), ("MiouMFive", T6, ["models", "M5", "mIoU"], "{:.4f}", 1), ("MiouMFour", T6, ["models", "M4", "mIoU"], "{:.4f}", 1),
    ("MiouMTwo", T6, ["models", "M2", "mIoU"], "{:.4f}", 1), ("MiouMThree", T6, ["models", "M3", "mIoU"], "{:.4f}", 1),
    ("BaseMcIoU", TF, ["mcIoU", "M1", "mcIoU"], "{:.4f}", 1), ("McIoUMFive", TF, ["mcIoU", "M5", "mcIoU"], "{:.4f}", 1), ("McIoUMTwo", TF, ["mcIoU", "M2", "mcIoU"], "{:.4f}", 1),
    ("ClassesScored", TF, ["mcIoU", "M1", "n_classes_scored"], "{:d}", 1),
    ("BoundaryOne", T6, ["models", "M1", "boundary_f1_instruments"], "{:.3f}", 1), ("ClShaft", T6, ["models", "M1", "cldice", "shaft"], "{:.3f}", 1),
    ("ClThread", T6, ["models", "M1", "cldice", "thread"], "{:.3f}", 1), ("ClNeedle", T6, ["models", "M1", "cldice", "needle"], "{:.3f}", 1),
    # ---- gate and routing
    ("Tau", T5, ["tau"], "{:.2f}", 1), ("Alpha", S2, ["alpha"], "{:.2f}", 1), ("RiskAcceptedTest", S2, ["risk_bad_accepted_test"], "{:.4f}", 1), ("RiskThreshold", S2, ["empirical_risk_threshold_selection"], "{:.4f}", 1),
    ("FlaggedPct", TFC, ["flagged_share_test"], "{:.1f}", 100), ("GateAucInstTest", T6, ["gate", "auroc_detect_bad_base_frames"], "{:.3f}", 1), ("GateBadRateInst", T6, ["gate", "bad_rate"], "{:.3f}", 1),
    # ---- Task A: aligner
    ("AlignIdOof", AL, ["groups", "oof", "policies", "identity", "A2", "mean_on_flagged"], "{:+.4f}", 1), ("AlignFbOof", AL, ["groups", "oof", "policies", "farneback", "A2", "mean_on_flagged"], "{:+.4f}", 1),
    ("AlignRgOof", AL, ["groups", "oof", "policies", "region", "A2", "mean_on_flagged"], "{:+.4f}", 1),
    ("UnchangedIdOof", AL, ["groups", "oof", "policies", "identity", "A2", "pct_unchanged_of_flagged"], "{:.1f}", 1), ("UnchangedRgTest", AL, ["groups", "test", "policies", "region", "A2", "pct_unchanged_of_flagged"], "{:.1f}", 1),
    ("UnchangedFbTest", AL, ["groups", "test", "policies", "farneback", "A2", "pct_unchanged_of_flagged"], "{:.1f}", 1), ("UnchangedIdTest", AL, ["groups", "test", "policies", "identity", "A2", "pct_unchanged_of_flagged"], "{:.1f}", 1),
    ("AlignRgDiff", AL, ["groups", "oof", "paired_difference_vs_reference", "region", "A2", "diff"], "{:+.4f}", 1), ("AlignRgLo", AL, ["groups", "oof", "paired_difference_vs_reference", "region", "A2", "ci_lo"], "{:+.4f}", 1),
    ("AlignRgHi", AL, ["groups", "oof", "paired_difference_vs_reference", "region", "A2", "ci_hi"], "{:+.4f}", 1),
    # ---- Task B: flagged frames
    ("FlagAThree", PT, ["test", "always-A3 (routed)", "mean_on_flagged"], "{:+.4f}", 1), ("FlagOracle", PT, ["test", "oracle (routed)", "mean_on_flagged"], "{:+.4f}", 1),
    ("FlagSelector", PT, ["test", "selector", "mean_on_flagged"], "{:+.4f}", 1), ("FlagATwo", PT, ["test", "always-A2 (routed)", "mean_on_flagged"], "{:+.4f}", 1),
    ("FlagOracleMinusAThree", PT, ["test", "_headroom_on_flagged", "oracle_minus_A3"], "{:+.4f}", 1), ("AThreeShareOracle", PT, ["test", "_headroom_on_flagged", "A3_share_of_oracle"], "{:.0f}", 100),
    ("SelShareOracle", PT, ["test", "_headroom_on_flagged", "selector_share_of_oracle"], "{:.0f}", 100),
    ("GapUnion", GAPF, ["instrument_iou_union_change"], "{:+.4f}", 1), ("GapGtOnly", GAPF, ["instrument_iou_gt_only_change"], "{:+.4f}", 1), ("GapOfficial", GAPF, ["official_frame_iou_change"], "{:+.4f}", 1),
    ("GapRemoved", GAPF, ["hallucinated_instrument_classes_removed"], "{:d}", 1), ("GapAdded", GAPF, ["hallucinated_instrument_classes_added"], "{:d}", 1),
    ("GapFlaggedBefore", GAPF, ["flagged_frames_with_hallucinated_instrument_class_before"], "{:d}", 1), ("GapFlaggedAfter", GAPF, ["flagged_frames_with_hallucinated_instrument_class_after"], "{:d}", 1),
    # ---- Task C: failure detection
    ("OofBadRate", TC, ["groups", "oof", "bad_rate"], "{:.1f}", 100), ("TestBadRate", TC, ["groups", "test", "bad_rate"], "{:.1f}", 100),
    ("OofFloor", TC, ["groups", "oof", "aurc_floor"], "{:.3f}", 1), ("OofRandom", TC, ["groups", "oof", "aurc_random"], "{:.3f}", 1), ("TestFloor", TC, ["groups", "test", "aurc_floor"], "{:.3f}", 1),
    ("TestRandom", TC, ["groups", "test", "aurc_random"], "{:.3f}", 1),
    ("MsAurcOof", TC, DET("oof", "max-softmax", "aurc"), "{:.3f}", 1), ("MsAurocOof", TC, DET("oof", "max-softmax", "auroc"), "{:.3f}", 1),
    ("MsAurcTest", TC, DET("test", "max-softmax", "aurc"), "{:.3f}", 1), ("MsAurocTest", TC, DET("test", "max-softmax", "auroc"), "{:.3f}", 1),
    ("EntAurcOof", TC, DET("oof", "mean entropy", "aurc"), "{:.3f}", 1), ("EntAurcTest", TC, DET("test", "mean entropy", "aurc"), "{:.3f}", 1),
    ("SdcAurcTest", TC, DET("test", "Soft Dice Confidence", "aurc"), "{:.3f}", 1), ("SdcAurocTest", TC, DET("test", "Soft Dice Confidence", "auroc"), "{:.3f}", 1),
    ("McAurcTest", TC, DET("test", "MC stochastic depth (10 passes)", "aurc"), "{:.3f}", 1),
    ("TtaEntAurcOof", TC, DET("oof", "TTA entropy (6 passes)", "aurc"), "{:.3f}", 1), ("TtaEntAurcTest", TC, DET("test", "TTA entropy (6 passes)", "aurc"), "{:.3f}", 1),
    ("TtaEntAurocTest", TC, DET("test", "TTA entropy (6 passes)", "auroc"), "{:.3f}", 1),
    ("EnsAurcTest", TC, DET("test", "4-model ensemble pairwise Dice", "aurc"), "{:.3f}", 1), ("EnsAurocTest", TC, DET("test", "4-model ensemble pairwise Dice", "auroc"), "{:.3f}", 1),
    ("GateAurcOof", TC, DET("oof", "gate: production (MLP, 9 indicators)", "aurc"), "{:.3f}", 1), ("GateAurcTest", TC, DET("test", "gate: production (MLP, 9 indicators)", "aurc"), "{:.3f}", 1),
    ("GateAurocOof", TC, DET("oof", "gate: production (MLP, 9 indicators)", "auroc"), "{:.3f}", 1), ("GateAurocTest", TC, DET("test", "gate: production (MLP, 9 indicators)", "auroc"), "{:.3f}", 1),
    ("GateInstAurocOof", TC, DET("oof", "gate: production (MLP, 9 indicators)", "auroc_instr_label"), "{:.3f}", 1), ("GateInstAurocTest", TC, DET("test", "gate: production (MLP, 9 indicators)", "auroc_instr_label"), "{:.3f}", 1),
    ("MsInstAurocOof", TC, DET("oof", "max-softmax", "auroc_instr_label"), "{:.3f}", 1), ("MsInstAurocTest", TC, DET("test", "max-softmax", "auroc_instr_label"), "{:.3f}", 1),
    ("XvGateAurcOof", TC, DET("oof", "gate (GBM): 4 + 6 cross-view indicators", "aurc"), "{:.3f}", 1), ("XvGateAurcTest", TC, DET("test", "gate (GBM): 4 + 6 cross-view indicators", "aurc"), "{:.3f}", 1),
    ("MsMs", TC, DET("test", "max-softmax", "ms_per_frame"), "{:.0f}", 1), ("TtaMs", TC, DET("test", "TTA entropy (6 passes)", "ms_per_frame"), "{:.0f}", 1),
    ("McMs", TC, DET("test", "MC stochastic depth (10 passes)", "ms_per_frame"), "{:.0f}", 1), ("EnsMs", TC, DET("test", "4-model ensemble pairwise Dice", "ms_per_frame"), "{:.0f}", 1),
    ("GateMs", TC, DET("test", "gate: production (MLP, 9 indicators)", "ms_per_frame"), "{:.0f}", 1), ("XvMs", TC, DET("test", "gate (GBM): 4 + 6 cross-view indicators", "ms_per_frame"), "{:.0f}", 1),
    # ---- Task D: cross-view
    ("DGain", TD, ["decision", "auroc_gain_full_sample"], "{:+.3f}", 1), ("DLo", TD, ["decision", "ci95", 0], "{:+.3f}", 1), ("DHi", TD, ["decision", "ci95", 1], "{:+.3f}", 1),
    ("DPaperFour", TD, ["cv", "paper4", "auroc"], "{:.3f}", 1), ("DPaperXv", TD, ["cv", "paper4_crossview6", "auroc"], "{:.3f}", 1), ("DXvAlone", TD, ["cv", "crossview6", "auroc"], "{:.3f}", 1),
    ("DPaperStereo", TD, ["cv", "paper4_stereo3", "auroc"], "{:.3f}", 1), ("DTestPaperFour", TDT, ["test", "paper4", "auroc"], "{:.3f}", 1), ("DTestPaperXv", TDT, ["test", "paper4_crossview6", "auroc"], "{:.3f}", 1),
    ("DTestStereo", TDT, ["test", "paper4_stereo3", "auroc"], "{:.3f}", 1),
    ("DAurcDiff", TD, ["cv_paired", "paper4_crossview6 vs paper4", "aurc", "mean"], "{:+.3f}", 1), ("DAurcLo", TD, ["cv_paired", "paper4_crossview6 vs paper4", "aurc", "ci95", 0], "{:+.3f}", 1),
    ("DAurcHi", TD, ["cv_paired", "paper4_crossview6 vs paper4", "aurc", "ci95", 1], "{:+.3f}", 1),
    ("DStereoGain", TD, ["cv_paired", "paper4_stereo3 vs paper4", "auroc_full_sample_diff"], "{:+.3f}", 1),
    # ---- Task E
    ("EMioSingle", TE, ["systems", "single", "mIoU"], "{:.4f}", 1), ("EMioTta", TE, ["systems", "tta", "mIoU"], "{:.4f}", 1), ("EMioEns", TE, ["systems", "ensemble", "mIoU"], "{:.4f}", 1),
    ("EMcSingle", TE, ["systems", "single", "mcIoU"], "{:.4f}", 1), ("EMcTta", TE, ["systems", "tta", "mcIoU"], "{:.4f}", 1), ("EMcEns", TE, ["systems", "ensemble", "mcIoU"], "{:.4f}", 1),
    ("ETtaPts", TE, ["differences_vs_single", "tta", "mIoU_points"], "{:+.2f}", 1), ("ETtaLo", TE, ["differences_vs_single", "tta", "ci95_points", 0], "{:+.2f}", 1), ("ETtaHi", TE, ["differences_vs_single", "tta", "ci95_points", 1], "{:+.2f}", 1),
    ("EEnsPts", TE, ["differences_vs_single", "ensemble", "mIoU_points"], "{:+.2f}", 1), ("EEnsLo", TE, ["differences_vs_single", "ensemble", "ci95_points", 0], "{:+.2f}", 1),
    ("EEnsHi", TE, ["differences_vs_single", "ensemble", "ci95_points", 1], "{:+.2f}", 1),
    # ---- Task F
    ("HallFrames", TF, ["hallucination", "M1", "main_0.05pct", "frames_with_event_pct"], "{:.1f}", 1), ("HallPerFrame", TF, ["hallucination", "M1", "main_0.05pct", "events_per_frame"], "{:.2f}", 1),
    ("HallShare", TF, ["hallucination", "M1", "main_0.05pct", "pooled_share_of_predicted"], "{:.1f}", 100), ("HallFloorPx", TF, ["floor_px_at_full_resolution", "main_0.05pct"], "{:d}", 1),
    ("SelNoAOofAll", TFS, ["oof", "selector", "mean_delta_all_frames"], "{:+.4f}", 1), ("SelNoATestAll", TFS, ["test", "selector", "mean_delta_all_frames"], "{:+.4f}", 1),
    ("ThreeNoATestAll", TFS, ["test", "always-A3 (routed)", "mean_delta_all_frames"], "{:+.4f}", 1), ("SelNoAAct", TFS, ["test", "selector", "activation_pct_of_all"], "{:.1f}", 1),
    ("SelNoAHarmAll", F6, ["models", "M6", "harmful_pct_of_all"], "{:.2f}", 1), ("SelNoAHarmCorr", F6, ["models", "M6", "harmful_pct_of_corrected"], "{:.2f}", 1),
    ("FourHarmAll", T6, ["models", "M4", "harmful_pct_of_all"], "{:.2f}", 1), ("FourHarmCorr", T6, ["models", "M4", "harmful_pct_of_corrected"], "{:.2f}", 1),
    ("FourAct", T6, ["models", "M4", "activation_pct"], "{:.1f}", 1),
    ("CostSelNoA", TFC, ["M6_without_A1_ms"], "{:.0f}", 1), ("CostSelWithA", TFC, ["M6_with_A1_ms"], "{:.0f}", 1), ("CostFour", TFC, ["M4_fusion_every_frame_ms"], "{:.0f}", 1),
    ("CostAllTwo", TFC, ["correct_every_frame_A2_then_A3_ms"], "{:.0f}", 1), ("SpeedUp", TFC, ["speedup_correct_every_frame_A2_A3_over_M6_without_A1"], "{:.1f}", 1),
    ("CostBase", TFC, ["M1_segmentation_plus_gate_ms"], "{:.0f}", 1),
    ("ATwoAvail", TFS, ["available_A2_among_routed"], "{:d}", 1), ("OofRouted", TFS, ["n_oof_routed"], "{:d}", 1),
    ("AFourFramesPct", TFA, ["test", "frames_changed_pct"], "{:.1f}", 1), ("AFourPrecision", TFA, ["test", "drop_precision"], "{:.0f}", 100), ("AFourDropped", TFA, ["test", "classes_dropped"], "{:d}", 1),
    ("AFourBefore", TFA, ["test", "frames_with_event_pct_before"], "{:.1f}", 1), ("AFourAfter", TFA, ["test", "frames_with_event_pct_after"], "{:.1f}", 1),
    ("AFourMiou", TFA, ["test", "mIoU_change_points"], "{:+.2f}", 1),
    # ---- Task G
    ("GStereo", TG, ["metrics", "stereo_two_map", "mean"], "{:.3f}", 1), ("GSameMap", TG, ["metrics", "stereo_same_map_own_set", "mean"], "{:.3f}", 1),
    ("GFlow", TG, ["metrics", "temporal_farneback_on_stereo_set", "mean"], "{:.3f}", 1), ("GPaste", TG, ["metrics", "temporal_identity_on_stereo_set", "mean"], "{:.3f}", 1),
    ("GDiff", TG, ["stereo_two_map_minus_best_temporal", "mean"], "{:+.3f}", 1), ("GDiffLo", TG, ["stereo_two_map_minus_best_temporal", "ci95", 0], "{:+.3f}", 1),
    ("GDiffHi", TG, ["stereo_two_map_minus_best_temporal", "ci95", 1], "{:+.3f}", 1), ("GValid", TG, ["metrics", "instr_pixels_valid_two_map", "mean"], "{:.0f}", 100),
    ("GFrames", TG, ["n_frames"], "{:d}", 1),
    # ---- Task H
    ("HOcc", TH, ["groups", "oof", "scores", "occluded", "overall", "auroc"], "{:.3f}", 1), ("HOccLo", TH, ["groups", "oof", "scores", "occluded", "overall", "ci95", 0], "{:.3f}", 1),
    ("HOccHi", TH, ["groups", "oof", "scores", "occluded", "overall", "ci95", 1], "{:.3f}", 1), ("HLr", TH, ["groups", "oof", "scores", "lr_fail", "overall", "auroc"], "{:.3f}", 1),
    ("HConf", TH, ["groups", "oof", "scores", "low_conf", "overall", "auroc"], "{:.3f}", 1), ("HBoundary", TH, ["groups", "oof", "scores", "boundary_prox", "overall", "auroc"], "{:.3f}", 1),
    # ---- Task I
    ("IBg", TI, ["rows", 1, "delta_points_vs_reference"], "{:+.2f}", 1), ("IUnion", TI, ["rows", 2, "delta_points_vs_reference"], "{:+.2f}", 1), ("IMc", TI, ["rows", 3, "delta_points_vs_reference"], "{:+.2f}", 1),
    ("ISeq", TI, ["rows", 4, "delta_points_vs_reference"], "{:+.2f}", 1), ("ILow", TI, ["rows", 5, "delta_points_vs_reference"], "{:+.2f}", 1),
    # ---- Task J
    ("JKidneyFour", TJ, ["test_seq4_vs_others", 0, "seq4_iou"], "{:.3f}", 1), ("JKidneyOthers", TJ, ["test_seq4_vs_others", 0, "others_iou"], "{:.3f}", 1),
    ("JRigFiles", TJ, ["rigs", "n_distinct_calibration_files"], "{:d}", 1), ("JRigs", TJ, ["rigs", "n_rigs"], "{:d}", 1),
    ("JSeqOneMc", TJ, ["test_per_sequence", "test/seq_1", "mcIoU"], "{:.3f}", 1), ("JSeqFourMc", TJ, ["test_per_sequence", "test/seq_4", "mcIoU"], "{:.3f}", 1),
    # ---- Task K
    ("KMean", TK, ["groups", "train", "pooled", "consecutive_instrument_iou", "mean"], "{:.3f}", 1), ("KLo", TK, ["groups", "train", "pooled", "consecutive_instrument_iou", "mean_ci95_sequence_bootstrap", 0], "{:.3f}", 1),
    ("KHi", TK, ["groups", "train", "pooled", "consecutive_instrument_iou", "mean_ci95_sequence_bootstrap", 1], "{:.3f}", 1), ("KMedian", TK, ["groups", "train", "pooled", "consecutive_instrument_iou", "median"], "{:.3f}", 1),
    ("KBelow", TK, ["groups", "train", "pooled", "consecutive_instrument_iou", "frac_below_0_5"], "{:.0f}", 100), ("KRandom", TK, ["groups", "train", "pooled", "random_pairs_20_apart_instrument_iou", "mean"], "{:.3f}", 1),
    ("KPairs", TK, ["groups", "train", "pooled", "consecutive_instrument_iou", "n"], "{:d}", 1),
    # ---- Task L
    ("LBound", TL, ["upper_bound", "value"], "{:.1f}", 100), ("LGateFail", TL, ["gate_failures_oof", "official_iou_below_0_5"], "{:d}", 1), ("LTopGate", TL, ["top20_that_are_gate_failures", "official_iou_below_0_5"], "{:d}", 1),
    # ---- Task Q
    ("QMiouPts", TQ, ["ceiling_all_instrument_classes", "mIoU_points_over_M6"], "{:+.2f}", 1), ("QMcPts", TQ, ["ceiling_all_instrument_classes", "mcIoU_points_over_M6"], "{:+.2f}", 1),
    ("QWristToClasp", TQ, ["clasper_wrist", "gt_wrist_pixels_predicted_clasper"], "{:.1f}", 100), ("QNeedleCeil", TQ, ["ceiling_all_instrument_classes", "per_class", "suturing-needle"], "{:.3f}", 1),
    ("QFrames", TQ, ["ceiling_frames_with_any_relabelled_pixel_pct"], "{:.0f}", 1),
    # ---- image-based difficulty index (negative result)
    ("DifPearOof", DF, ["oof_train_sequences", "vs_instrument_error", "pearson"], "{:+.3f}", 1), ("DifPearTest", DF, ["official_test_sequences", "vs_instrument_error", "pearson"], "{:+.3f}", 1),
    ("DifQPearOof", DF, ["oof_train_sequences", "confidence_reference", "pearson_q_vs_iou"], "{:.3f}", 1), ("DifQPearTest", DF, ["official_test_sequences", "confidence_reference", "pearson_q_vs_iou"], "{:.3f}", 1),
    ("DifWithinLo", DF, ["official_test_sequences", "vs_instrument_error", "within_seq_spearman_ci95", 0], "{:+.3f}", 1), ("DifWithinHi", DF, ["official_test_sequences", "vs_instrument_error", "within_seq_spearman_ci95", 1], "{:+.3f}", 1),
    # ---- pixel accuracy, Dice, precision and recall (Task R)
    ("PdOursAcc", PD, ["models", 0, "pixel_accuracy"], "{:.4f}", 1), ("PdOursDice", PD, ["models", 0, "mdice"], "{:.4f}", 1), ("PdOursMcDice", PD, ["models", 0, "mcdice"], "{:.4f}", 1), ("PdOursPrec", PD, ["models", 0, "mprecision"], "{:.4f}", 1), ("PdOursRec", PD, ["models", 0, "mrecall"], "{:.4f}", 1), ("PdSegAcc", PD, ["models", 1, "pixel_accuracy"], "{:.4f}", 1), ("PdSegDice", PD, ["models", 1, "mdice"], "{:.4f}", 1), ("PdSegMcDice", PD, ["models", 1, "mcdice"], "{:.4f}", 1), ("PdSegPrec", PD, ["models", 1, "mprecision"], "{:.4f}", 1), ("PdSegRec", PD, ["models", 1, "mrecall"], "{:.4f}", 1), ("PdUnetAcc", PD, ["models", 2, "pixel_accuracy"], "{:.4f}", 1), ("PdUnetDice", PD, ["models", 2, "mdice"], "{:.4f}", 1), ("PdUnetMcDice", PD, ["models", 2, "mcdice"], "{:.4f}", 1), ("PdUnetPrec", PD, ["models", 2, "mprecision"], "{:.4f}", 1), ("PdUnetRec", PD, ["models", 2, "mrecall"], "{:.4f}", 1),
    # ---- Task P.1 baselines (index 0 = ours, 1 = SegFormer-B0, 2 = U-Net; DeepLabV3+ did not finish)
    ("BlOursMiou", TP, ["models", 0, "mIoU"], "{:.4f}", 1), ("BlOursMc", TP, ["models", 0, "mcIoU"], "{:.4f}", 1), ("BlOursParams", TP, ["models", 0, "params_millions"], "{:.1f}", 1),
    ("BlOursMs", TP, ["models", 0, "ms_per_frame"], "{:.0f}", 1), ("BlOursGflops", TP, ["models", 0, "gflops"], "{:.0f}", 1),
    ("BlSegMiou", TP, ["models", 1, "mIoU"], "{:.4f}", 1), ("BlSegMc", TP, ["models", 1, "mcIoU"], "{:.4f}", 1), ("BlSegParams", TP, ["models", 1, "params_millions"], "{:.1f}", 1),
    ("BlSegMs", TP, ["models", 1, "ms_per_frame"], "{:.0f}", 1), ("BlSegGflops", TP, ["models", 1, "gflops"], "{:.0f}", 1),
    ("BlSegDiff", TP, ["models", 1, "difference_from_ours_mIoU_points"], "{:+.1f}", 1), ("BlSegLo", TP, ["models", 1, "difference_from_ours_ci95_points", 0], "{:+.1f}", 1),
    ("BlSegHi", TP, ["models", 1, "difference_from_ours_ci95_points", 1], "{:+.1f}", 1),
    ("BlUnetMiou", TP, ["models", 2, "mIoU"], "{:.4f}", 1), ("BlUnetMc", TP, ["models", 2, "mcIoU"], "{:.4f}", 1), ("BlUnetParams", TP, ["models", 2, "params_millions"], "{:.1f}", 1),
    ("BlUnetMs", TP, ["models", 2, "ms_per_frame"], "{:.0f}", 1), ("BlUnetGflops", TP, ["models", 2, "gflops"], "{:.0f}", 1),
    ("BlUnetDiff", TP, ["models", 2, "difference_from_ours_mIoU_points"], "{:+.1f}", 1), ("BlUnetLo", TP, ["models", 2, "difference_from_ours_ci95_points", 0], "{:+.1f}", 1),
    ("BlUnetHi", TP, ["models", 2, "difference_from_ours_ci95_points", 1], "{:+.1f}", 1),
    # ---- Task P.3 B: EndoVis 2017 zero-shot (binary instrument versus background) and its 2018-test reference under the same mapping
    ("ZsFrames", ZS, ["endovis2017", "detectors", "_n_frames"], "{:d}", 1), ("ZsSeqs", ZS, ["endovis2017", "detectors", "_n_sequences"], "{:d}", 1),
    ("ZsIou", ZS, ["endovis2017", "segmentation", "mean_binary_iou"], "{:.3f}", 1), ("ZsIouRef", ZS, ["endovis2018_reference", "segmentation", "mean_binary_iou"], "{:.3f}", 1),
    ("ZsBelow", ZS, ["endovis2017", "segmentation", "frac_below_0_5"], "{:.1f}", 100), ("ZsBelowRef", ZS, ["endovis2018_reference", "segmentation", "frac_below_0_5"], "{:.1f}", 100),
    ("ZsFlagged", ZS, ["endovis2017", "flagged_pct"], "{:.1f}", 1), ("ZsFlaggedRef", ZS, ["endovis2018_reference", "flagged_pct"], "{:.1f}", 1),
    ("ZsEmptyRef", ZS, ["endovis2018_reference", "segmentation", "frames_with_no_gt_instrument_pct"], "{:.1f}", 1),
    ("ZsGateAuroc", ZS, ["endovis2017", "detectors", "r", "auroc"], "{:.3f}", 1), ("ZsGateAurocLo", ZS, ["endovis2017", "detectors", "r", "auroc_ci95", 0], "{:.3f}", 1),
    ("ZsGateAurocHi", ZS, ["endovis2017", "detectors", "r", "auroc_ci95", 1], "{:.3f}", 1), ("ZsGateAurc", ZS, ["endovis2017", "detectors", "r", "aurc"], "{:.4f}", 1),
    ("ZsMsAuroc", ZS, ["endovis2017", "detectors", "q", "auroc"], "{:.3f}", 1), ("ZsMsAurc", ZS, ["endovis2017", "detectors", "q", "aurc"], "{:.4f}", 1),
    ("ZsEntAuroc", ZS, ["endovis2017", "detectors", "ent", "auroc"], "{:.3f}", 1), ("ZsEntAurc", ZS, ["endovis2017", "detectors", "ent", "aurc"], "{:.4f}", 1),
    ("ZsAurcFloor", ZS, ["endovis2017", "detectors", "_aurc_floor"], "{:.4f}", 1), ("ZsAurcRandom", ZS, ["endovis2017", "detectors", "_aurc_random"], "{:.4f}", 1),
    ("ZsGateAurcLo", ZS, ["endovis2017", "detectors", "r", "aurc_ci95", 0], "{:.4f}", 1), ("ZsGateAurcHi", ZS, ["endovis2017", "detectors", "r", "aurc_ci95", 1], "{:.4f}", 1),
    ("ZsMsAurcLo", ZS, ["endovis2017", "detectors", "q", "aurc_ci95", 0], "{:.4f}", 1), ("ZsMsAurcHi", ZS, ["endovis2017", "detectors", "q", "aurc_ci95", 1], "{:.4f}", 1),
    ("ZsSeqLow", ZS, ["endovis2017", "segmentation", "per_sequence", "val1"], "{:.3f}", 1), ("ZsSeqHigh", ZS, ["endovis2017", "segmentation", "per_sequence", "val10"], "{:.3f}", 1),
    # ---- calibration (figure 6) and the gate's optimism
    ("CalEceBefore", CALP, ["before", "ece"], "{:.3f}", 1), ("CalEceAfter", CALP, ["after", "ece"], "{:.3f}", 1), ("CalTemp", CALP, ["temperature_stored"], "{:.2f}", 1),
    ("CalConfAfter", CALP, ["after", "mean_confidence"], "{:.3f}", 1), ("CalAcc", CALP, ["after", "pixel_accuracy"], "{:.3f}", 1),
    ("GateScoreTest", GCO, ["groups", "test", "mean_gate_score"], "{:.3f}", 1), ("GateAchTest", GCO, ["groups", "test", "mean_achieved_instrument_iou"], "{:.3f}", 1),
    ("GateOffTest", GCO, ["groups", "test", "offset_score_minus_achieved"], "{:+.3f}", 1), ("GateOffOof", GCO, ["groups", "oof", "offset_score_minus_achieved"], "{:+.3f}", 1),
    ("GateAurocOwnTest", GCO, ["ranking", "test", "auroc_own_label"], "{:.3f}", 1), ("GateAurocOwnOof", GCO, ["ranking", "oof", "auroc_own_label"], "{:.3f}", 1),
]


# ---- owner's request: gate with and without max-softmax, and five gate seeds (tools/gate_ablation_seeds.py)
GA = "results/taskC/gate_ablation_seeds.json"
_GA_DET = {"Ms": "max-softmax baseline", "Wq": "gate with max-softmax (9 indicators), refit seed 0", "Nq": "gate without max-softmax (8 indicators), refit seed 0"}
_GA_MET = {"Auroc": "auroc_official", "Aurc": "aurc_official", "AurocInst": "auroc_instrument", "AurcInst": "aurc_instrument"}
for _d, _dn in _GA_DET.items():
    for _g, _gn in (("Oof", "oof"), ("Test", "test")):
        for _m, _mn in _GA_MET.items():
            SPEC.append((f"Ga{_d}{_g}{_m}", GA, ["item1", _gn, "detectors", _dn, _mn], "{:.3f}", 1))
for _d in ("Wq", "Nq"):
    for _g, _gn in (("Oof", "oof"), ("Test", "test")):
        for _k, _kn in (("Aurc", "aurc_official_minus_max_softmax"), ("Auroc", "auroc_official_minus_max_softmax")):
            SPEC.append((f"Ga{_d}{_g}D{_k}", GA, ["item1", _gn, "detectors", _GA_DET[_d], _kn, "mean"], "{:+.3f}", 1))
            SPEC.append((f"Ga{_d}{_g}D{_k}Lo", GA, ["item1", _gn, "detectors", _GA_DET[_d], _kn, "ci95", 0], "{:+.3f}", 1))
            SPEC.append((f"Ga{_d}{_g}D{_k}Hi", GA, ["item1", _gn, "detectors", _GA_DET[_d], _kn, "ci95", 1], "{:+.3f}", 1))
for _m, _mn in _GA_MET.items():
    for _s, _sn, _f in (("Mean", "mean", "{:.3f}"), ("Sd", "std", "{:.3f}"), ("Min", "min", "{:.3f}"), ("Max", "max", "{:.3f}")):
        SPEC.append((f"GaSeed{_m}{_s}", GA, ["item3", "summary", _mn, _sn], _f, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--tex", default="cvavish/numbers.tex")
    a = ap.parse_args()
    tex, md, missing = [], [], []
    for name, f, path, fmt, scale in SPEC:
        d = J(f)
        try:
            v = dig(d, path) if d is not None else None
            if v is None:
                raise KeyError("null")
            s = fmt.format(int(v) if fmt.endswith("d}") else v * scale)
        except Exception as e:  # noqa: BLE001 -- reported, never hidden
            s = "--"
            missing.append((name, f, ".".join(str(p) for p in path), f"{type(e).__name__}: {e}"))
        tex.append(f"\\newcommand{{\\N{name}}}{{{s}}}")
        md.append(f"| `\\N{name}` | {s} | {f} | `{'.'.join(str(p) for p in path)}` |")
    os.makedirs(a.out, exist_ok=True)
    open(a.tex, "w", encoding="utf-8").write("% generated by tools/paper_numbers.py from files under results/ -- do not edit\n" + "\n".join(tex) + "\n")
    lines = ["# Numbers used in the paper prose", "", "Each macro `\\N<name>` in `cvavish/main.tex` is defined in `cvavish/numbers.tex`, generated by `tools/paper_numbers.py` from the file and key path below "
             "(keys joined by dots; list positions are integers). Scale and format are in the tool. Table numbers are listed in `NUMBERS.md`.", "",
             "| macro | value | file | key path |", "|---|---|---|---|"] + md
    if missing:
        lines += ["", "## Missing (defined as `--`)", ""] + [f"- `{n}`: {f} `{p}` ({e})" for n, f, p, e in missing]
    open(os.path.join(a.out, "NUMBERS_TEXT.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{len(SPEC)} macros, {len(missing)} missing")
    for m in missing:
        print("  MISSING", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
