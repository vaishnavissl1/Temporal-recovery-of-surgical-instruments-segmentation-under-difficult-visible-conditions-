"""
Web export blocks for the final-stretch tasks (C, D, E, F, G, H, I, J, K, L, P, Q). Imported by tools/export_web_data.py.

Each block copies selected values out of ONE or a few JSON files under results/ and the function returns, besides the blocks, the map {block: [source files]} that
tools/verify_web_data.py uses to prove that every numeric field of the block equals a number stored in one of those files. Nothing is computed from other numbers here except
selections and re-keying; a block whose source is missing is exported empty with `available: false` so a page can say so instead of inventing a figure.
"""
import json
import os


def _load(R, rel):
    p = os.path.join(R, rel)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def extra_blocks(R):
    B, S = {}, {}

    # ---- C: failure-detection baselines
    c = _load(R, "taskC/failure_detection_cost.json")
    if c:
        B["taskC"] = {"available": True, "hardware": c["hardware"], "batch": c["batch"], "groups": {
            g: {"n_frames": G["n_frames"], "bad_rate": G["bad_rate"], "aurc_floor": G["aurc_floor"], "aurc_random": G["aurc_random"],
                "detectors": {n: {k: d[k] for k in ("auroc", "auroc_ci95", "aurc", "aurc_ci95", "forward_passes", "segmenters_trained", "ms_per_frame", "ms_basis", "auroc_instr_label")}
                              for n, d in G["detectors"].items()}} for g, G in c["groups"].items()}}
        S["taskC"] = ["taskC/failure_detection_cost.json"]
    else:
        B["taskC"], S["taskC"] = {"available": False}, []

    # ---- D: cross-view
    d, dt = _load(R, "taskD/decision.json"), _load(R, "taskD/gate_comparison_test.json")
    if d and dt:
        B["taskD"] = {"available": True, "decision": d["decision"],
                      "oof": {"sets": {k: {"auroc": v["auroc"], "aurc": v["aurc"]} for k, v in d["cv"].items()},
                              "paired": {k: {"auroc_full_sample_diff": v["auroc_full_sample_diff"], "auroc_ci95": v["auc"]["ci95"], "aurc_diff": v["aurc"]["mean"], "aurc_ci95": v["aurc"]["ci95"]}
                                         for k, v in d["cv_paired"].items()},
                              "leave_one_out": {k: v["auroc_drop"] for k, v in d["loo_paper4_crossview6"].items()}},
                      "test": {"sets": {k: {"auroc": v["auroc"], "aurc": v["aurc"]} for k, v in dt["test"].items()},
                               "paired": {k: {"auroc_full_sample_diff": v["auroc_full_sample_diff"], "auroc_ci95": v["auc"]["ci95"], "aurc_diff": v["aurc"]["mean"], "aurc_ci95": v["aurc"]["ci95"]}
                                          for k, v in dt["test_paired"].items()}}}
        S["taskD"] = ["taskD/decision.json", "taskD/gate_comparison_test.json"]
    else:
        B["taskD"], S["taskD"] = {"available": False}, []

    # ---- E: TTA and ensemble
    e = _load(R, "taskE/tta_ensemble.json")
    if e:
        B["taskE"] = {"available": True, "systems": {k: {x: v[x] for x in ("mIoU", "mcIoU", "n_classes_scored", "boundary_f1_instruments", "cldice", "trained_models", "forward_passes")} for k, v in e["systems"].items()},
                      "differences_vs_single": e["differences_vs_single"]}
        S["taskE"] = ["taskE/tta_ensemble.json"]
    else:
        B["taskE"], S["taskE"] = {"available": False}, []

    # ---- F: mcIoU, hallucination, A1 retired, A4, cost
    mh, a4, cost = _load(R, "taskF/mciou_hallucination.json"), _load(R, "taskF/a4/a4.json"), _load(R, "taskF/cost.json")
    if mh and a4 and cost:
        h = lambda m: mh["hallucination"][m]["main_0.05pct"]  # noqa: E731
        B["taskF"] = {"available": True, "floor_px": mh["floor_px_at_full_resolution"]["main_0.05pct"],
                      "mcIoU": {m: {"mIoU": v["mIoU"], "mcIoU": v["mcIoU"], "n_classes_scored": v["n_classes_scored"]} for m, v in mh["mcIoU"].items()},
                      "hallucination": {m: {k: h(m)[k] for k in ("frames_with_event_pct", "events_per_frame", "pooled_share_of_predicted")} for m in mh["hallucination"]},
                      "false_presence_M1": {c: v["false_presence_rate"] for c, v in h("M1")["per_class"].items() if v["false_presence_rate"] is not None},
                      "a4": {g: {k: a4[g][k] for k in ("frames", "frames_changed", "frames_changed_pct", "classes_dropped", "dropped_truly_absent_from_ground_truth", "dropped_present_in_ground_truth",
                                                       "drop_precision", "frames_with_event_pct_before", "frames_with_event_pct_after", "share_of_predicted_invented_before",
                                                       "share_of_predicted_invented_after", "mIoU_before", "mIoU_after", "mIoU_change_points", "mcIoU_before", "mcIoU_after", "mcIoU_change_points")}
                             for g in ("oof", "test")},
                      "cost": {k: cost[k] for k in ("flagged_share_test", "M1_segmentation_plus_gate_ms", "M6_with_A1_ms", "M4_fusion_every_frame_ms", "M6_without_A1_ms", "correct_every_frame_A2_then_A3_ms",
                                                    "speedup_old_tile_M4_over_M6", "speedup_M4_over_M6_without_A1", "speedup_correct_every_frame_A2_A3_over_M6_without_A1")}}
        S["taskF"] = ["taskF/mciou_hallucination.json", "taskF/a4/a4.json", "taskF/cost.json"]
    else:
        B["taskF"], S["taskF"] = {"available": False}, []

    # ---- G: information bound
    g = _load(R, "taskG/information_bound.json")
    if g:
        keep = ("stereo_two_map", "stereo_same_map_own_set", "temporal_identity_on_stereo_set", "temporal_farneback_on_stereo_set", "temporal_region_on_stereo_set",
                "instr_pixels_valid_two_map", "instr_pixels_valid_same_map", "valid_fraction_all_pixels")
        B["taskG"] = {"available": True, "n_frames": g["n_frames"], "n_sequences": g["n_sequences"], "limitation": g["limitation"],
                      "metrics": {k: {x: g["metrics"][k][x] for x in ("mean", "median", "p05", "p25", "p75", "p95", "frac_below_0_5", "mean_ci95_sequence_bootstrap")} for k in keep},
                      "stereo_minus_best_temporal": g["stereo_two_map_minus_best_temporal"]}
        S["taskG"] = ["taskG/information_bound.json"]
    else:
        B["taskG"], S["taskG"] = {"available": False}, []

    # ---- H: per-pixel occlusion
    o = _load(R, "taskH/occlusion_localisation.json")
    if o:
        B["taskH"] = {"available": True, "groups": {gr: {"n_sequences": G["n_sequences"], "n_pixels": G["n_pixels"], "wrong_rate": G["wrong_rate"],
                                                         "scores": {n: {k: {"auroc": v["auroc"], "ci95": v["ci95"]} for k, v in d_.items()} for n, d_ in G["scores"].items()}}
                                                    for gr, G in o["groups"].items()}}
        S["taskH"] = ["taskH/occlusion_localisation.json"]
    else:
        B["taskH"], S["taskH"] = {"available": False}, []

    # ---- I: metric ablation
    i = _load(R, "taskI/metric_ablation.json")
    if i:
        B["taskI"] = {"available": True, "reference_mIoU": i["reference_mIoU"], "frames": i["frames"],
                      "rows": [{k: r[k] for k in ("name", "mIoU", "delta_points_vs_reference", "classes_scored", "resolution")} | {"combined": bool(r.get("combined"))} for r in i["rows"]]}
        S["taskI"] = ["taskI/metric_ablation.json"]
    else:
        B["taskI"], S["taskI"] = {"available": False}, []

    # ---- J: per class, per sequence, rigs
    j = _load(R, "taskJ/per_class_and_rigs.json")
    if j:
        B["taskJ"] = {"available": True, "per_sequence": j["test_per_sequence"], "pooled": j["test_pooled"], "active_classes": j["test_active_classes"],
                      "absent_classes": j["test_classes_with_no_ground_truth_pixels"],
                      "per_class": {n: {"pooled_iou": v["pooled_iou"], "n_frames": v["n_frames"]} for n, v in j["test_per_class"].items()},
                      "seq4_vs_others": j["test_seq4_vs_others"],
                      "rigs": {"n_distinct_calibration_files": j["rigs"]["n_distinct_calibration_files"], "n_rigs": j["rigs"]["n_rigs"], "groups": j["rigs"]["groups"],
                               "test_sequence_rig": j["rigs"]["test_sequence_rig"], "max_within": j["rigs"]["max_within_group_parameter_difference"],
                               "min_between": j["rigs"]["min_between_group_parameter_difference"]},
                      "leave_one_rig_out": {rg: {"held_out_sequences": v["held_out_sequences"], "training_sequences_left": v["training_sequences_left"],
                                                 "classes_with_zero_training_frames": v["classes_with_zero_training_frames"]} for rg, v in j["leave_one_rig_out"].items()},
                      "leave_one_rig_out_feasible": j["leave_one_rig_out_feasible"],
                      "needle_training_frames": {k: v["suturing-needle"] for k, v in j["rigs"]["train_frames_per_class_per_sequence"].items() if v["suturing-needle"]}}
        S["taskJ"] = ["taskJ/per_class_and_rigs.json"]
    else:
        B["taskJ"], S["taskJ"] = {"available": False}, []

    # ---- K: consecutive-frame IoU
    k = _load(R, "taskK/consecutive_frame_iou.json")
    if k:
        P = k["groups"]["train"]["pooled"]
        B["taskK"] = {"available": True,
                      "train_consecutive": {x: P["consecutive_instrument_iou"][x] for x in ("n", "mean", "median", "p05", "p25", "p75", "p95", "frac_below_0_5", "histogram_10_bins", "mean_ci95_sequence_bootstrap")},
                      "train_random": {x: P["random_pairs_20_apart_instrument_iou"][x] for x in ("n", "mean", "median")},
                      "per_sequence": {s: {x: v["instrument_iou"][x] for x in ("mean", "median", "frac_below_0_5")} for s, v in k["groups"]["train"]["sequences"].items()}}
        S["taskK"] = ["taskK/consecutive_frame_iou.json"]
    else:
        B["taskK"], S["taskK"] = {"available": False}, []

    # ---- L: label noise
    lz = _load(R, "taskL/label_noise_summary.json")
    if lz:
        B["taskL"] = {"available": True, "n_training_frames": lz["n_training_frames"], "top20_by_sequence": lz["top20_by_sequence"], "top20_loss_range": lz["top20_loss_range"],
                      "median_loss_all_frames": lz["median_loss_all_frames"], "gate_failures_oof": lz["gate_failures_oof"], "top20_that_are_gate_failures": lz["top20_that_are_gate_failures"],
                      "looks_like_label_error": lz["inspection"]["looks_like_label_error"], "upper_bound": lz["upper_bound"]["value"], "answer": lz["answer"]}
        S["taskL"] = ["taskL/label_noise_summary.json"]
    else:
        B["taskL"], S["taskL"] = {"available": False}, []

    # ---- Q: identity ceiling
    q = _load(R, "taskQ/identity_headroom.json")
    if q:
        B["taskQ"] = {"available": True, "M6": {"mIoU": q["M6"]["mIoU"], "mcIoU": q["M6"]["mcIoU"]},
                      "ceiling_all": {x: q["ceiling_all_instrument_classes"][x] for x in ("mIoU", "mcIoU", "mIoU_points_over_M6", "mcIoU_points_over_M6")},
                      "ceiling_pair": {x: q["ceiling_clasper_wrist_only"][x] for x in ("mIoU", "mcIoU", "mIoU_points_over_M6", "mcIoU_points_over_M6")},
                      "per_class_M6": q["M6"]["per_class"], "per_class_ceiling": q["ceiling_all_instrument_classes"]["per_class"], "clasper_wrist": q["clasper_wrist"],
                      "frames_relabelled_pct": q["ceiling_frames_with_any_relabelled_pixel_pct"],
                      "relabelled_pixel_share": q["ceiling_relabelled_pixels_share_of_predicted_instrument_pixels"]}
        S["taskQ"] = ["taskQ/identity_headroom.json"]
    else:
        B["taskQ"], S["taskQ"] = {"available": False}, []

    # ---- P.1: baselines (present only after training and evaluation have finished)
    pb = _load(R, "taskP/baselines.json")
    if pb and pb.get("models"):
        B["taskP"] = {"available": True, "n_frames": pb["n_frames"],
                      "models": [{"label": m["label"], "mIoU": m["mIoU"], "mcIoU": m["mcIoU"], "boundary_f1": m["boundary_f1_instruments"], "cldice": m["cldice"],
                                  "params_millions": m["params_millions"], "gflops": m["gflops"], "ms_per_frame": m["ms_per_frame"],
                                  "per_class": {c: v["iou"] for c, v in m["per_class"].items()}, "per_sequence": m["per_sequence"],
                                  "diff_points": m.get("difference_from_ours_mIoU_points"), "diff_ci95": m.get("difference_from_ours_ci95_points")} for m in pb["models"]],
                      "unavailable": [u["label"] for u in pb.get("unavailable", [])]}
        S["taskP"] = ["taskP/baselines.json"]
    else:
        B["taskP"], S["taskP"] = {"available": False, "models": [], "unavailable": []}, []

    # ---- P.3 B: EndoVis 2017 zero-shot (evaluation only)
    z = _load(R, "taskP3/endovis17_zeroshot.json")
    if z:
        def _grp(g):
            d = g["detectors"]
            return {"segmentation": g["segmentation"], "flagged_pct": g["flagged_pct"], "n_frames": d["_n_frames"], "n_sequences": d["_n_sequences"], "bad_rate": d["_bad_rate"],
                    "aurc_floor": d["_aurc_floor"], "aurc_random": d["_aurc_random"],
                    "detectors": {k: {x: d[k][x] for x in ("auroc", "auroc_ci95", "aurc", "aurc_ci95")} for k in ("r", "q", "ent", "margin")}}
        B["taskP3"] = {"available": True, "tau": z["tau"], "pred_instrument_classes": z["pred_instrument_classes"],
                       "excluded_classes": sorted(z["excluded_predicted_classes_mapped_to_background"].keys()),
                       "endovis2017": _grp(z["endovis2017"]), "endovis2018_reference": _grp(z["endovis2018_reference"])}
        S["taskP3"] = ["taskP3/endovis17_zeroshot.json"]
    else:
        B["taskP3"], S["taskP3"] = {"available": False}, []

    # ---- calibration of the segmenter (pixel level) and of the gate (frame level)
    cp, go = _load(R, "paper/calibration_pixels.json"), _load(R, "paper/gate_calibration_offset.json")
    if cp and go:
        B["calibration"] = {"available": True,
                            "pixels": {"n_frames": cp["n_frames"], "n_pixels": cp["n_pixels"], "temperature_stored": cp["temperature_stored"],
                                       **{k: {x: cp[k][x] for x in ("temperature", "ece", "nll_per_pixel", "mean_confidence", "pixel_accuracy")} for k in ("before", "after")}},
                            "gate": {"groups": go["groups"], "test_decile_offsets": go["test_decile_offsets_score_minus_achieved"], "conformal": go["conformal"], "ranking": go["ranking"]}}
        S["calibration"] = ["paper/calibration_pixels.json", "paper/gate_calibration_offset.json"]
    else:
        B["calibration"], S["calibration"] = {"available": False}, []
    # ---- the gate with and without max-softmax, and five seeds (owner's request)
    ga = _load(R, "taskC/gate_ablation_seeds.json")
    if ga:
        B["gateAblation"] = {"available": True, "recipe": ga["recipe"], "note": ga["note"], "item1": ga["item1"],
                             "seeds": {"summary": ga["item3"]["summary"], "per_seed": ga["item3"]["per_seed"], "max_softmax_oof": ga["item3"]["max_softmax_oof"]}}
        S["gateAblation"] = ["taskC/gate_ablation_seeds.json"]
    else:
        B["gateAblation"], S["gateAblation"] = {"available": False}, []
    return B, S
