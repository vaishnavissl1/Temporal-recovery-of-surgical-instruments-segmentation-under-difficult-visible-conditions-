# Numbers in the paper tables

Every number placed in a table by `tools/paper_tables.py`, with the file and key path it was read from (keys joined by dots; list positions are integers). Re-run the tool to regenerate; the tables and this list come from the same pass, so they cannot disagree.

| table | quantity | value as printed | file | key path |
|---|---|---|---|---|
| failure | max-softmax oof AURC | 0.269 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.max-softmax.aurc` |
| failure | max-softmax oof AUROC | 0.786 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.max-softmax.auroc` |
| failure | max-softmax test AURC | 0.285 | results/taskC/failure_detection_cost.json | `groups.test.detectors.max-softmax.aurc` |
| failure | max-softmax test AUROC | 0.792 | results/taskC/failure_detection_cost.json | `groups.test.detectors.max-softmax.auroc` |
| failure | max-softmax ms | 80 | results/taskC/failure_detection_cost.json | `groups.test.detectors.max-softmax.ms_per_frame` |
| failure | mean entropy oof AURC | 0.270 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.mean entropy.aurc` |
| failure | mean entropy oof AUROC | 0.773 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.mean entropy.auroc` |
| failure | mean entropy test AURC | 0.285 | results/taskC/failure_detection_cost.json | `groups.test.detectors.mean entropy.aurc` |
| failure | mean entropy test AUROC | 0.789 | results/taskC/failure_detection_cost.json | `groups.test.detectors.mean entropy.auroc` |
| failure | mean entropy ms | 80 | results/taskC/failure_detection_cost.json | `groups.test.detectors.mean entropy.ms_per_frame` |
| failure | Soft Dice Confidence oof AURC | 0.285 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.Soft Dice Confidence.aurc` |
| failure | Soft Dice Confidence oof AUROC | 0.734 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.Soft Dice Confidence.auroc` |
| failure | Soft Dice Confidence test AURC | 0.341 | results/taskC/failure_detection_cost.json | `groups.test.detectors.Soft Dice Confidence.aurc` |
| failure | Soft Dice Confidence test AUROC | 0.685 | results/taskC/failure_detection_cost.json | `groups.test.detectors.Soft Dice Confidence.auroc` |
| failure | Soft Dice Confidence ms | 85 | results/taskC/failure_detection_cost.json | `groups.test.detectors.Soft Dice Confidence.ms_per_frame` |
| failure | MC stochastic depth (10 passes) oof AURC | 0.288 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.MC stochastic depth (10 passes).aurc` |
| failure | MC stochastic depth (10 passes) oof AUROC | 0.741 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.MC stochastic depth (10 passes).auroc` |
| failure | MC stochastic depth (10 passes) test AURC | 0.325 | results/taskC/failure_detection_cost.json | `groups.test.detectors.MC stochastic depth (10 passes).aurc` |
| failure | MC stochastic depth (10 passes) test AUROC | 0.792 | results/taskC/failure_detection_cost.json | `groups.test.detectors.MC stochastic depth (10 passes).auroc` |
| failure | MC stochastic depth (10 passes) ms | 782 | results/taskC/failure_detection_cost.json | `groups.test.detectors.MC stochastic depth (10 passes).ms_per_frame` |
| failure | TTA entropy (6 passes) oof AURC | 0.271 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.TTA entropy (6 passes).aurc` |
| failure | TTA entropy (6 passes) oof AUROC | 0.764 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.TTA entropy (6 passes).auroc` |
| failure | TTA entropy (6 passes) test AURC | 0.273 | results/taskC/failure_detection_cost.json | `groups.test.detectors.TTA entropy (6 passes).aurc` |
| failure | TTA entropy (6 passes) test AUROC | 0.850 | results/taskC/failure_detection_cost.json | `groups.test.detectors.TTA entropy (6 passes).auroc` |
| failure | TTA entropy (6 passes) ms | 535 | results/taskC/failure_detection_cost.json | `groups.test.detectors.TTA entropy (6 passes).ms_per_frame` |
| failure | TTA pairwise Dice (6 passes) oof AURC | 0.282 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.TTA pairwise Dice (6 passes).aurc` |
| failure | TTA pairwise Dice (6 passes) oof AUROC | 0.757 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.TTA pairwise Dice (6 passes).auroc` |
| failure | TTA pairwise Dice (6 passes) test AURC | 0.320 | results/taskC/failure_detection_cost.json | `groups.test.detectors.TTA pairwise Dice (6 passes).aurc` |
| failure | TTA pairwise Dice (6 passes) test AUROC | 0.808 | results/taskC/failure_detection_cost.json | `groups.test.detectors.TTA pairwise Dice (6 passes).auroc` |
| failure | TTA pairwise Dice (6 passes) ms | 557 | results/taskC/failure_detection_cost.json | `groups.test.detectors.TTA pairwise Dice (6 passes).ms_per_frame` |
| failure | 4-model ensemble pairwise Dice test AURC | 0.312 | results/taskC/failure_detection_cost.json | `groups.test.detectors.4-model ensemble pairwise Dice.aurc` |
| failure | 4-model ensemble pairwise Dice test AUROC | 0.840 | results/taskC/failure_detection_cost.json | `groups.test.detectors.4-model ensemble pairwise Dice.auroc` |
| failure | 4-model ensemble pairwise Dice ms | 312 | results/taskC/failure_detection_cost.json | `groups.test.detectors.4-model ensemble pairwise Dice.ms_per_frame` |
| failure | gate: production (MLP, 9 indicators) oof AURC | 0.303 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.gate: production (MLP, 9 indicators).aurc` |
| failure | gate: production (MLP, 9 indicators) oof AUROC | 0.740 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.gate: production (MLP, 9 indicators).auroc` |
| failure | gate: production (MLP, 9 indicators) test AURC | 0.308 | results/taskC/failure_detection_cost.json | `groups.test.detectors.gate: production (MLP, 9 indicators).aurc` |
| failure | gate: production (MLP, 9 indicators) test AUROC | 0.771 | results/taskC/failure_detection_cost.json | `groups.test.detectors.gate: production (MLP, 9 indicators).auroc` |
| failure | gate: production (MLP, 9 indicators) ms | 109 | results/taskC/failure_detection_cost.json | `groups.test.detectors.gate: production (MLP, 9 indicators).ms_per_frame` |
| failure | gate (GBM): 4 + 6 cross-view indicators oof AURC | 0.298 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.gate (GBM): 4 + 6 cross-view indicators.aurc` |
| failure | gate (GBM): 4 + 6 cross-view indicators oof AUROC | 0.729 | results/taskC/failure_detection_cost.json | `groups.oof.detectors.gate (GBM): 4 + 6 cross-view indicators.auroc` |
| failure | gate (GBM): 4 + 6 cross-view indicators test AURC | 0.305 | results/taskC/failure_detection_cost.json | `groups.test.detectors.gate (GBM): 4 + 6 cross-view indicators.aurc` |
| failure | gate (GBM): 4 + 6 cross-view indicators test AUROC | 0.813 | results/taskC/failure_detection_cost.json | `groups.test.detectors.gate (GBM): 4 + 6 cross-view indicators.auroc` |
| failure | gate (GBM): 4 + 6 cross-view indicators ms | 288 | results/taskC/failure_detection_cost.json | `groups.test.detectors.gate (GBM): 4 + 6 cross-view indicators.ms_per_frame` |
| policy | oracle (routed) activation | 20.7 | results/taskB/policy_table.json | `oracle (routed).activation_pct_of_all` |
| policy | oracle (routed) change flagged | +0.0707 | results/taskB/policy_table.json | `oracle (routed).mean_on_flagged` |
| policy | oracle (routed) change all | +0.0215 | results/taskB/policy_table.json | `oracle (routed).mean_all_frames` |
| policy | oracle (routed) official | +0.0005 | results/taskB/policy_table.json | `oracle (routed).official_all_frames` |
| policy | oracle (routed) harmful | 0.0 | results/taskB/policy_table.json | `oracle (routed).harmful_pct_of_corrected` |
| policy | selector activation | 25.9 | results/taskB/policy_table.json | `selector.activation_pct_of_all` |
| policy | selector change flagged | +0.0315 | results/taskB/policy_table.json | `selector.mean_on_flagged` |
| policy | selector change all | +0.0096 | results/taskB/policy_table.json | `selector.mean_all_frames` |
| policy | selector official | -0.0000 | results/taskB/policy_table.json | `selector.official_all_frames` |
| policy | selector harmful | 1.2 | results/taskB/policy_table.json | `selector.harmful_pct_of_corrected` |
| policy | always-A3 (routed) activation | 30.4 | results/taskB/policy_table.json | `always-A3 (routed).activation_pct_of_all` |
| policy | always-A3 (routed) change flagged | +0.0621 | results/taskB/policy_table.json | `always-A3 (routed).mean_on_flagged` |
| policy | always-A3 (routed) change all | +0.0189 | results/taskB/policy_table.json | `always-A3 (routed).mean_all_frames` |
| policy | always-A3 (routed) official | -0.0001 | results/taskB/policy_table.json | `always-A3 (routed).official_all_frames` |
| policy | always-A3 (routed) harmful | 1.0 | results/taskB/policy_table.json | `always-A3 (routed).harmful_pct_of_corrected` |
| policy | always-A2 (routed) activation | 20.5 | results/taskB/policy_table.json | `always-A2 (routed).activation_pct_of_all` |
| policy | always-A2 (routed) change flagged | +0.0102 | results/taskB/policy_table.json | `always-A2 (routed).mean_on_flagged` |
| policy | always-A2 (routed) change all | +0.0031 | results/taskB/policy_table.json | `always-A2 (routed).mean_all_frames` |
| policy | always-A2 (routed) official | +0.0001 | results/taskB/policy_table.json | `always-A2 (routed).official_all_frames` |
| policy | always-A2 (routed) harmful | 1.5 | results/taskB/policy_table.json | `always-A2 (routed).harmful_pct_of_corrected` |
| policy | always-A1 (routed) activation | 30.4 | results/taskB/policy_table.json | `always-A1 (routed).activation_pct_of_all` |
| policy | always-A1 (routed) change flagged | +0.0004 | results/taskB/policy_table.json | `always-A1 (routed).mean_on_flagged` |
| policy | always-A1 (routed) change all | +0.0001 | results/taskB/policy_table.json | `always-A1 (routed).mean_all_frames` |
| policy | always-A1 (routed) official | +0.0004 | results/taskB/policy_table.json | `always-A1 (routed).official_all_frames` |
| policy | always-A1 (routed) harmful | 0.7 | results/taskB/policy_table.json | `always-A1 (routed).harmful_pct_of_corrected` |
| policy | always-fuse (routed) activation | 30.4 | results/taskB/policy_table.json | `always-fuse (routed).activation_pct_of_all` |
| policy | always-fuse (routed) change flagged | +0.0072 | results/taskB/policy_table.json | `always-fuse (routed).mean_on_flagged` |
| policy | always-fuse (routed) change all | +0.0022 | results/taskB/policy_table.json | `always-fuse (routed).mean_all_frames` |
| policy | always-fuse (routed) official | +0.0005 | results/taskB/policy_table.json | `always-fuse (routed).official_all_frames` |
| policy | always-fuse (routed) harmful | 1.7 | results/taskB/policy_table.json | `always-fuse (routed).harmful_pct_of_corrected` |
| policy | always-fuse (EVERY frame) activation | 100.0 | results/taskB/policy_table.json | `always-fuse (EVERY frame).activation_pct_of_all` |
| policy | always-fuse (EVERY frame) change flagged | +0.0072 | results/taskB/policy_table.json | `always-fuse (EVERY frame).mean_on_flagged` |
| policy | always-fuse (EVERY frame) change all | +0.0009 | results/taskB/policy_table.json | `always-fuse (EVERY frame).mean_all_frames` |
| policy | always-fuse (EVERY frame) official | +0.0010 | results/taskB/policy_table.json | `always-fuse (EVERY frame).official_all_frames` |
| policy | always-fuse (EVERY frame) harmful | 1.1 | results/taskB/policy_table.json | `always-fuse (EVERY frame).harmful_pct_of_corrected` |
| perclass | instrument-shaft ours | 0.859 | results/taskJ/per_class_and_rigs.json | `test_per_class.instrument-shaft.pooled_iou` |
| perclass | instrument-clasper ours | 0.566 | results/taskJ/per_class_and_rigs.json | `test_per_class.instrument-clasper.pooled_iou` |
| perclass | instrument-wrist ours | 0.630 | results/taskJ/per_class_and_rigs.json | `test_per_class.instrument-wrist.pooled_iou` |
| perclass | kidney-parenchyma ours | 0.645 | results/taskJ/per_class_and_rigs.json | `test_per_class.kidney-parenchyma.pooled_iou` |
| perclass | covered-kidney ours | 0.401 | results/taskJ/per_class_and_rigs.json | `test_per_class.covered-kidney.pooled_iou` |
| perclass | thread ours | 0.430 | results/taskJ/per_class_and_rigs.json | `test_per_class.thread.pooled_iou` |
| perclass | clamps ours | 0.653 | results/taskJ/per_class_and_rigs.json | `test_per_class.clamps.pooled_iou` |
| perclass | suturing-needle ours | 0.000 | results/taskJ/per_class_and_rigs.json | `test_per_class.suturing-needle.pooled_iou` |
| perclass | intestine ours | 0.464 | results/taskJ/per_class_and_rigs.json | `test_per_class.intestine.pooled_iou` |
| perclass | ultrasound-probe ours | 0.307 | results/taskJ/per_class_and_rigs.json | `test_per_class.ultrasound-probe.pooled_iou` |
| perclass | mIoU ours | 0.613 | results/taskJ/per_class_and_rigs.json | `test_pooled.mIoU` |
| perclass | mcIoU ours | 0.496 | results/taskJ/per_class_and_rigs.json | `test_pooled.mcIoU` |
| ablation | row 0 mIoU | 0.6128 | results/taskI/metric_ablation.json | `rows.0.mIoU` |
| ablation | row 0 change | +0.00 | results/taskI/metric_ablation.json | `rows.0.delta_points_vs_reference` |
| ablation | row 1 mIoU | 0.6452 | results/taskI/metric_ablation.json | `rows.1.mIoU` |
| ablation | row 1 change | +3.23 | results/taskI/metric_ablation.json | `rows.1.delta_points_vs_reference` |
| ablation | row 2 mIoU | 0.5021 | results/taskI/metric_ablation.json | `rows.2.mIoU` |
| ablation | row 2 change | -11.07 | results/taskI/metric_ablation.json | `rows.2.delta_points_vs_reference` |
| ablation | row 3 mIoU | 0.4956 | results/taskI/metric_ablation.json | `rows.3.mIoU` |
| ablation | row 3 change | -11.72 | results/taskI/metric_ablation.json | `rows.3.delta_points_vs_reference` |
| ablation | row 4 mIoU | 0.6128 | results/taskI/metric_ablation.json | `rows.4.mIoU` |
| ablation | row 4 change | -0.01 | results/taskI/metric_ablation.json | `rows.4.delta_points_vs_reference` |
| ablation | row 5 mIoU | 0.6132 | results/taskI/metric_ablation.json | `rows.5.mIoU` |
| ablation | row 5 change | +0.04 | results/taskI/metric_ablation.json | `rows.5.delta_points_vs_reference` |
| ablation | row 6 mIoU | 0.5447 | results/taskI/metric_ablation.json | `rows.6.mIoU` |
| ablation | row 6 change | -6.81 | results/taskI/metric_ablation.json | `rows.6.delta_points_vs_reference` |
| ablation | row 7 mIoU | 0.6455 | results/taskI/metric_ablation.json | `rows.7.mIoU` |
| ablation | row 7 change | +3.27 | results/taskI/metric_ablation.json | `rows.7.delta_points_vs_reference` |
| ablation | row 8 mIoU | 0.5218 | results/taskI/metric_ablation.json | `rows.8.mIoU` |
| ablation | row 8 change | -9.11 | results/taskI/metric_ablation.json | `rows.8.delta_points_vs_reference` |

## Not produced

- baseline_comparison: results/taskP/baselines.json does not exist yet
