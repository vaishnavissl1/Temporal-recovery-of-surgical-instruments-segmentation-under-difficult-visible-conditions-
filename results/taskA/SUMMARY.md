# Task A: the aligner bug

`tools/make_action_data.py` hardcoded `build_align("identity")`, so the temporal action (A2) in the action data behind
Tasks 5 and 6 pasted the previous mask on with no motion compensation. The tool now has `--aligner
{identity,farneback,region}`. Action data was regenerated for Farneback and region (out of fold on the 15 training
sequences with their fold models, and on the 4 official test sequences). Identity was not regenerated: the new code
with `--aligner identity` reproduced the existing `work_v2/actions` file for test seq_1 exactly (all 35 stored arrays
equal apart from timing), so the existing directory is the identity data.

Sources: `aligner_comparison.json`, `compare_output.txt` (side by side, sequence-clustered intervals),
`{identity,farneback,region}/selector_results.json`, `{farneback,region}/final/final_results.json`
(identity = `results/task5` and `results/task6`, unchanged).

## What was compared

Policy "apply the action on every flagged frame, keep the rest". Flagged = reliability below tau = 0.64 and not
degenerate (the selector's routing rule): 1110 of 2235 out-of-fold frames, 303 of 997 test frames (30.4%).
"Instrument IoU" is the mean IoU over the instrument classes that appear in the ground truth OR the prediction
(`present="union"`); see `results/taskB/` for why that matters. "Unchanged" means a change within +-0.01 (the margin
`tools/train_selector.py` and `tools/measure_recovery.py` already use; stated choice).

## Result, out of fold (15 sequences; the decision data)

| aligner | A2 change on flagged | A2 change, all frames | improved / worse / unchanged % | oracle on flagged |
|---|---|---|---|---|
| identity | +0.0047 | +0.0024 | 2.9 / 1.1 / 96.0 | +0.0258 |
| farneback | +0.0030 | +0.0015 | 2.8 / 1.2 / 96.0 | +0.0240 |
| region | +0.0007 | +0.0003 | 2.1 / 1.3 / 96.7 | +0.0222 |

Paired difference to identity, A2 on flagged frames, cluster bootstrap over the 15 sequences (2000 draws, seed 0):
farneback -0.0017 [-0.0072, +0.0002]; region -0.0041 [-0.0115, -0.0002]. Oracle: farneback -0.0018
[-0.0072, +0.0001]; region -0.0036 [-0.0109, +0.0001].

## Result, official test (4 sequences; scored once, after the out-of-fold decision)

| aligner | A2 change on flagged | A2 change, all frames | improved / worse / unchanged % | oracle on flagged |
|---|---|---|---|---|
| identity | +0.0102 | +0.0031 | 2.0 / 1.0 / 97.0 | +0.0707 |
| farneback | +0.0067 | +0.0020 | 1.0 / 1.7 / 97.4 | +0.0706 |
| region | +0.0005 | +0.0002 | 1.0 / 1.0 / 98.0 | +0.0641 |

Four sequences give coarse intervals; read them as warnings. Paired difference to identity: farneback -0.0034
[-0.0048, -0.0003]; region -0.0096 [-0.0141, +0.0003].

## Selector and final comparison per aligner

Official mIoU (`final_results.json`), identity / farneback / region: M1 0.6128 / 0.6128 / 0.6128, M3 0.6129 / 0.6128 /
0.6129, M4 0.6138 / 0.6138 / 0.6139, M5 0.6141 / 0.6140 / 0.6142, M6 0.6128 / 0.6128 / 0.6128. No published mIoU
changes by more than 0.0002.

Selector on flagged test frames (instrument IoU, union rule): identity +0.0315, farneback +0.0412, region +0.0613.
Out of fold the same quantity is +0.0109, +0.0091, +0.0098, so the test difference is not supported out of fold. What
changed is the selector's predictions of A2 and fusion gain (out-of-fold Pearson for A2: 0.331 / 0.014 / -0.069; for
fusion 0.460 / 0.003 / 0.024): they became noise, and the selector chose the morphological refinement (A3) more often
(test share of all frames: A3 19.5% / 20.5% / 22.9%, A2 4.0% / 2.7% / 0.8%). Always-A3 is +0.0621 on flagged test
frames for all three. So the selector under region moves toward always-A3; it does not find a better temporal
action. M6 harmful corrections, % of corrected frames: 3.1 (published) / 2.4 / 2.4.

## Was the null temporal result partly this bug? No.

- The share of flagged frames whose A2 result is unchanged, the number the brief asked to watch, was 96.0% (identity)
  out of fold and 97.0% on test; it is 96.0% / 96.7% out of fold and 97.4% / 98.0% on test with Farneback / region.
  Task 3 reported 95.7%. Motion compensation did not reduce it.
- A2 on flagged frames did not improve with either aligner out of fold; Farneback is lower with an interval that
  includes zero, region is lower with an interval that excludes zero (upper end -0.0002).
- The oracle ceiling on flagged test frames is +0.0707 / +0.0706 / +0.0641, so better alignment did not raise it.

## How the stop rule resolved

The rule (written in `tools/compare_aligners.py` before any new number existed): if every out-of-fold interval for the
difference to identity contains zero, record that identity was not the cause and move on. Applied literally, one
interval (region, A2) excludes zero, so the rule does not report "same answer". It excludes zero on the harmful side:
region is worse than identity, not better. The question the rule exists for is whether the unaligned temporal
candidate was hiding a benefit. It was not: no aligner helped, one hurt. The rule is reported as written and its
literal output is recorded (`decision_oof.identity_was_not_the_cause: false`); the interpretation above is stated
openly, not substituted for it.

## Decision of record

Identity stays the system of record: it is the best of the three out of fold on A2 and on the oracle, and it is the
setting behind the published Task 5 and 6 tables, which therefore do not change. What changes is the sentence "the
temporal action was only measured with no motion compensation": stronger forms were measured and are not better.
No other aligner was added.
