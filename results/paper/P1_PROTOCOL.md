# Task P.1: the baseline comparison protocol

No paper in this literature releases a checkpoint, so no published model can be run on this data. The only honest comparison is to train baselines here, under one protocol, and that also
makes this the only protocol-matched comparison available. Everything below is what was actually run.

## Models

| Label | Architecture | Encoder initialisation | Notes |
|---|---|---|---|
| Ours | CAFormer-S18 encoder + FPN decoder + a full-resolution branch (`caformer_surgenet`) | SurgeNetXL (DINO, self-supervised on surgical video) | trained once on all 15 training sequences (work_v2/final); CC-BY-NC-SA 4.0 weights |
| DeepLabV3+ (ResNet-18) | ResNet-18 at output stride 16 (layer 4 dilated), ASPP (1x1, atrous 6/12/18, image pooling), DeepLabV3+ decoder with the 1/4-resolution layer-1 features | ImageNet-1k (torchvision) | the backbone of STswinCL; GroupNorm replaces BatchNorm in the new blocks (batch 1) |
| SegFormer-B0 | MiT-B0 encoder + all-MLP decoder (3.7 M parameters) | ImageNet-1k (`nvidia/mit-b0`, via the `transformers` library, version 4.46.3) | logits at 1/4 resolution, upsampled bilinearly |
| U-Net (classic) | four down- and four up-samplings, doubled width, skip concatenation, double 3x3 convolutions (base width 32) | none: trained from scratch | the floor of the comparison; GroupNorm instead of BatchNorm (batch 1) |

The encoder pretraining differs by design and is part of what each baseline is. This is a stated limitation of the comparison, not something removed: the proposed encoder was pretrained on
surgical video, the DeepLabV3+ and SegFormer encoders on ImageNet, the U-Net on nothing.

## Identical training recipe (copied from the proposed model's own run, `work_v2/final`)

15 training sequences, the same 12-class label map (`work_v2/class_map.json`), prepared data at 512x640 (`work_v2/prepared_512x640_full`), 20 epochs, batch size 1 with gradient accumulation 8
(effective batch 8), AdamW with learning rate 3e-4 and 0.1 times that on the pretrained encoder (no reduction for the from-scratch U-Net, which has no pretrained part), weight decay 1e-4,
3% linear warm-up then cosine decay to 1% of the base rate, gradient clipping 5, exponential moving average of the weights 0.999 (the EMA weights are what is evaluated), loss = cross-entropy + Dice + 0.5 x Tversky with
inverse-square-root class weights clipped at 12, the same augmentation pipeline, fp16 autocast, channels-last, seed 42, a 3.0 GB GPU memory cap on the 4 GB laptop GPU. The single difference: the
proposed model uses stochastic depth 0.1 inside its encoder (a property of that architecture); the baselines have none (SegFormer uses its own configured value).
The command lines are in `work_v2/run_baselines.sh` and `work_v2/baselines/*.log`.

## Training safeguards and what happened (added 12 November)

The proposed model and all baselines train on all 15 sequences, so there is no validation split and the "val mIoU" printed by the training script was `nan` on every epoch of every run. A metric that
is always `nan` hides failure: the first DeepLabV3+ run was allowed to reach epoch 11 while its training loss had been rising since epoch 6. Two changes were made to `tools/train_stage1.py`:

1. **Divergence guard on the training loss.** The run stops at once (exit code 3) if an epoch loss is not finite, or if the epoch loss has risen for three consecutive epochs (four values, each larger than
   the last), or if three consecutive optimiser steps give a non-finite loss (added earlier). The driver script `work_v2/run_baselines3.sh` then moves the failed run to `work_v2/baselines/_failed_<model>` and relaunches it once at one third of the
   learning rate (1e-4). A relaunched run that fails again stays failed and is reported as absent.
2. **No always-`nan` metric.** With no validation split the log prints the training loss, its change from the previous epoch ("UP" or "down") and the words "val: none", instead of three `nan` metrics.
   No validation split was held out: doing so for the baselines only would have changed their training data from the proposed model's, and the baselines must stay protocol-matched.

Applied to every baseline. SegFormer-B0 had already started when the epoch-level guard was added (it carries the step-level guard only, and its loss was falling: 2.657 then 1.718 in epochs 1 and 2); its finished
`stage1_history.json` is checked after the run with the identical rule and the result recorded here. U-Net and DeepLabV3+ start after the change and carry both guards. Divergences so far, all before the guard:

| Run | What happened | Action |
|---|---|---|
| DeepLabV3+, run 1 | loss 0.56 at epoch 6 rising to 1.90 at epoch 11; pretrained BatchNorm uses batch statistics at batch size 1 | stopped by hand; BatchNorm statistics frozen (kept in `work_v2/baselines/_diverged_run1`) |
| DeepLabV3+, run 2 | loss 0.67 at epoch 4, 0.94 at epoch 5, non-finite at epoch 6; the image-pooling branch had no normalisation | GroupNorm added to that branch, step-level guard added (kept in `_diverged_run2`) |

**Outcome of launch 3 (finished 12 November, 04:58 IST).** SegFormer-B0 trained all 20 epochs (loss 2.657 to 0.591; the epoch-level rule applied afterwards finds no three-epoch rise). U-Net trained all 20 epochs
(loss 3.578 to 1.220, monotone; it is the least converged model and is the floor of the comparison). **DeepLabV3+ failed twice more and is absent from the comparison**: at lr 3e-4 the loss rose at epoch 6 (0.721 to 1.124)
and went non-finite; the single permitted retry at lr 1e-4 rose at epoch 5 (0.636 to 1.263) and went non-finite at epoch 6. Four DeepLabV3+ runs in all were unstable at epochs 5 to 6 whatever the learning rate (3e-4, 1e-4) and
whatever the normalisation fix, so the cause is not the learning rate; it was not diagnosed (a candidate, fp16 overflow, was tested by the fifth attempt below and ruled out; the fixed-seed data order and the architecture at batch size 1 remain candidates). The partial weights were
archived (`work_v2/baselines/_failed_deeplabv3p_r18_retry_lr1e-4`) so that they cannot be evaluated as a finished model. The comparison therefore has two baselines. A line in `work_v2/baselines/status.txt` reading "retry exit 0" is
wrong (a driver-script bug in how the exit code was read); the retry aborted with exit 3 and the status file carries a correcting note.

Run 3 changes the order (SegFormer-B0 and U-Net first, DeepLabV3+ last). The frozen BatchNorm statistics and the extra GroupNorm are properties of this DeepLabV3+ implementation at batch size 1, not changes of the shared recipe.
Any learning-rate retry is a deviation from the 3e-4 shared with the other models and is listed in the results table.

**Fifth DeepLabV3+ attempt: full precision (12 November, 10:27 to 10:38 IST).** On the owner's request, one more attempt with automatic mixed precision switched off (`--no-amp`), everything else identical to the shared
recipe (lr 3e-4, 20 epochs, batch 1 with 8-step accumulation, EMA 0.999, same seed, both divergence guards on, same architecture fixes). It did not train: epoch 1 loss 2.4345, epoch 2 loss 1.1098, then the step-level guard
aborted at epoch 2 ("NON-FINITE LOSS for 3 consecutive steps"). It diverged earlier than the AMP runs, which failed at epochs 5 to 6, so mixed-precision overflow is not the cause of this instability. The cause remains
undiagnosed. The rule fixed beforehand was to stop and keep two baselines on a second failure; no further attempt (and no learning-rate retry) was made. Files kept in `work_v2/baselines/_failed_deeplabv3p_r18_fp32`. The driver
script logged this run's exit code as 1 where the guard's own message says 3; the log message, not the code, is the record. The comparison has two baselines and no DeepLabV3+ row appears anywhere.

## Identical evaluation

The model's argmax at 512x640, nearest-neighbour upsampled to 1280x1024, scored with the official rule (class 0 never scored, a class counted only if it is in the ground truth, mean over classes
per frame, then over frames) on the 997 labelled frames of the 4 official test sequences. Reported: mIoU, mcIoU (10 scored classes), per-class IoU, boundary F1 on the instrument union (3 px),
clDice for shaft, thread and needle, parameters, GFLOPs (torch flop counter, 2 per multiply-add, one 512x640 forward), measured ms per frame (batch 1, fp16, image on the GPU, mean of 40 frames).
Differences from the proposed model are per-frame, with a 95% interval from a bootstrap over the 4 test sequences (coarse; a warning, not a test). Code: `tools/baseline_eval.py`.

## What is not in the comparison

- Published numbers. They are reported only in a separate block with a protocol column (`docs/REFERENCES.md`), never in the same column as ours.
- The ensemble and test-time-augmentation rows of Task E, which are different systems and are reported separately (`results/taskE`).
- Any baseline that did not finish training: it is named as absent in `results/taskP/baselines.json` and on the site rather than estimated.
- A comparison on another dataset (Task P.3): not approved, so everything here is one dataset.
