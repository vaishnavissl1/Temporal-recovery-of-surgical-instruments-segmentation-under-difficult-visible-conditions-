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
