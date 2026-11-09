#!/usr/bin/env python3
"""
Evaluate a Stage 1 model on a split -- normally the 4 OFFICIAL TEST sequences.

Two protocols, reported separately and labelled, never mixed:

  working resolution  prediction and labels both at the training resolution
                      (e.g. 512x640). Fast, but NOT the challenge protocol.
  full resolution     (--full-res-root) logits are bilinearly upsampled to the
                      original 1280x1024 and compared with the ORIGINAL labels
                      read from the raw dataset (label repairs applied). This is
                      the protocol of the 2018 challenge (arXiv:2001.11190):
                      IoU = TP/(TP+FP+FN), averaged over the classes present in
                      each frame, then over frames. Only this number may be put
                      next to published EndoVis 2018 results -- and only if the
                      model was trained with --task full (all 12 classes).

    python -m tools.eval_stage1 --checkpoint work_v2/final/best.pt \
        --prepared work_v2/prepared_512x640_full --split test \
        --full-res-root "C:\\...\\endovis2018_cvdataset" --out work_v2/stage1_test.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.dataset import SequenceDataset, discover_sequences, load_prepared  # noqa: E402
from sitr.labels import ClassMap  # noqa: E402
from sitr.metrics import ConfusionMatrix, frame_iou  # noqa: E402
from sitr.models import build_model, predict_with_tta  # noqa: E402
from sitr.utils import amp_enabled, get_device, load_checkpoint, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--full-res-root", default=None,
                    help="raw dataset root: also evaluate at original resolution")
    ap.add_argument("--full-res", nargs=2, type=int, default=[1024, 1280],
                    metavar=("H", "W"))
    ap.add_argument("--class-map", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = get_device(args.device)
    use_amp = amp_enabled(device, True)
    ck = load_checkpoint(args.checkpoint, map_location=device)
    meta = ck["meta"]
    arch = meta.get("args", {}).get("arch", meta.get("arch"))
    C = int(meta["num_classes"])
    names = list(meta["target_names"])
    inst_ids = [int(c) for c in meta["instrument_ids"]]
    cm_path = args.class_map or meta.get("class_map_path")
    class_map = ClassMap.from_json(cm_path)
    model = build_model(C, arch=arch, pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    seqs, _ = load_prepared(args.prepared, (args.split,), meta["task"],
                            (int(meta["height"]), int(meta["width"])))
    raw = {}
    if args.full_res_root:
        for s in discover_sequences(args.full_res_root, require_labels=True,
                                    apply_repairs=True, splits=(args.split,)):
            raw[s.name] = s
    H, W = args.full_res

    protocols = ["working"] + (["full"] if raw else [])
    acc = {p: {"cm": ConfusionMatrix(C), "union": [], "gt": [], "inst": [],
               "per_seq": {}} for p in protocols}

    for s in seqs:
        loader = DataLoader(SequenceDataset(s, class_map, meta["task"],
                                            int(meta["height"]), int(meta["width"])),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
        rds = None
        if raw:
            if s.name not in raw:
                raise SystemExit(f"{s.key} not found under --full-res-root")
            rs = raw[s.name]
            by_name = {os.path.basename(f): i for i, f in enumerate(rs.frames)}
            rds = SequenceDataset(rs, class_map, meta["task"], H, W)
        seq_scores = {p: [] for p in protocols}
        fi = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device)
                with torch.autocast("cuda", enabled=use_amp):
                    if args.tta:
                        probs, _ = predict_with_tta(model, x)
                        logits = torch.log(probs.clamp_min(1e-8))
                    else:
                        logits, _ = model(x)
                logits = logits.float()
                work_pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
                work_gt = batch["mask"].numpy().astype(np.uint8)
                full_pred = None
                if raw:
                    full_pred = F.interpolate(logits, size=(H, W), mode="bilinear",
                                              align_corners=False
                                              ).argmax(1).cpu().numpy().astype(np.uint8)
                for b in range(work_pred.shape[0]):
                    pairs = [("working", work_pred[b], work_gt[b])]
                    if raw:
                        base = os.path.basename(s.frames[fi])
                        if base not in by_name:
                            raise SystemExit(f"{s.key}/{base}: no matching raw frame")
                        g = rds[by_name[base]]["mask"].numpy().astype(np.uint8)
                        pairs.append(("full", full_pred[b], g))
                    for p, pr, gt in pairs:
                        a = acc[p]
                        a["cm"].update(pr, gt)
                        u = frame_iou(pr, gt, C, present="union")
                        a["union"].append(u)
                        a["gt"].append(frame_iou(pr, gt, C, present="gt"))
                        a["inst"].append(frame_iou(pr, gt, C, class_ids=inst_ids,
                                                   present="union"))
                        seq_scores[p].append(u)
                    fi += 1
        for p in protocols:
            acc[p]["per_seq"][s.key] = float(np.mean(seq_scores[p]))

    report = {"checkpoint": os.path.abspath(args.checkpoint), "split": args.split,
              "arch": arch, "task": meta["task"], "num_classes": C, "tta": args.tta,
              "comparable_to_published": bool(raw) and meta["task"] == "full"}
    print(f"\n{args.split} split, {len(seqs)} sequences, arch={arch}, task={meta['task']}")
    for p in protocols:
        a = acc[p]
        summ = a["cm"].summary(names)
        res = {"challenge_IoU_union": float(np.mean(a["union"])),
               "challenge_IoU_gt": float(np.mean(a["gt"])),
               "instrument_IoU_per_frame": float(np.mean(a["inst"])),
               "dataset_mIoU": float(summ["mIoU"]),
               "per_sequence_challenge_IoU": a["per_seq"],
               "dataset_mDice": float(summ["mDice"]),
               "per_class_IoU": summ.get("per_class_iou", {})}
        report[p] = res
        label = ("FULL RESOLUTION (challenge protocol)" if p == "full"
                 else f"working resolution {meta['height']}x{meta['width']}")
        print(f"  [{label}]")
        print(f"    challenge IoU (present = gt or pred) {res['challenge_IoU_union']:.4f}")
        print(f"    challenge IoU (present = gt only)    {res['challenge_IoU_gt']:.4f}")
        print(f"    instrument IoU per frame             {res['instrument_IoU_per_frame']:.4f}")
        print(f"    dataset mIoU                         {res['dataset_mIoU']:.4f}")
        for k, v in a["per_seq"].items():
            print(f"      {k:<14} {v:.4f}")
    if not report["comparable_to_published"]:
        print("\n  NOTE: not comparable to published EndoVis 2018 numbers"
              + (" (working resolution only)" if not raw else "")
              + (" (model not trained on all 12 classes)" if meta["task"] != "full" else ""))
    write_json(args.out, report)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
