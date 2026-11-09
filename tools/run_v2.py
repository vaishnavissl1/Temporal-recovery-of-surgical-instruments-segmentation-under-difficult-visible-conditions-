#!/usr/bin/env python3
"""
STAGES 1-2, VERSION 2 -- one command, fully resumable, laptop-GPU friendly.

  0  audit the dataset (or reuse an existing class_map.json)
  1  bake a 512x640 cache of ALL 12 classes (train + official test)
  2  split the 15 training sequences into K folds (whole sequences only)
  3  CROSS-FITTING: for each fold, train Stage 1 on the other folds for a
     fixed number of epochs (no epoch picked with held-out labels), then
     predict the held-out fold -> honest out-of-fold (OOF) errors for Stage 2.
     Each fold's calibration temperature is taken from the OTHER folds.
  4  final Stage 1 on all 15 training sequences (same recipe)
  5  predict the 4 OFFICIAL TEST sequences with the final model
  6  evaluate Stage 1 on the official test set (challenge protocol if --root
     is given, which it always is here)
  7  Stage 2: model family / K / tau_train selected by leave-fold-out CV on
     the OOF errors; routing threshold by conformal risk control; tested on
     the official test set
  8  assemble a self-contained release and VERIFY it on the official test set

Every step is skipped if its output already exists, so if the laptop sleeps
or you press Ctrl+C, re-run the SAME command and it continues where it stopped
(an interrupted Stage 1 training resumes from its last epoch).

    python -m tools.run_v2 --root "C:\\...\\endovis2018_cvdataset" --out work_v2 ^
        --class-map work\\class_map.json

Run `python -m tools.check_backbone` FIRST: it verifies the SurgeNetXL weights
load, and picks a batch size that fits your GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(step, argv, done_if=None):
    if done_if and os.path.exists(done_if):
        print(f"\n=== {step}: already done — skipping ===")
        return
    print(f"\n{'='*72}\n=== {step}\n{'='*72}")
    cmd = [sys.executable, "-m"] + argv
    print("  " + " ".join(cmd[2:]) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"\nFAILED at: {step} (exit {r.returncode}). Fix the "
                         "error above and re-run the SAME command; finished "
                         "steps are skipped.")
    print(f"\n  {step} finished in {(time.time()-t0)/60:.1f} min", flush=True)


def read_T(path):
    with open(path) as f:
        return float(json.load(f)["temperature"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="work_v2")
    ap.add_argument("--class-map", default=None,
                    help="reuse an audited class_map.json (e.g. work\\class_map.json)")
    ap.add_argument("--arch", default="caformer_surgenet",
                    choices=["caformer_surgenet", "caformer_imagenet",
                             "convnext_unet", "resnet34_unet"])
    ap.add_argument("--task", default="full",
                    help="full = all 12 classes (needed for comparability "
                         "with published results)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu-mem-cap-gb", type=float, default=0.0,
                    help="hard GPU memory cap passed to every training "
                         "(0 = off). Use it on Windows laptops: a full GPU "
                         "spills to system RAM silently instead of erroring")
    ap.add_argument("--pretrained-path", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-bundle-data", action="store_true")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    prep = os.path.join(out, f"prepared_{args.height}x{args.width}_{args.task}")
    cv = os.path.join(out, "crossfit")
    oof = os.path.join(out, "cache_oof")
    test_cache = os.path.join(out, "cache_test")
    final = os.path.join(out, "final")
    s2 = os.path.join(out, "stage2")
    release = os.path.join(out, "release")

    # ------------------------------------------------------------ 0 audit
    cmap = os.path.join(out, "class_map.json")
    if args.class_map:
        if not os.path.isfile(args.class_map):
            raise SystemExit(f"--class-map {args.class_map} not found")
        if not os.path.exists(cmap):
            shutil.copy2(args.class_map, cmap)
        print(f"using audited class map {args.class_map}")
    else:
        run("STEP 0  audit the dataset",
            ["tools.audit_dataset", "--root", args.root, "--out", cmap,
             "--manifest", os.path.join(out, "dataset_manifest.json"),
             "--scan-colors"], done_if=cmap)

    # ---------------------------------------------------------- 1 prepare
    run("STEP 1  bake the resized cache (train + official test)",
        ["tools.prepare_dataset", "--root", args.root, "--class-map", cmap,
         "--task", args.task, "--height", str(args.height), "--width",
         str(args.width), "--splits", "train", "test", "--out", prep],
        done_if=os.path.join(prep, "prepared_meta.json"))

    # ------------------------------------------------------------ 2 folds
    from sitr.dataset import load_prepared
    from sitr.stage2v2 import make_folds
    fpath = os.path.join(out, "folds.json")
    if os.path.exists(fpath):
        with open(fpath) as f:
            folds = json.load(f)["folds"]
    else:
        seqs, _ = load_prepared(prep, ("train",), args.task, (args.height, args.width))
        folds = make_folds([(s.key, len(s)) for s in seqs], args.folds)
        with open(fpath, "w") as f:
            json.dump({"folds": folds, "k": args.folds,
                       "note": "whole sequences only; balanced by frame count"},
                      f, indent=2)
    print("\nfolds:")
    for i, f in enumerate(folds):
        print(f"  fold {i}: {f}")

    def s1_argv(outdir, extra):
        a = ["tools.train_stage1", "--prepared", prep, "--class-map", cmap,
             "--task", args.task, "--arch", args.arch,
             "--height", str(args.height), "--width", str(args.width),
             "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
             "--accum-steps", str(args.accum_steps), "--ema", str(args.ema),
             "--lr", str(args.lr), "--workers", str(args.workers),
             "--device", args.device, "--early-stop", "0", "--out", outdir,
             "--gpu-mem-cap-gb", str(args.gpu_mem_cap_gb)] + extra
        if args.grad_checkpoint:
            a.append("--grad-checkpoint")
        if args.pretrained_path:
            a += ["--pretrained-path", args.pretrained_path]
        last = os.path.join(outdir, "last.pt")
        if os.path.exists(last):
            a += ["--resume", last]
        return a

    # ------------------------------------------------------ 3 cross-fitting
    for i, fold in enumerate(folds):
        fd = os.path.join(cv, f"fold{i}")
        idx = [str(int(k.split("_")[-1])) for k in fold]
        run(f"STEP 3.{i}  Stage 1 fold {i+1}/{len(folds)} (held out: "
            f"{', '.join(k.split('/')[-1] for k in fold)})",
            s1_argv(fd, ["--val-seqs", *idx, "--select", "last"]),
            done_if=os.path.join(fd, "calibration.json"))

    Ts = [read_T(os.path.join(cv, f"fold{i}", "calibration.json"))
          for i in range(len(folds))]
    print(f"\nfold temperatures: {[round(t, 3) for t in Ts]}")
    os.makedirs(oof, exist_ok=True)
    for i, fold in enumerate(folds):
        fd = os.path.join(cv, f"fold{i}")
        T_other = sum(t for j, t in enumerate(Ts) if j != i) / (len(Ts) - 1)
        run(f"STEP 3.{i}b  out-of-fold predictions, fold {i+1} (T from other folds = "
            f"{T_other:.3f})",
            ["tools.cache_features", "--prepared", prep, "--checkpoint",
             os.path.join(fd, "best.pt"), "--splits", "train", "--seq-keys", *fold,
             "--temperature", f"{T_other:.6f}", "--out", os.path.join(fd, "cache"),
             "--workers", "0", "--device", args.device, "--overwrite"],
            done_if=os.path.join(fd, "cache", "cache_index.json"))
        for f in os.listdir(os.path.join(fd, "cache")):
            if f.endswith(".npz"):
                shutil.copy2(os.path.join(fd, "cache", f), os.path.join(oof, f))

    # ----------------------------------------------------------- 4 final
    T_final = sum(Ts) / len(Ts)
    run("STEP 4  final Stage 1 on all 15 training sequences",
        s1_argv(final, ["--no-val", "--temperature", f"{T_final:.6f}"]),
        done_if=os.path.join(final, "calibration.json"))

    # -------------------------------------------------- 5 official test set
    run("STEP 5  predict the official test sequences",
        ["tools.cache_features", "--prepared", prep, "--checkpoint",
         os.path.join(final, "best.pt"), "--splits", "test", "--out", test_cache,
         "--workers", "0", "--device", args.device, "--overwrite"],
        done_if=os.path.join(test_cache, "cache_index.json"))

    # ---------------------------------------------- 6 Stage 1 on test set
    s1_test = os.path.join(out, "stage1_test.json")
    run("STEP 6  evaluate Stage 1 on the official test set",
        ["tools.eval_stage1", "--checkpoint", os.path.join(final, "best.pt"),
         "--prepared", prep, "--split", "test", "--full-res-root", args.root,
         "--device", args.device, "--out", s1_test], done_if=s1_test)

    # ------------------------------------------------------------- 7 stage 2
    run("STEP 7  Stage 2 (cross-validated selection + conformal threshold)",
        ["tools.train_stage2_cv", "--oof-cache", oof, "--test-cache", test_cache,
         "--folds", fpath, "--alpha", str(args.alpha), "--out", s2],
        done_if=os.path.join(s2, "stage2_report.json"))

    # ------------------------------------------------------------ 8 release
    print(f"\n{'='*72}\n=== STEP 8  assemble the release\n{'='*72}")
    if os.path.isdir(release):
        shutil.rmtree(release)
    os.makedirs(os.path.join(release, "caches"))
    os.makedirs(os.path.join(release, "reports"))
    shutil.copy2(cmap, os.path.join(release, "class_map.json"))
    shutil.copy2(os.path.join(final, "best.pt"), os.path.join(release, "stage1_best.pt"))
    shutil.copy2(os.path.join(s2, "reliability_best.pt"),
                 os.path.join(release, "stage2_reliability.pt"))
    for d in (oof, test_cache):
        for f in os.listdir(d):
            if f.endswith(".npz"):
                shutil.copy2(os.path.join(d, f), os.path.join(release, "caches", f))
    for src, dst in ((s1_test, "stage1_test.json"),
                     (os.path.join(s2, "stage2_report.json"), "stage2_report.json"),
                     (fpath, "folds.json"),
                     (os.path.join(test_cache, "cache_index.json"), "cache_index.json")):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(release, "reports", dst))
    if not args.no_bundle_data:
        shutil.copytree(prep, os.path.join(release, "prepared"))

    import torch
    meta = torch.load(os.path.join(final, "best.pt"), map_location="cpu",
                      weights_only=False)["meta"]
    with open(os.path.join(s2, "stage2_report.json")) as f:
        s2rep = json.load(f)
    with open(s1_test) as f:
        s1rep = json.load(f)
    with open(os.path.join(test_cache, "cache_index.json")) as f:
        cidx = json.load(f)
    test_idx = sorted(int(s["index"]) for s in cidx["sequences"])
    manifest = {
        "version": 2,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": {"kind": "prepared", "path": prep},
        "task": args.task, "arch": args.arch,
        "num_classes": int(meta["num_classes"]),
        "target_names": list(meta["target_names"]),
        "instrument_ids": list(meta["instrument_ids"]),
        "height": int(meta["height"]), "width": int(meta["width"]),
        "temperature": float(meta.get("temperature", 1.0)),
        "mask_scale": int(cidx.get("mask_scale", 2)),
        "crossfit_folds": len(folds),
        "val_seqs": [],
        "indicator_set": s2rep["indicator_set"],
        "indicator_names": list(s2rep["indicator_names"]),
        "tau": float(s2rep["threshold"]["tau"]),
        "tau_is_provisional": True,
        "tau_criterion": s2rep["threshold"]["criterion"],
        "alpha": float(s2rep["threshold"]["alpha"]),
        "memory_size": int(s2rep["args"]["memory_size"]),
        "tau_train": float(s2rep["args"]["tau_train"]),
        "stage2_family": s2rep["selection"]["selected"]["family"],
        "stage1_test_full_res": s1rep.get("full"),
        "stage1_test_working_res": s1rep.get("working"),
        "stage1_best_mIoU": float((s1rep.get("full") or s1rep["working"])
                                  ["challenge_IoU_union"]),
        "stage2": s2rep["test_self_predicted_history"],
        "stage2_cv": s2rep["val_self_predicted_history"],
        "routing_rate_rho": s2rep["routing_rate_rho_test"],
        "verify": {"split": "test", "seqs": test_idx,
                   "rho": s2rep["routing_rate_rho_test"]},
    }
    with open(os.path.join(release, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    shutil.copy2(os.path.join(REPO, "HANDOFF.md"),
                 os.path.join(release, "README_FOR_STAGE3.md"))
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(release) for f in fs) / 1e6
    print(f"  release: {release} ({size:.0f} MB)")

    run("STEP 9  verify the release on the official test sequences",
        ["tools.verify_release", "--release", release, "--device", args.device])

    full = s1rep.get("full") or {}
    print(f"\n{'='*72}\nRESULTS (official EndoVis 2018 test sequences)\n{'='*72}")
    if full:
        print(f"  Stage 1 challenge IoU, full resolution : {full['challenge_IoU_union']:.4f}")
    print(f"  Stage 1 challenge IoU, {args.height}x{args.width}      : "
          f"{s1rep['working']['challenge_IoU_union']:.4f}")
    st = s2rep["test_self_predicted_history"]
    print(f"  Stage 2 AUC / Pearson / MSE           : {st['auc_detect_bad_frames']:.4f} / "
          f"{st['pearson_r']:.4f} / {st['mse']:.5f}")
    print(f"  routing rate at alpha={manifest['alpha']}          : {st['rho']:.3f}   "
          f"(bad-and-accepted {st['risk_bad_accepted']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
