#!/usr/bin/env python3
"""
ONE COMMAND: raw dataset in, a self-contained release folder out.

    python -m tools.run_all --root /path/to/endovis2018_cvdataset --out work

Runs, in order: audit -> prepared cache -> Stage 1 -> temperature calibration
-> feature cache -> Stage 2 -> release assembly. Every step is skipped if its
output already exists, so re-running after a crash resumes rather than restarts.

The result is `work/release/`, which is the ONLY thing your teammate needs:

    release/
      manifest.json           task, classes, tau, K, temperature, indicators
      class_map.json          the class table derived from your labels.json
      stage1_best.pt          frozen segmentation model
      stage2_reliability.pt   trained reliability estimator
      caches/*.npz            per-frame indicators + predicted masks + r*
      reports/                audit, stage1 history, stage2 report, calibration
      README_FOR_STAGE3.md    how to plug in, with a runnable example

They then write three classes and nothing else:

    from sitr.runtime import ReleaseBundle
    bundle = ReleaseBundle.load("release")
    runner = bundle.runner(recovery=MyRecovery())
    for seq in bundle.sequences("train"):
        runner.reset()
        for state in bundle.stream(seq):
            final, _ = runner.step(state)
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


def run(step: str, argv, skip_if: str = None) -> None:
    if skip_if and os.path.exists(skip_if):
        print(f"\n=== {step}: already done ({skip_if}) — skipping ===")
        return
    print(f"\n{'='*72}\n=== {step}\n{'='*72}")
    cmd = [sys.executable, "-m"] + argv
    print("  " + " ".join(cmd[2:]) + "\n")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"\nFAILED at step: {step} (exit {r.returncode}).\n"
                         "Nothing after this point ran. Fix the error above and "
                         "re-run the same run_all command — completed steps are "
                         "skipped automatically.")
    print(f"\n  {step} finished in {time.time()-t0:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="endovis2018_cvdataset folder")
    ap.add_argument("--out", default="work", help="working directory")
    ap.add_argument("--task", default="instruments",
                    choices=["binary", "parts", "instruments", "full"])
    ap.add_argument("--arch", default="resnet34_unet",
                    choices=["resnet34_unet", "convnext_unet", "unet11"])
    ap.add_argument("--indicators", default="extended",
                    choices=["paper4", "extended"])
    ap.add_argument("--val-seqs", nargs="+", type=int, default=[2, 5, 9, 15])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stage2-epochs", type=int, default=300)
    ap.add_argument("--memory-size", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--include-test", action="store_true", default=True,
                    help="also cache the held-out test sequences for later eval")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--no-bundle-data", action="store_true",
                    help="do NOT copy the prepared images into release/prepared "
                         "(the release will then only work on this machine)")
    ap.add_argument("--sweep-stage2", action="store_true",
                    help="select K and tau_train from the validation data by "
                         "sweeping, instead of using the defaults (~10 min, "
                         "no GPU, no Stage 1 re-run)")
    ap.add_argument("--strict-audit", action="store_true",
                    help="stop if the audit reports warnings, not just problems")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    prep = os.path.join(out, f"prepared_{args.height}x{args.width}_{args.task}")
    s1 = os.path.join(out, "stage1")
    cache = os.path.join(out, "cache")
    s2 = os.path.join(out, "stage2")
    release = os.path.join(out, "release")
    reports = os.path.join(release, "reports")
    os.makedirs(out, exist_ok=True)
    cmap = os.path.join(out, "class_map.json")
    splits = ["train", "test"] if args.include_test else ["train"]

    # ------------------------------------------------------------- step 0
    if not args.skip_audit:
        run("STEP 0/5  audit the dataset",
            ["tools.audit_dataset", "--root", args.root, "--out", cmap,
             "--manifest", os.path.join(out, "dataset_manifest.json"),
             "--scan-colors"] + (["--strict"] if args.strict_audit else []),
            skip_if=cmap)

    # ------------------------------------------------------------ step 0b
    run("STEP 1/5  bake the resized cache",
        ["tools.prepare_dataset", "--root", args.root, "--class-map", cmap,
         "--task", args.task, "--height", str(args.height),
         "--width", str(args.width), "--splits", *splits, "--out", prep],
        skip_if=os.path.join(prep, "prepared_meta.json"))

    # ------------------------------------------------------------- step 1
    s1_argv = ["tools.train_stage1", "--prepared", prep, "--class-map", cmap,
               "--task", args.task, "--arch", args.arch,
               "--val-seqs", *map(str, args.val_seqs),
               "--height", str(args.height), "--width", str(args.width),
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--workers", str(args.workers), "--device", args.device,
               "--out", s1]
    if args.no_pretrained:
        s1_argv.append("--no-pretrained")
    if args.no_amp:
        s1_argv.append("--no-amp")
    run("STEP 2/5  Stage 1 — segmentation (the long one)", s1_argv,
        skip_if=os.path.join(s1, "best.pt"))

    # ------------------------------------------------------------- step 2a
    c_argv = ["tools.cache_features", "--prepared", prep,
              "--checkpoint", os.path.join(s1, "best.pt"),
              "--splits", *splits, "--out", cache,
              "--workers", str(args.workers), "--device", args.device,
              "--overwrite"]
    if args.no_amp:
        c_argv.append("--no-amp")
    run("STEP 3/5  cache per-frame indicators (frozen pass)", c_argv,
        skip_if=os.path.join(cache, "cache_index.json"))

    # ------------------------------------------------------------- step 2b
    if args.sweep_stage2:
        run("STEP 4/5  Stage 2 — reliability estimator (K and tau_train "
            "selected from validation)",
            ["tools.sweep_stage2", "--cache", cache,
             "--val-seqs", *map(str, args.val_seqs),
             "--indicators", args.indicators,
             "--epochs", str(args.stage2_epochs), "--out", s2],
            skip_if=os.path.join(s2, "stage2_report.json"))
    else:
        run("STEP 4/5  Stage 2 — reliability estimator",
            ["tools.train_stage2", "--cache", cache,
             "--val-seqs", *map(str, args.val_seqs),
             "--indicators", args.indicators,
             "--memory-size", str(args.memory_size),
             "--epochs", str(args.stage2_epochs), "--out", s2],
            skip_if=os.path.join(s2, "stage2_report.json"))

    # ------------------------------------------------------- step 3: release
    print(f"\n{'='*72}\n=== STEP 5/5  assemble the release folder\n{'='*72}")
    if os.path.isdir(release):
        shutil.rmtree(release)
    os.makedirs(reports, exist_ok=True)
    os.makedirs(os.path.join(release, "caches"), exist_ok=True)

    shutil.copy2(cmap, os.path.join(release, "class_map.json"))
    shutil.copy2(os.path.join(s1, "best.pt"),
                 os.path.join(release, "stage1_best.pt"))
    # Ship the BEST checkpoint by validation MSE, not the last one. Stage 2
    # overfits (train MSE is typically several times lower than validation,
    # because train and validation are different surgical procedures), so the
    # final epoch is routinely worse than the best one.
    best_rel = os.path.join(s2, "reliability_best.pt")
    last_rel = os.path.join(s2, "reliability_last.pt")
    src_rel = best_rel if os.path.exists(best_rel) else last_rel
    shutil.copy2(src_rel, os.path.join(release, "stage2_reliability.pt"))
    print(f"  reliability model: {os.path.basename(src_rel)}")
    for f in sorted(os.listdir(cache)):
        if f.endswith(".npz"):
            shutil.copy2(os.path.join(cache, f),
                         os.path.join(release, "caches", f))

    # The runtime recomputes P_t and Phi_t from the images, so the release
    # needs them. Without this copy the release only works on this machine.
    if not args.no_bundle_data:
        shutil.copytree(prep, os.path.join(release, "prepared"))
        print("  bundled prepared image data -> release/prepared (portable)")

    for src, dst in ((os.path.join(out, "dataset_manifest.json"), "dataset_audit.json"),
                     (os.path.join(s1, "stage1_history.json"), "stage1_history.json"),
                     (os.path.join(s1, "calibration.json"), "calibration.json"),
                     (os.path.join(cache, "cache_index.json"), "cache_index.json"),
                     (os.path.join(s2, "stage2_report.json"), "stage2_report.json"),
                     (os.path.join(s2, "stage2_diagnostics.png"), "stage2_diagnostics.png"),
                     (os.path.join(s2, "hyperparameter_sweep.json"), "hyperparameter_sweep.json")):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(reports, dst))

    with open(os.path.join(s2, "stage2_report.json")) as f:
        s2rep = json.load(f)
    with open(os.path.join(s1, "stage1_history.json")) as f:
        s1hist = json.load(f)
    import torch
    meta = torch.load(os.path.join(s1, "best.pt"), map_location="cpu",
                      weights_only=False)["meta"]

    def _cache_meta(cache_dir):
        try:
            with open(os.path.join(cache_dir, "cache_index.json")) as f:
                return json.load(f)
        except Exception:
            return {}

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": os.path.abspath(args.root),
        "data_source": {"kind": "prepared", "path": prep},
        "task": args.task, "arch": args.arch,
        "num_classes": int(meta["num_classes"]),
        "target_names": list(meta["target_names"]),
        "instrument_ids": list(meta["instrument_ids"]),
        "height": int(meta["height"]), "width": int(meta["width"]),
        "temperature": float(meta.get("temperature", 1.0)),
        # The cache's mask downsample factor: T_temporal must be recomputed at
        # the SAME resolution Stage 2 was trained on, or the shipped system
        # disagrees with its own report.
        "mask_scale": int(_cache_meta(cache).get("mask_scale", 2)),
        "cache_tta": bool(_cache_meta(cache).get("tta", False)),
        "val_seqs": list(args.val_seqs),
        "indicator_set": args.indicators,
        "indicator_names": list(s2rep["indicator_names"]),
        "tau": float(s2rep["threshold"]["tau"]),
        "tau_is_provisional": True,
        "memory_size": int(s2rep.get("args", {}).get("memory_size",
                                                     args.memory_size)),
        "tau_train": float(s2rep.get("args", {}).get("tau_train", 0.5)),
        "hyperparams_swept": bool(args.sweep_stage2),
        "stage1_best_mIoU": float(s1hist.get("best_mIoU", -1)),
        "stage2": s2rep["val_self_predicted_history"],
        "routing_rate_rho": next(
            (r["rho"] for r in s2rep["threshold"]["grid"]
             if abs(r["tau"] - s2rep["threshold"]["tau"]) < 1e-9), None),
    }
    with open(os.path.join(release, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    shutil.copy2(os.path.join(REPO, "HANDOFF.md"),
                 os.path.join(release, "README_FOR_STAGE3.md"))

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(release) for f in fs) / 1e6
    print(f"\n  release written to: {release}  ({size:.0f} MB)")
    print(f"  Stage 1 best val mIoU : {manifest['stage1_best_mIoU']:.4f}")
    print(f"  Stage 2 val Pearson r : {manifest['stage2'].get('pearson_r', float('nan')):.4f}")
    print(f"  Stage 2 val AUC       : {manifest['stage2'].get('auc_detect_bad_frames', float('nan')):.4f}")
    print(f"  provisional tau       : {manifest['tau']:.2f} "
          f"(routes {manifest['routing_rate_rho']:.1%} of frames)")
    print(f"  temperature           : {manifest['temperature']:.3f}")
    print(f"\n  Hand your teammate the whole '{os.path.basename(release)}' folder "
          "plus this repo.")
    print("  They start at release/README_FOR_STAGE3.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
