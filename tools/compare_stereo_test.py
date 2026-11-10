#!/usr/bin/env python3
"""Official-test half of tools/compare_stereo.py (same functions, same decision rule).
Gate fit on ALL out-of-fold training frames, scored on the 4 official test sequences.

    python -m tools.compare_stereo_test --oof work_v2/cache_oof_stereo --test work_v2/cache_test_stereo
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.compare_stereo import (SETS, boot_diff, load_dir, metrics, test_run,  # noqa: E402
                                  verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--family", default="gbm")
    ap.add_argument("--memory-size", type=int, default=5)
    ap.add_argument("--tau-train", type=float, default=0.5)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/task4")
    a = ap.parse_args()
    oof = load_dir(a.oof)
    test = {k: c for k, c in load_dir(a.test).items() if k.startswith("test/")}
    kw = {"seed": 0}
    tp = {s: test_run(oof, test, a.family, s, a.memory_size, a.tau_train, a.theta, kw) for s in SETS}
    rs, seq = tp[SETS[0]][1], tp[SETS[0]][2]
    R = {"n_test_frames": int(len(rs)), "sets": {s: metrics(tp[s][0], tp[s][1], a.theta) for s in SETS}, "paired": {}}
    print(f"OFFICIAL TEST ({len(test)} sequences, {len(rs)} frames); AURC floor {R['sets'][SETS[0]]['aurc_oracle']:.4f}, random {R['sets'][SETS[0]]['aurc_random']:.4f}")
    print(f"{'set':20s} {'AUC':>7s} {'AURC':>8s} {'Pearson':>8s} {'MSE':>8s}")
    for s in SETS:
        m = R["sets"][s]
        print(f"{s:20s} {m['auc']:7.4f} {m['aurc']:8.4f} {m['pearson']:8.4f} {m['mse']:8.5f}")
    print("(4 test sequences: the sequence-bootstrap interval is coarse; read it as a warning, not a test)")
    for x, y in [("extended_stereo", "extended"), ("nontemporal_stereo", "nontemporal"),
                 ("nontemporal_stereo", "extended"), ("stereo3", "nontemporal"), ("extended", "nontemporal")]:
        d = boot_diff(tp[x][0], tp[y][0], rs, seq, a.theta, a.n_boot)
        d["verdict"] = verdict(d, x, y)
        R["paired"][f"{x} vs {y}"] = d
        print(f"  {x} vs {y}: dAUC {d['auc']['mean']:+.4f} [{d['auc']['ci95'][0]:+.4f}, {d['auc']['ci95'][1]:+.4f}]  "
              f"dAURC {d['aurc']['mean']:+.4f} -> {d['verdict']}")
    json.dump(R, open(os.path.join(a.out, "gate_comparison_test.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
