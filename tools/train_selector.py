#!/usr/bin/env python3
"""
TASK 5 -- train and evaluate the benefit selector (predict-before-apply, with abstention).

    python -m tools.train_selector --actions work_v2/actions --folds work_v2/folds.json \
        --out results/task5

Data: tools/make_action_data.py output. For every TRAINING frame the measured IoU
change of every action (ground truth, labels only) is the regression target; the
features are observable only. The selector is CROSS-FITTED with the same folds as
Stage 1 and Stage 2, so a held-out sequence is never scored by a selector that
learned from it. Test sequences are scored by a selector fit on all training
frames. Nothing is tuned on the test set.

Policies compared on the frames the gate ROUTES (reliability below tau):
    oracle     the truly best action per frame (needs ground truth: a CEILING)
    selector   our predict-before-apply choice, abstaining below MARGIN
    always-A1 / always-A2 / always-A3   one action on every routed frame
    always-fuse   stereo+temporal fusion on every routed frame
and, separately, always-fuse on EVERY frame (the LACOSTE-style baseline, M4).

Reported per policy: mean change in instrument IoU (over routed frames and over
all frames), HARMFUL-CORRECTION RATE (share of corrected frames that got worse by
more than `--margin-eval`), correction activation %, mean correction CPU time,
all split by difficulty quartile, plus how often each action is chosen.

STOP CONDITION (from the task): if the oracle's gain barely exceeds always-A2's,
the action space is not worth the complexity: say so and simplify.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sitr.difficulty import (CUE_NAMES, DifficultyIndex, IMAGE_CUE_NAMES,  # noqa: E402
                             cue_table, within_sequence_blur)
from sitr.selector import (CANDIDATES, MARGIN, BenefitSelector, oracle_choice,  # noqa: E402
                           policy_value)

KEYS = ("A1", "A2", "A3", "FUSE", "FUSE_REFINE")
RETIRE_A1 = False


def load_group(d, group):
    seqs = {}
    for f in sorted(os.listdir(d)):
        if f.startswith(group + "__") and f.endswith(".npz"):
            z = np.load(os.path.join(d, f), allow_pickle=False)
            seqs[str(z["seq"])] = {k: z[k] for k in z.files if k != "seq"}
    return seqs


def stack(seqs, order=None):
    keys = order or sorted(seqs)
    out = {}
    for k in seqs[keys[0]]:
        out[k] = np.concatenate([seqs[s][k] for s in keys], 0)
    out["seq"] = np.array([s for s in keys for _ in range(len(seqs[s]["t"]))])
    return out


def cues_from_raw(raw, seq):
    stereo = dict(stereo_lr_fail=raw[:, 0], occluded=raw[:, 1], specular=raw[:, 2])
    return cue_table(stereo, raw[:, 3:3 + len(IMAGE_CUE_NAMES)], seq)


def features(D, diff_score):
    F = np.concatenate([D["ind9"], D["st3"], D["r"][:, None], diff_score[:, None]], 1).astype(np.float32)
    G = {a: D[f"f_{a}"].astype(np.float32) for a in CANDIDATES}
    return F, G


def policy_table(D, routed, choices, diff, margin_eval, quart_edges):
    """Summary of every policy on one dataset. choices: name -> per-frame action array (object)."""
    res = {}
    dm = {a: D[f"d_{a}"] for a in KEYS}
    om = {a: D[f"o_{a}"] for a in KEYS}
    secs = {a: D[f"sec_{a}"] for a in KEYS}
    quart = DifficultyIndex.quartile_of(diff, quart_edges)
    for name, ch in choices.items():
        ch = ch.copy()
        # an action that is unavailable on a frame (e.g. A2 with an empty memory)
        # cannot be applied, so that frame is NOT counted as corrected
        applied_mask = np.zeros(len(ch), bool)
        for a_ in KEYS:
            applied_mask |= (ch == a_) & np.isfinite(dm[a_])
        ch = np.where(applied_mask, ch, "A0").astype(object)
        v = policy_value(ch, dm)                      # instrument IoU change per frame
        w = policy_value(ch, om)
        sec = np.zeros(len(ch))
        for a in KEYS:
            m = ch == a
            sec[m] = secs[a][m]
        corrected = applied_mask
        row = dict(
            n_frames=int(len(ch)), n_routed=int(routed.sum()), n_corrected=int(corrected.sum()),
            activation_pct_of_all=100 * float(corrected.mean()),
            mean_delta_all_frames=float(v.mean()), mean_delta_official_all_frames=float(np.nanmean(w)),
            mean_delta_routed=float(v[routed].mean()) if routed.any() else float("nan"),
            mean_delta_when_corrected=float(v[corrected].mean()) if corrected.any() else 0.0,
            harmful_rate_of_corrected=100 * float((v[corrected] < -margin_eval).mean()) if corrected.any() else 0.0,
            harmful_pct_of_all=100 * float(((v < -margin_eval) & corrected).mean()),
            improved_rate_of_corrected=100 * float((v[corrected] > margin_eval).mean()) if corrected.any() else 0.0,
            mean_correction_cpu_ms_per_frame=1000 * float(sec.mean()),
            action_share={a: float((ch == a).mean()) for a in ("A0",) + KEYS if (ch == a).any()})
        row["by_difficulty_quartile"] = [
            dict(quartile=q + 1, n=int((quart == q).sum()), mean_delta=float(v[quart == q].mean()),
                 harmful_pct_of_all=100 * float(((v < -margin_eval) & corrected & (quart == q)).sum() / max((quart == q).sum(), 1)),
                 activation_pct=100 * float(corrected[quart == q].mean())) for q in range(4)]
        res[name] = row
    return res


def print_table(title, res):
    print(f"\n{title}")
    print(f"{'policy':16s} {'activation%':>11s} {'mean dIoU(all)':>14s} {'mean dIoU(corrected)':>20s} {'HARMFUL % of corrected':>23s} {'improved %':>10s} {'CPU ms':>7s}")
    for k, r in res.items():
        print(f"{k:16s} {r['activation_pct_of_all']:11.1f} {r['mean_delta_all_frames']:+14.4f} {r['mean_delta_when_corrected']:+20.4f} "
              f"{r['harmful_rate_of_corrected']:23.1f} {r['improved_rate_of_corrected']:10.1f} {r['mean_correction_cpu_ms_per_frame']:7.1f}")


def evaluate(D, sel_choice, routed, margin_eval, diff):
    n = len(routed)
    A0 = np.array(["A0"] * n, dtype=object)
    on = lambda a: np.where(routed, a, "A0").astype(object)
    dm = {a: D[f"d_{a}"] for a in KEYS}
    choices = {
        "A0 (keep)": A0,
        "oracle (routed)": np.where(routed, oracle_choice(dm, CANDIDATES), "A0").astype(object),
        "selector": np.where(routed, sel_choice, "A0").astype(object),
        "always-A1 (routed)": on("A1"), "always-A2 (routed)": on("A2"), "always-A3 (routed)": on("A3"),
        "always-fuse (routed)": on("FUSE"),
        "always-fuse (EVERY frame)": np.array(["FUSE"] * n, dtype=object),
        "always-fuse+refine (EVERY)": np.array(["FUSE_REFINE"] * n, dtype=object),
    }
    if RETIRE_A1:
        for k in ("always-A1 (routed)", "always-fuse (routed)", "always-fuse (EVERY frame)", "always-fuse+refine (EVERY)"):
            choices.pop(k)
    edges = DifficultyIndex.quartile_edges(diff)
    return policy_table(D, routed, choices, diff, margin_eval, edges)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--tau", type=float, default=None,
                    help="routing threshold; default: read from --release's manifest")
    ap.add_argument("--release", default=None)
    ap.add_argument("--margin", type=float, default=MARGIN, help="selector abstention margin (stated choice)")
    ap.add_argument("--margin-eval", type=float, default=0.01,
                    help="IoU change smaller than this is not counted as improved/harmful")
    ap.add_argument("--out", default="results/task5")
    ap.add_argument("--retire-a1", action="store_true",
                    help="Task F.3: retire action A1 (stereo label transfer) and with it the stereo-temporal fusion that contains it. The candidate set becomes {A2, A3}; "
                         "the always-A1 / fusion policies are not reported. The code and the earlier measurement of A1 stay in the record (results/task5, results/FINDINGS.md).")
    a = ap.parse_args()
    global CANDIDATES, RETIRE_A1
    RETIRE_A1 = bool(a.retire_a1)
    if RETIRE_A1:
        CANDIDATES = ("A2", "A3")
    os.makedirs(a.out, exist_ok=True)
    tau = a.tau
    if tau is None and a.release:
        tau = float(json.load(open(os.path.join(a.release, "manifest.json")))["tau"])
    if tau is None:
        raise SystemExit("give --tau or --release")
    folds = json.load(open(a.folds))["folds"]

    oof, test = load_group(a.actions, "oof"), load_group(a.actions, "test")
    D = stack(oof)
    # difficulty reference = the TRAINING frames only
    cues_tr = cues_from_raw(D["cue_raw"], D["seq"])
    idx = DifficultyIndex().fit(cues_tr)
    diff = idx.score(cues_tr)
    routed = (D["r"] < tau) & ~D["degenerate"].astype(bool)
    print(f"OOF frames {len(D['t'])}, routed by the gate (r < {tau:.3f}): {int(routed.sum())} ({100*routed.mean():.1f}%)")
    F, G = features(D, diff)
    delta = {x: np.where(routed, D[f"d_{x}"], np.nan) for x in CANDIDATES}
    R = {"tau": tau, "margin": a.margin, "margin_eval": a.margin_eval,
         "n_oof": int(len(D["t"])), "n_oof_routed": int(routed.sum())}
    for x in CANDIDATES:
        ok = np.isfinite(delta[x])
        R[f"available_{x}_among_routed"] = int(ok.sum())
        R[f"truth_mean_delta_{x}_among_routed"] = float(np.nanmean(delta[x])) if ok.any() else None

    # ---------------------------------------------- cross-fitted predictions (OOF)
    pred = {x: np.full(len(F), np.nan) for x in CANDIDATES}
    for fold in folds:
        te = np.isin(D["seq"], fold)
        tr = ~te
        sel = BenefitSelector(margin=a.margin, candidates=CANDIDATES).fit(F[tr], {x: G[x][tr] for x in CANDIDATES},
                                                    {x: delta[x][tr] for x in CANDIDATES})
        p = sel.predict(F[te], {x: G[x][te] for x in CANDIDATES})
        for x in CANDIDATES:
            pred[x][te] = p[x]
    sel_oof = BenefitSelector(margin=a.margin, candidates=CANDIDATES)
    choice_oof = sel_oof.choose(pred)
    R["oof"] = evaluate(D, choice_oof, routed, a.margin_eval, diff)
    print_table("HELD-OUT TRAINING SEQUENCES (cross-fitted selector)", R["oof"])
    # how good are the predictions themselves
    R["prediction_quality_oof"] = {}
    for x in CANDIDATES:
        ok = np.isfinite(delta[x]) & np.isfinite(pred[x])
        if ok.sum() > 10:
            R["prediction_quality_oof"][x] = dict(
                n=int(ok.sum()), pearson=float(np.corrcoef(pred[x][ok], delta[x][ok])[0, 1]),
                mae=float(np.mean(np.abs(pred[x][ok] - delta[x][ok]))),
                true_mean=float(delta[x][ok].mean()), pred_mean=float(pred[x][ok].mean()))
    print("\nprediction quality on routed OOF frames (predicted vs measured IoU change):")
    for x, q in R["prediction_quality_oof"].items():
        print(f"  {x}: n={q['n']}  Pearson {q['pearson']:+.3f}  MAE {q['mae']:.4f}  (true mean {q['true_mean']:+.4f}, predicted mean {q['pred_mean']:+.4f})")

    # margin sweep (reported, NOT used to choose the margin)
    R["margin_sweep_oof"] = []
    print("\nabstention-margin sweep on OOF (information only; the margin is a stated choice):")
    for m in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1):
        s2 = BenefitSelector(margin=m, candidates=CANDIDATES)
        ch = np.where(routed, s2.choose(pred), "A0").astype(object)
        v = policy_value(ch, {x: D[f"d_{x}"] for x in KEYS})
        c = ch != "A0"
        row = dict(margin=m, activation_pct_of_all=100 * float(c.mean()), mean_delta_all=float(v.mean()),
                   harmful_pct_of_corrected=100 * float((v[c] < -a.margin_eval).mean()) if c.any() else 0.0)
        R["margin_sweep_oof"].append(row)
        print(f"  margin {m:5.3f}: activation {row['activation_pct_of_all']:5.1f}%  mean dIoU {row['mean_delta_all']:+.4f}  harmful {row['harmful_pct_of_corrected']:5.1f}% of corrected")

    # ------------------------------------------------------------- official test
    if test:
        T = stack(test)
        cues_te = cues_from_raw(T["cue_raw"], T["seq"])
        diff_te = idx.score(cues_te)
        routed_te = (T["r"] < tau) & ~T["degenerate"].astype(bool)
        Ft, Gt = features(T, diff_te)
        sel = BenefitSelector(margin=a.margin, candidates=CANDIDATES).fit(F, G, delta)
        choice_te = sel.choose(sel.predict(Ft, Gt))
        R["n_test"], R["n_test_routed"] = int(len(T["t"])), int(routed_te.sum())
        print(f"\nOFFICIAL TEST: {len(T['t'])} frames, routed {int(routed_te.sum())} ({100*routed_te.mean():.1f}%)")
        R["test"] = evaluate(T, choice_te, routed_te, a.margin_eval, diff_te)
        print_table("OFFICIAL TEST SEQUENCES (selector fit on all training frames)", R["test"])
        pt = sel.predict(Ft, Gt)
        np.savez(os.path.join(a.out, "choices_test.npz"), seq=T["seq"].astype("U32"), t=T["t"],
                 choice=np.where(routed_te, choice_te, "A0").astype("U8"), routed=routed_te,
                 difficulty=diff_te, r=T["r"], instr_before=T["instr_before"],
                 **{f"pred_{x}": pt[x] for x in CANDIDATES})

    # --------------------------------------------------------------- stop check
    def gain(res, k):
        return res[k]["mean_delta_all_frames"]
    for tag in [t for t in ("oof", "test") if t in R]:
        o, a2 = gain(R[tag], "oracle (routed)"), gain(R[tag], "always-A2 (routed)")
        R[f"stop_check_{tag}"] = dict(oracle=o, always_A2=a2, oracle_minus_always_A2=o - a2,
                                      barely_above=bool(o - a2 < 0.005))
        print(f"\nSTOP CHECK [{tag}]: oracle gain {o:+.4f} vs always-A2 {a2:+.4f} (difference {o-a2:+.4f}) -> "
              f"{'ACTION SPACE NOT WORTH THE COMPLEXITY (oracle barely above always-A2)' if o - a2 < 0.005 else 'oracle has real headroom over always-A2'}")
    with open(os.path.join(a.out, "selector_results.json"), "w") as f:
        json.dump(R, f, indent=1)
    print(f"\nwrote {a.out}/selector_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
