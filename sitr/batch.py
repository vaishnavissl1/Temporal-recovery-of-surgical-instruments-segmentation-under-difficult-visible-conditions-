"""
Batch analysis of one uploaded sequence with the released pipeline, frame by frame.

The stages are the ones the paper evaluates, run through the same code:

  1 segmentation          ReleaseBundle.stream  (released segmenter, temperature-scaled)
  2 reliability gate      the released gate (9 indicators, temporal memory K = 3)
  3 routing               a frame is flagged when r < tau and an instrument is predicted
  4 candidate actions     A2 temporal, A3 refine (A1 stereo is retired, Task F)
  5 selector              gradient-boosted benefit predictors with abstention, fitted on the
                          15 out-of-fold training sequences exactly as tools/train_selector.py
                          fits it for the test sequences
  6 identity verification the shipped Stage 5, which by design returns the mask unchanged

Stereo evidence (left/right pair + calibration) feeds the selector's inputs: three stereo
indicators, three stereo cues of the difficulty score, and, for the flagged frames, the
right-view agreement. The right view is segmented by the same model.

Ground truth, when present, is read only AFTER every mask exists, to score them. Nothing
here uses it to decide anything.

Two honest differences from the paper's run, both measured and reported by the tool:
  * the memory is empty at the first uploaded frame (the paper's run had seen the sequence
    from its first frame), so the first K frames are cold start;
  * the vertical stereo self-calibration uses the frames 0, 5, 10, ... of the UPLOAD (the
    paper used frames 0 to 35 of the whole sequence).
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch

from .actions import ACTIONS, ActionContext, ActionParams, action_features
from .align import build_align
from .dataset import IMAGENET_MEAN, IMAGENET_STD, SequenceInfo
from .difficulty import DifficultyIndex, IMAGE_CUE_NAMES, cue_table, stereo_cues
from .identity import HungarianIdentityVerifier
from .metrics import frame_iou, official_class_ious
from .recovery import TemporalRecovery
from .runtime import ReleaseBundle
from .selector import BenefitSelector
from .stereo import (SGBMParams, VerticalCorrection, calibration_path, compute_evidence,
                     estimate_vertical_correction, get_rectifier, instrument_stereo_indicators,
                     right_path, right_to_left_grid)

CANDIDATES = ("A2", "A3")           # A1 (stereo label transfer) is retired: Task F
MARGIN_EVAL = 0.01                  # an IoU change inside +-0.01 is "unchanged" (the paper's stated choice)
REFINE_FRAMES = (0, 5, 10, 15, 20, 25, 30, 35)
CLASS_NAMES = ["background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist",
               "kidney-parenchyma", "covered-kidney", "thread", "clamps", "suturing-needle",
               "suction-instrument", "small-intestine", "ultrasound-probe"]


def _load_group(d: str, group: str) -> Dict[str, Dict[str, np.ndarray]]:
    seqs = {}
    for f in sorted(os.listdir(d)):
        if f.startswith(group + "__") and f.endswith(".npz"):
            z = np.load(os.path.join(d, f), allow_pickle=False)
            seqs[str(z["seq"])] = {k: z[k] for k in z.files if k != "seq"}
    return seqs


def _stack(seqs):
    keys = sorted(seqs)
    out = {k: np.concatenate([seqs[s][k] for s in keys], 0) for k in seqs[keys[0]]}
    out["seq"] = np.array([s for s in keys for _ in range(len(seqs[s]["t"]))])
    return out


def _cues(raw, seq):
    stereo = dict(stereo_lr_fail=raw[:, 0], occluded=raw[:, 1], specular=raw[:, 2])
    return cue_table(stereo, raw[:, 3:3 + len(IMAGE_CUE_NAMES)], seq)


class BatchPipeline:
    def __init__(self, release: str, actions_dir: str, sgbm_json: str, device: str = "cpu",
                 bundle: Optional[ReleaseBundle] = None):
        t0 = time.time()
        self.b = bundle if bundle is not None else ReleaseBundle.load(release, device=device)
        self.nc = self.b.num_classes
        self.inst = list(self.b.instrument_ids)
        self.names = self.b.target_names
        self.tau = float(self.b.tau)
        self.params = ActionParams()
        t2 = json.load(open(sgbm_json))
        self.sgbm = SGBMParams(**t2["sgbm_chosen"])
        self.dist, self.rot = t2["chosen_convention"].split("/")
        # selector: the same fit tools/train_selector.py does for the test sequences
        D = _stack(_load_group(actions_dir, "oof"))
        cues = _cues(D["cue_raw"], D["seq"])
        self.diff_index = DifficultyIndex().fit(cues)
        diff = self.diff_index.score(cues)
        routed = (D["r"] < self.tau) & ~D["degenerate"].astype(bool)
        F = np.concatenate([D["ind9"], D["st3"], D["r"][:, None], diff[:, None]], 1).astype(np.float32)
        G = {a: D[f"f_{a}"].astype(np.float32) for a in CANDIDATES}
        delta = {a: np.where(routed, D[f"d_{a}"], np.nan) for a in CANDIDATES}
        self.selector = BenefitSelector(candidates=CANDIDATES).fit(F, G, delta)
        self.n_selector_frames = int(len(F))
        self.identity = HungarianIdentityVerifier(instrument_ids=self.inst)
        self.load_seconds = time.time() - t0

    # ------------------------------------------------------------------ right view
    @torch.no_grad()
    def _segment_right(self, path: str):
        h, w = int(self.b.manifest["height"]), int(self.b.manifest["width"])
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        x = ((img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        logits, _ = self.b.model(torch.from_numpy(np.ascontiguousarray(x))[None].to(self.b.device))
        conf, lab = torch.softmax(logits.float() / self.b.temperature, 1).max(1)
        return (lab[0].cpu().numpy().astype(np.uint8),
                np.clip(conf[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8))

    # ---------------------------------------------------------------------- run
    def run(self, root: str, progress: Callable[[int, int, str], None] = lambda *a: None) -> dict:
        from tools.check_stereo import read_mask
        t_start = time.time()
        ld = os.path.join(root, "left_frames")
        lefts = sorted(os.path.join(ld, f) for f in os.listdir(ld) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        n = len(lefts)
        if n == 0:
            raise ValueError("no frames found in left_frames/")
        for p in lefts:
            if not os.path.isfile(right_path(p)):
                raise ValueError(f"missing right-camera frame for {os.path.basename(p)}")
        calib = calibration_path(ld)
        if not os.path.isfile(calib):
            raise ValueError("camera_calibration.txt is missing")
        stems = [os.path.splitext(os.path.basename(p))[0] for p in lefts]
        lab_dir = os.path.join(root, "labels")
        label_paths: List[Optional[str]] = []
        for p in lefts:
            q = os.path.join(lab_dir, os.path.basename(p))
            label_paths.append(q if os.path.isfile(q) else None)
        has_labels = all(q is not None for q in label_paths)

        H, W = int(self.b.manifest["height"]), int(self.b.manifest["width"])
        sec = {k: 0.0 for k in ("segment_gate", "stereo", "right_view", "actions", "selector", "identity")}

        # stereo geometry + vertical self-calibration from the upload's own frames
        t0 = time.perf_counter()
        rect = get_rectifier(calib, 1.0, self.dist, self.rot)
        refine = [lefts[i] for i in REFINE_FRAMES if i < n]
        corr = estimate_vertical_correction(rect, refine) if refine else VerticalCorrection()
        sec["stereo"] += time.perf_counter() - t0

        seq = SequenceInfo(name="upload", split="test", index=0, frame_dir=ld, label_dir=None,
                           frames=lefts, labels=[None] * n, prepared=False)
        rec = TemporalRecovery(aligner=build_align("identity"), tau=self.b.tau, gamma=0.95,
                               num_classes=self.nc, instrument_ids=self.inst)
        rows = []
        stream = self.b.stream(seq, batch_size=1, with_gt=False)
        for i in range(n):
            progress(i, n, f"Frame {i + 1} of {n}: segmentation and gate")
            t0 = time.perf_counter()
            state = next(stream)
            sec["segment_gate"] += time.perf_counter() - t0
            base = state.mask

            progress(i, n, f"Frame {i + 1} of {n}: stereo evidence")
            t0 = time.perf_counter()
            L = cv2.imread(lefts[i], cv2.IMREAD_COLOR)
            R = cv2.imread(right_path(lefts[i]), cv2.IMREAD_COLOR)
            ev = compute_evidence(L, R, rect, corr, self.sgbm, (H, W))
            sec["stereo"] += time.perf_counter() - t0
            evid = dict(lr_ok=ev.lr_ok.astype(bool), occluded=ev.occluded.astype(bool),
                        confidence=ev.confidence, in_view=ev.in_view.astype(bool))
            st3 = np.asarray(instrument_stereo_indicators(base, evid["lr_ok"], evid["occluded"],
                                                          evid["confidence"], evid["in_view"], self.inst), np.float32)
            sc = stereo_cues(evid["lr_ok"], evid["occluded"], ev.specular.astype(bool), evid["in_view"])
            cue_raw = np.array([sc["stereo_lr_fail"], sc["occluded"], sc["specular"], *ev.image_cues], np.float32)

            routed = bool(state.reliability < self.tau and not state.degenerate)
            masks: Dict[str, Optional[np.ndarray]] = {}
            feats: Dict[str, np.ndarray] = {}
            if routed:
                progress(i, n, f"Frame {i + 1} of {n}: candidate actions")
                t0 = time.perf_counter()
                lab_r, conf_r = self._segment_right(right_path(lefts[i]))
                sec["right_view"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                disp = ev.disparity.astype(np.float32)
                rl = right_to_left_grid(lab_r, disp, rect, corr, (H, W), cv2.INTER_NEAREST, 255)
                rc = right_to_left_grid(conf_r, disp, rect, corr, (H, W), cv2.INTER_NEAREST, 0)
                ctx = ActionContext(state=state, evidence=evid, right_label=rl, right_conf=rc,
                                    target_names=self.names, instrument_ids=self.inst, recovery=rec)
                for a in CANDIDATES:
                    masks[a] = ACTIONS[a](ctx, self.params)
                    feats[a] = (action_features(ctx, masks[a]) if masks[a] is not None
                                else np.full(8, np.nan, np.float32))
                sec["actions"] += time.perf_counter() - t0
            rows.append(dict(state=state, st3=st3, cue_raw=cue_raw, routed=routed, masks=masks, feats=feats))

        # selector: needs the frame's difficulty score, which ranks the upload's own blur within the upload
        progress(n, n, "Choosing actions")
        t0 = time.perf_counter()
        raw = np.stack([r["cue_raw"] for r in rows])
        diff = self.diff_index.score(_cues(raw, np.array(["upload"] * n)))
        F = np.stack([np.concatenate([r["state"].indicators, r["st3"], [r["state"].reliability], [diff[i]]])
                      for i, r in enumerate(rows)]).astype(np.float32)
        G = {a: np.stack([rows[i]["feats"].get(a, np.full(8, np.nan, np.float32)) for i in range(n)]) for a in CANDIDATES}
        for a in CANDIDATES:      # an action that was not computed (frame not routed) is unavailable
            for i in range(n):
                if a not in rows[i]["masks"] or rows[i]["masks"][a] is None:
                    G[a][i] = np.nan
        choice = self.selector.choose(self.selector.predict(F, G))
        sec["selector"] += time.perf_counter() - t0

        # final masks, identity verification, scoring
        out_rows, f_before, f_after, cls_before, cls_after = [], [], [], [], []
        for i, r in enumerate(rows):
            st = r["state"]
            c = str(choice[i]) if (r["routed"] and r["masks"].get(str(choice[i])) is not None) else "A0"
            final = st.mask if c == "A0" else r["masks"][c]
            t0 = time.perf_counter()
            idr = self.identity(final, st)
            sec["identity"] += time.perf_counter() - t0
            row = dict(frame=stems[i], reliability=float(st.reliability), flagged=bool(r["routed"]),
                       cold_start=bool(st.cold_start), action=c, changed=bool((final != st.mask).any()),
                       identity_regions=len(idr.assignments), identity_uncertain=len(idr.uncertain),
                       identity_mask_changed=bool((idr.mask != final).any()))
            if has_labels:
                gt_full = read_mask(label_paths[i])
                gt_w = cv2.resize(gt_full, (W, H), interpolation=cv2.INTER_NEAREST)
                ib = frame_iou(st.mask, gt_w, self.nc, class_ids=self.inst)
                ia = frame_iou(final, gt_w, self.nc, class_ids=self.inst)
                full_b = cv2.resize(st.mask, (gt_full.shape[1], gt_full.shape[0]), interpolation=cv2.INTER_NEAREST)
                full_a = full_b if c == "A0" else cv2.resize(final, (gt_full.shape[1], gt_full.shape[0]), interpolation=cv2.INTER_NEAREST)
                cb, ca = official_class_ious(full_b, gt_full, self.nc), official_class_ious(full_a, gt_full, self.nc)
                fb = float(np.mean(list(cb.values()))) if cb else None
                fa = float(np.mean(list(ca.values()))) if ca else None
                d = ia - ib
                row.update(instr_iou_before=float(ib), instr_iou_after=float(ia), delta=float(d),
                           effect=("helped" if d > MARGIN_EVAL else "hurt" if d < -MARGIN_EVAL else "unchanged") if c != "A0" else "no change",
                           official_before=fb, official_after=fa)
                f_before.append(fb); f_after.append(fa); cls_before.append(cb); cls_after.append(ca)
            out_rows.append(row)
            progress(i + 1, n, "Identity verification and scoring")

        total = time.time() - t_start
        mean_ms = {k: 1000.0 * v / n for k, v in sec.items()}
        res = dict(n_frames=n, has_labels=has_labels, tau=self.tau, device=str(self.b.device),
                   total_seconds=total, seconds_per_frame=total / n, mean_latency_ms=sum(mean_ms.values()),
                   stage_ms_per_frame=mean_ms, rows=out_rows,
                   cold_start_frames=int(sum(r["cold_start"] for r in out_rows)),
                   self_calibration=dict(frames_used=len(refine), applied=bool(corr.applied), n_matches=int(corr.n_matches)),
                   n_selector_training_frames=self.n_selector_frames, memory_size=self.b.memory_size,
                   identity=dict(regions=sum(r["identity_regions"] for r in out_rows),
                                 uncertain=sum(r["identity_uncertain"] for r in out_rows),
                                 mask_changed_frames=sum(r["identity_mask_changed"] for r in out_rows)))
        act = [r for r in out_rows if r["action"] != "A0"]
        res["activation_rate"] = len(act) / n
        res["flagged_rate"] = sum(r["flagged"] for r in out_rows) / n
        res["action_counts"] = {a: sum(r["action"] == a for r in out_rows) for a in ("A0",) + CANDIDATES}
        if has_labels:
            ok = [i for i in range(n) if f_before[i] is not None]
            miou_b = float(np.mean([f_before[i] for i in ok])) if ok else None
            miou_a = float(np.mean([f_after[i] for i in ok])) if ok else None
            pc_b, pc_a, per_class = {}, {}, []
            for c in range(1, self.nc):
                vb = [x[c] for x in cls_before if c in x]; va = [x[c] for x in cls_after if c in x]
                if vb:
                    per_class.append(dict(id=c, name=CLASS_NAMES[c], n_frames=len(vb),
                                          iou_before=float(np.mean(vb)), iou_after=float(np.mean(va))))
            mc_b = float(np.mean([p["iou_before"] for p in per_class])) if per_class else None
            mc_a = float(np.mean([p["iou_after"] for p in per_class])) if per_class else None
            changed = [r for r in out_rows if r["action"] != "A0" and "delta" in r]
            hurt = [r for r in changed if r["effect"] == "hurt"]
            helped = [r for r in changed if r["effect"] == "helped"]
            fo = [(f_after[i] - f_before[i]) for i in ok if out_rows[i]["action"] != "A0" and f_after[i] is not None]
            hurt_off = sum(1 for d in fo if d < -MARGIN_EVAL)
            res["scores"] = dict(
                official_miou_before=miou_b, official_miou_after=miou_a, mciou_before=mc_b, mciou_after=mc_a,
                n_scored=len(ok), n_corrected=len(changed), helped=len(helped), hurt=len(hurt),
                unchanged=len(changed) - len(helped) - len(hurt),
                harmful_of_corrected=(len(hurt) / len(changed)) if changed else 0.0,
                harmful_of_all=len(hurt) / n,
                harmful_official_of_corrected=(hurt_off / len(fo)) if fo else 0.0,
                harmful_official_of_all=hurt_off / n,
                margin=MARGIN_EVAL, per_class=per_class)
        return res
