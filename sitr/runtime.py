"""
THE TEAMMATE'S ENTRY POINT.

`tools/run_all.py` writes a self-contained `release/` folder. This module loads
it and turns it back into a live stream of fully-populated `FrameState` objects
— including the dense decoder features `Phi_t` and the calibrated posterior
`P_t` that eqs. (19)-(22) need, which are too large to cache to disk and are
therefore recomputed from the frozen backbone on the fly.

Everything Stages 1-2 decided is already applied: the segmentation, the nine
indicators, the calibrated confidence, the trained reliability estimator, the
threshold tau, and the bounded reliable memory `B_t`.

    from sitr.runtime import ReleaseBundle

    bundle = ReleaseBundle.load("release")
    runner = bundle.runner(recovery=MyRecovery(), identity=MyIdentity())

    for seq in bundle.sequences(split="train"):
        runner.reset()                       # memory must not cross procedures
        for state in bundle.stream(seq):     # state: FrameState, fully populated
            final_mask, _ = runner.step(state)
            ...
        print(runner.stats)

If `recovery` is omitted you get the honest Stage 1+2 system: reliable frames
accepted, unreliable frames flagged, nothing invented. That runs today and is
the baseline to diff every later change against.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import cv2
import numpy as np
import torch

from .dataset import (IMAGENET_MEAN, IMAGENET_STD, Preprocessor,  # noqa: F401
                      SequenceInfo, discover_sequences, load_prepared)
from .labels import ClassMap
from .metrics import foreground_area, instrument_iou
from .models import ReliabilityMLP, build_model
from .reliability import (SequenceCache, StreamConfig, cosine,  # noqa: F401
                          indicator_names, mask_stability)
from .stage34_interface import FrameState, MemoryEntry, SelectiveInference
from .utils import get_device, load_checkpoint


@dataclass
class ReleaseBundle:
    """Everything Stages 1-2 produced, loaded and ready to run."""
    root: str
    manifest: Dict
    class_map: ClassMap
    model: torch.nn.Module
    reliability: ReliabilityMLP
    device: torch.device
    data_root: Optional[str] = None

    # ------------------------------------------------------------- loading
    @staticmethod
    def load(root: str, device: str = "auto", strict: bool = True,
             data_root: Optional[str] = None) -> "ReleaseBundle":
        """Load a release folder.

        data_root : the prepared image data (a folder containing
            prepared_meta.json). Normally leave it None: the bundle looks for
            `<release>/prepared` first, which is where run_all.py puts it, so a
            release copied to another machine works as-is. Pass it only if you
            keep the data somewhere else.
        """
        man_path = os.path.join(root, "manifest.json")
        if not os.path.isfile(man_path):
            raise FileNotFoundError(
                f"{root!r} is not a release folder (no manifest.json). "
                "Produce one with `python -m tools.run_all ...`.")
        with open(man_path) as f:
            manifest = json.load(f)

        dev = get_device(device)
        cm = ClassMap.from_json(os.path.join(root, "class_map.json"))

        seg_ckpt = load_checkpoint(os.path.join(root, "stage1_best.pt"),
                                   map_location=dev)
        meta = seg_ckpt["meta"]
        model = build_model(int(meta["num_classes"]),
                            arch=meta.get("arch", "resnet34_unet"),
                            pretrained=False).to(dev)
        model.load_state_dict(seg_ckpt["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        rel_ckpt = load_checkpoint(os.path.join(root, "stage2_reliability.pt"),
                                   map_location=dev)
        names = rel_ckpt["meta"]["feature_names"]
        rel = ReliabilityMLP(in_dim=len(names),
                             hidden=int(rel_ckpt["meta"].get("args", {})
                                        .get("hidden", 64))).to(dev)
        rel.load_state_dict(rel_ckpt["model"])
        rel.eval()

        b = ReleaseBundle(root=root, manifest=manifest, class_map=cm,
                          model=model, reliability=rel, device=dev,
                          data_root=data_root)
        if strict:
            b._check()
        return b

    def _check(self) -> None:
        exp = self.manifest["indicator_names"]
        if list(exp) != list(indicator_names(self.manifest["indicator_set"])):
            raise ValueError(
                "manifest indicator names disagree with this code's indicator "
                "set. The release was produced by a different version.")
        if self.reliability.in_dim != len(exp):
            raise ValueError(
                f"reliability model expects {self.reliability.in_dim} inputs "
                f"but the manifest lists {len(exp)} indicators.")

    # ---------------------------------------------------------- properties
    @property
    def tau(self) -> float:
        return float(self.manifest["tau"])

    @property
    def memory_size(self) -> int:
        return int(self.manifest["memory_size"])

    @property
    def mask_scale(self) -> int:
        """Downsample factor the CACHE used when storing masks.

        `tools/cache_features.py --mask-scale` (default 2) stores masks at
        height//scale, and Stage 2 therefore trained on a T_temporal computed
        between DOWNSAMPLED masks.  Recomputing it at full resolution here
        would make the indicator systematically different from the one the
        estimator was fitted on.
        """
        if "mask_scale" in self.manifest:
            return max(1, int(self.manifest["mask_scale"]))
        idx = os.path.join(self.root, "reports", "cache_index.json")
        if os.path.isfile(idx):
            try:
                with open(idx) as f:
                    return max(1, int(json.load(f).get("mask_scale", 2)))
            except Exception:
                pass
        return 2

    @property
    def tau_memory(self) -> float:
        """Threshold for ADMISSION TO MEMORY (eq. 15) -- not the routing tau.

        Stage 2 builds and evaluates history with `tau_train`; the deployed
        `tau` only decides which frames are routed to recovery.  Using `tau`
        for admission makes memory stricter at inference than it was in
        training, which inflates the routing rate through a feedback loop.
        """
        return float(self.manifest.get("tau_train", self.manifest["tau"]))

    @property
    def temperature(self) -> float:
        return float(self.manifest.get("temperature", 1.0))

    @property
    def instrument_ids(self) -> List[int]:
        return [int(c) for c in self.manifest["instrument_ids"]]

    @property
    def num_classes(self) -> int:
        return int(self.manifest["num_classes"])

    @property
    def target_names(self) -> List[str]:
        return list(self.manifest["target_names"])

    # ------------------------------------------------------------ sequences
    def resolve_data_root(self) -> str:
        """Where the prepared images live, checked in this order:

          1. `data_root` passed to load()
          2. the SITR_DATA environment variable
          3. `<release>/prepared`   <- the normal case; makes the release portable
          4. the absolute path recorded in manifest.json on the machine that
             built the release (only valid on that machine)
        """
        cands = [self.data_root, os.environ.get("SITR_DATA"),
                 os.path.join(self.root, "prepared"),
                 self.manifest.get("data_source", {}).get("path")]
        for c in cands:
            if c and os.path.isfile(os.path.join(c, "prepared_meta.json")):
                return c
        raise FileNotFoundError(
            "cannot find the prepared image data this release needs to "
            "recompute P_t and Phi_t. Looked in: "
            + ", ".join(repr(c) for c in cands if c)
            + ". Put the prepared folder at <release>/prepared, or pass "
            "data_root= to ReleaseBundle.load(), or set SITR_DATA.")

    def sequences(self, split: str = "train") -> List[SequenceInfo]:
        src = self.manifest["data_source"]
        if src["kind"] == "prepared":
            seqs, _ = load_prepared(self.resolve_data_root(), (split,),
                                    self.manifest["task"],
                                    (self.manifest["height"], self.manifest["width"]))
            return seqs
        return discover_sequences(src["path"], require_labels=True,
                                  apply_repairs=True, splits=(split,))

    def cache(self, seq: SequenceInfo) -> Optional[SequenceCache]:
        """The cached per-frame scalars for this sequence, if present."""
        p = os.path.join(self.root, "caches", f"{seq.split}__{seq.name}.npz")
        return SequenceCache.load(p) if os.path.isfile(p) else None

    def runner(self, recovery=None, identity=None, aligner=None,
               store_probs: bool = True) -> SelectiveInference:
        return SelectiveInference(
            self.model, self.reliability,
            instrument_ids=self.instrument_ids, num_classes=self.num_classes,
            tau=self.tau, memory_size=self.memory_size,
            tau_memory=self.tau_memory, aligner=aligner,
            recovery=recovery, identity=identity, device=str(self.device),
            store_probs=store_probs)

    # -------------------------------------------------------------- stream
    @torch.no_grad()
    def stream(self, seq: SequenceInfo, batch_size: int = 4,
               with_gt: bool = True) -> Iterator[FrameState]:
        """Yield a fully-populated FrameState per frame, in temporal order.

        `probs` and `features` are recomputed here rather than read from disk:
        a dense posterior and a 32-channel feature map per frame would be tens
        of gigabytes for 2235 frames, and the backbone is frozen so recomputing
        them is exact, not an approximation.

        The reliable memory and the indicators T/A/F are maintained across the
        sequence exactly as Stage 2 did, using this bundle's own tau, so the
        routing decisions you see match the ones stage2_report.json describes.
        """
        from .dataset import SequenceDataset
        from torch.utils.data import DataLoader

        ds = SequenceDataset(seq, self.class_map, self.manifest["task"],
                             self.manifest["height"], self.manifest["width"])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

        names = list(self.manifest["indicator_names"])
        inst_ids = self.instrument_ids
        K = self.memory_size
        tau_mem = self.tau_memory      # eq. (15) admission -- NOT the routing tau
        temp = self.temperature
        ms = self.mask_scale
        sh = max(1, int(self.manifest["height"]) // ms)
        sw = max(1, int(self.manifest["width"]) // ms)

        def _at_cache_scale(m):
            """Masks must be compared at the resolution the cache used."""
            if ms <= 1:
                return m
            return cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
        log_c = float(np.log(max(self.num_classes, 2)))

        memory: List[MemoryEntry] = []
        prev_emb: Optional[np.ndarray] = None
        t = 0

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            logits, feats = self.model(images)
            probs = torch.softmax(logits.float() / temp, dim=1)
            conf, pred = probs.max(dim=1)
            feats = feats.float()

            inst = torch.zeros_like(pred, dtype=torch.bool)
            for c in inst_ids:
                inst |= (pred == c)
            fgm = inst.to(probs.dtype)
            denom = fgm.sum(dim=(1, 2)).clamp_min(1e-6)

            q_b = ((conf * fgm).sum(dim=(1, 2)) / denom).cpu().numpy()
            emb_b = ((feats * fgm.unsqueeze(1)).sum(dim=(2, 3))
                     / denom.unsqueeze(1)).cpu().numpy()
            ent_map = -(probs * probs.clamp_min(1e-8).log()).sum(1) / log_c
            ent_b = (1.0 - (ent_map * fgm).sum(dim=(1, 2)) / denom).cpu().numpy()
            top2 = probs.topk(2, dim=1).values
            mar_b = (((top2[:, 0] - top2[:, 1]) * fgm).sum(dim=(1, 2))
                     / denom).cpu().numpy()

            probs_np = probs.cpu().numpy()
            feats_np = feats.cpu().numpy()
            pred_np = pred.cpu().numpy().astype(np.uint8)
            inst_np = inst.cpu().numpy()
            imgs_np = images.cpu().numpy()
            gts = batch["mask"].numpy().astype(np.uint8) if with_gt else None

            for i in range(pred_np.shape[0]):
                while memory and memory[0].t < t - K:
                    memory.pop(0)

                p, fg = pred_np[i], inst_np[i]
                a = foreground_area(p, inst_ids)
                emb = emb_b[i] if a > 0 else np.zeros_like(emb_b[i])
                vals = {n: 0.0 for n in names}

                if a > 0:
                    vals["q_confidence"] = float(q_b[i])
                    if "H_entropy_conf" in vals:
                        vals["H_entropy_conf"] = float(ent_b[i])
                    if "M_margin" in vals:
                        vals["M_margin"] = float(mar_b[i])
                    if "Bnd_compactness" in vals or "Frag_largest_cc" in vals:
                        cmp_, frg = _compact_frag(p, fg)
                        vals["Bnd_compactness"] = cmp_
                        vals["Frag_largest_cc"] = frg
                    if "Drift_prev_cos" in vals and prev_emb is not None:
                        vals["Drift_prev_cos"] = cosine(emb, prev_emb)
                    if memory:
                        tp = memory[-1]
                        vals["T_temporal"] = instrument_iou(
                            _at_cache_scale(p), _at_cache_scale(tp.mask),
                            self.num_classes, inst_ids)
                        vals["A_area_stability"] = mask_stability(a, tp.area)
                        vals["F_feature_cos"] = cosine(
                            emb, np.mean([m.embedding for m in memory], axis=0))

                vec = np.array([vals[n] for n in names], dtype=np.float32)
                r = float(self.reliability(
                    torch.from_numpy(vec).view(1, -1).to(self.device)).item())

                img = imgs_np[i].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
                state = FrameState(
                    t=t, image=np.clip(img * 255, 0, 255).astype(np.uint8),
                    probs=probs_np[i], mask=p, features=feats_np[i],
                    embedding=emb, indicators=tuple(vec.tolist()),
                    reliability=r, reliable=r >= self.tau, foreground_area=int(a),
                    degenerate=(a == 0), cold_start=(len(memory) == 0),
                    memory=list(memory))
                if gts is not None:
                    state.ground_truth = gts[i]   # offline evaluation ONLY
                yield state

                if r >= tau_mem:
                    # Admission uses tau_memory (eq. 15), the same threshold
                    # Stage 2 trained and evaluated with.  `state.reliable`
                    # (r >= tau) is the ROUTING decision of eq. (2) and is a
                    # different question; see SelectiveInference.__init__.
                    # Full entry M_s = {S_s, P_s, E_s, ID_s, r_s} of eq. (14).
                    # P_s and the grayscale frame are carried so that a caller
                    # iterating stream() directly -- without SelectiveInference
                    # -- still has everything eqs. (16) and (18) need.
                    gray_s = cv2.cvtColor(state.image, cv2.COLOR_RGB2GRAY)
                    memory.append(MemoryEntry(
                        t=t, mask=p, probs=probs_np[i], embedding=emb,
                        identities={}, reliability=r, area=float(a),
                        gray=gray_s))
                    while len(memory) > K:
                        memory.pop(0)
                prev_emb = emb
                t += 1


def _compact_frag(pred: np.ndarray, fg: np.ndarray):
    """Same computation as tools/cache_features.py, kept in sync by tests."""
    n_fg = int(fg.sum())
    if n_fg == 0:
        return 0.0, 0.0
    b = np.zeros_like(fg)
    d_v = pred[1:, :] != pred[:-1, :]
    d_h = pred[:, 1:] != pred[:, :-1]
    b[1:, :] |= d_v
    b[:-1, :] |= d_v
    b[:, 1:] |= d_h
    b[:, :-1] |= d_h
    compact = float(np.clip(1.0 - float((b & fg).sum()) / float(n_fg), 0.0, 1.0))
    n_lab, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    if n_lab <= 1:
        return compact, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return compact, float(sizes.max()) / float(max(sizes.sum(), 1))


__all__ = ["ReleaseBundle"]
