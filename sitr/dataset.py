"""
EndoVis 2018 Robotic Scene Segmentation — dataset layer.

Real on-disk layout of the challenge download (as distributed, unmodified):

    endovis2018_cvdataset/
      miccai_challenge_2018_release_1/
        __MACOSX/miccai_challenge_2018_release_1/seq_1/...   <- JUNK, excluded
        miccai_challenge_2018_release_1/
          labels.json
          seq_1/ seq_2/ seq_3/ seq_4/
            camera_calibration.txt
            left_frames/frame000.png … frame148.png   1280x1024
            right_frames/…                            (unused: labels are left-eye)
            labels/frame000.png …                     RGB colour-coded
      miccai_challenge_release_2/miccai_challenge_release_2/   seq_5 … seq_7
      miccai_challenge_release_3/miccai_challenge_release_3/   seq_9 … seq_12
      miccai_challenge_release_4/miccai_challenge_release_4/   seq_13 … seq_16
      repairs/repairs/seq_1_frame042.png …                     <- CORRECTED labels
      test_data/test_data/
        labels.json  utils.py  run.py  requirements.txt
        seq_1-<timestamp>-001/seq_1/{camera_calibration.txt,labels,left_frames,right_frames}
        … seq_4                                        4 x 250 frames

Facts encoded here, all verified against the download rather than assumed:

* There is **no seq_8**. Releases 1-4 give sequences 1-7 and 9-16 = 15 training
  sequences of 149 annotated frames each (2235 frames).
* `__MACOSX/` mirrors the whole tree with 1 KB AppleDouble `._*` files. Those
  directories contain real `labels/`, `left_frames/` and `right_frames/`
  folders, so any discovery that just looks for `left_frames` finds four
  phantom copies of seq_1…seq_4 and trains on junk. Excluded.
* `repairs/` holds the seven corrected ground-truth masks the challenge
  published for release 1 (seq_1 frames 042/043/044/073, seq_4 frames
  135/137/138). They are applied as a non-destructive override: your files on
  disk are never modified.
* Training sequence `seq_1` and test sequence `seq_1` are different procedures.
  Sequences are keyed `<split>/<name>` so the two can never be confused.

Splits are made at the SEQUENCE level. At 1 Hz adjacent frames are still highly
correlated and a random frame-level split leaks (report Sec. V-B.5).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import (ClassMap, IGNORE_INDEX, decode_label_image, is_junk_path,
                     prune_junk, task_remap)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_FRAME_DIRS = ("left_frames", "images", "left_frame", "frames", "img")
_LABEL_DIRS = ("labels", "label", "ground_truth", "gt", "annotations", "lbl")
_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

cv2.setNumThreads(0)   # avoid thread oversubscription inside DataLoader workers


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
@dataclass
class SequenceInfo:
    name: str                       # "seq_1"
    split: str                      # "train" | "test"
    index: int                      # 1
    frame_dir: str
    label_dir: Optional[str]
    frames: List[str]
    labels: List[Optional[str]]
    release: str = ""               # "miccai_challenge_2018_release_1"
    prepared: bool = False          # labels already canonical single-channel ids
    repaired: List[int] = field(default_factory=list)   # frame indices overridden

    @property
    def key(self) -> str:
        return f"{self.split}/{self.name}"

    def __len__(self) -> int:
        return len(self.frames)


def _first_existing(parent: str, names: Sequence[str]) -> Optional[str]:
    for n in names:
        p = os.path.join(parent, n)
        if os.path.isdir(p):
            return p
    return None


def _sorted_images(d: str) -> List[str]:
    files = [f for f in os.listdir(d)
             if f.lower().endswith(_IMG_EXT) and not f.startswith("._")]

    def key(f: str):
        nums = re.findall(r"\d+", f)
        return ([int(n) for n in nums], f)

    return [os.path.join(d, f) for f in sorted(files, key=key)]


def _seq_index(name: str) -> int:
    m = re.search(r"seq[_-]?(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def _sequence_name(dirpath: str) -> str:
    """Innermost `seq_N` component, e.g. `seq_1-2026…-001/seq_1` -> `seq_1`."""
    base = os.path.basename(dirpath.rstrip(os.sep))
    m = re.match(r"(seq[_-]?\d+)", base, re.IGNORECASE)
    if m:
        return m.group(1).lower().replace("-", "_")
    return base


def _release_of(dirpath: str, root: str) -> str:
    rel = os.path.relpath(dirpath, root)
    parts = [p for p in rel.split(os.sep) if p not in (".", "")]
    for p in parts:
        if p.lower().startswith(("miccai", "test_data")):
            return p
    return parts[0] if parts else ""


def _split_of(dirpath: str, root: str) -> str:
    """Classify a sequence as train or test from its path RELATIVE to the root.

    Relative, not absolute: a dataset living under a directory whose name
    happens to contain "test" (C:/latest/endovis..., or a repo's tests/ folder)
    would otherwise have every sequence misfiled as test, silently leaving the
    training split empty.
    """
    try:
        rel = os.path.relpath(dirpath, root)
    except ValueError:          # different drives on Windows
        rel = dirpath
    parts = [p.lower() for p in os.path.normpath(rel).split(os.sep)
             if p not in (".", "..", "")]
    # An exact "train" component wins: a prepared cache written under a folder
    # whose name merely contains "test" must not be misfiled as the test split.
    if "train" in parts:
        return "train"
    return "test" if any("test" in p for p in parts) else "train"


# --------------------------------------------------------------------------- #
# repairs
# --------------------------------------------------------------------------- #
_REPAIR_RE = re.compile(r"(seq[_-]?\d+)[_-]?(frame\d+)", re.IGNORECASE)


def find_repairs(root: str) -> Dict[Tuple[str, str], str]:
    """Map (seq_name, frame_stem) -> corrected label path from `repairs/`.

    The challenge published seven corrected masks for release 1. Training
    without them means training on seven masks the organisers have stated are
    wrong. They are applied as an override; nothing on disk is modified.
    """
    out: Dict[Tuple[str, str], str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        prune_junk(dirnames)
        if is_junk_path(dirpath):
            continue
        if "repair" not in os.path.basename(dirpath).lower():
            continue
        for fn in filenames:
            if fn.startswith("._") or not fn.lower().endswith(_IMG_EXT):
                continue
            m = _REPAIR_RE.match(os.path.splitext(fn)[0])
            if not m:
                continue
            seq = m.group(1).lower().replace("-", "_")
            stem = m.group(2).lower()
            out[(seq, stem)] = os.path.join(dirpath, fn)
    return out


def discover_sequences(root: str,
                       require_labels: bool = True,
                       apply_repairs: bool = True,
                       splits: Sequence[str] = ("train",),
                       prepared: bool = False) -> List[SequenceInfo]:
    """Find every sequence under `root`, junk excluded, repairs applied."""
    repairs = find_repairs(root) if apply_repairs else {}
    want = set(splits)
    seqs: List[SequenceInfo] = []

    for dirpath, dirnames, _files in os.walk(root):
        prune_junk(dirnames)
        if is_junk_path(dirpath):
            continue
        frame_dir = _first_existing(dirpath, _FRAME_DIRS)
        if frame_dir is None:
            continue
        dirnames[:] = []                       # do not descend into a sequence

        split = _split_of(dirpath, root)
        if split not in want:
            continue

        label_dir = _first_existing(dirpath, _LABEL_DIRS)
        if label_dir is None and require_labels:
            continue

        frames = _sorted_images(frame_dir)
        if not frames:
            continue

        name = _sequence_name(dirpath)
        labels: List[Optional[str]] = []
        repaired: List[int] = []

        if label_dir is not None:
            by_stem = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in _sorted_images(label_dir)}
            for i, f in enumerate(frames):
                stem = os.path.splitext(os.path.basename(f))[0]
                rep = repairs.get((name, stem.lower())) if split == "train" else None
                if rep is not None:
                    labels.append(rep)
                    repaired.append(i)
                else:
                    labels.append(by_stem.get(stem))
            missing = [i for i, l in enumerate(labels) if l is None]
            if missing:
                if require_labels and split == "train":
                    # A training frame without a label is corruption: it would
                    # either crash later or, worse, be silently supervised by
                    # nothing. Refuse.
                    raise FileNotFoundError(
                        f"{dirpath}: {len(missing)}/{len(frames)} TRAINING "
                        f"frames have no matching label in {label_dir} "
                        f"(first missing: {os.path.basename(frames[missing[0]])})")

                # The challenge's own test release ships partial labels (test
                # seq_3 has 250 frames and 249 labels). The test split is held
                # out and fits nothing, so an unlabelled frame there is dropped
                # rather than fatal -- but it is reported, and a gap in the
                # MIDDLE of a sequence is called out separately because it
                # breaks temporal contiguity, which every Stage 2 indicator
                # defined against the previous frame depends on.
                interior = [i for i in missing if 0 < i < len(frames) - 1]
                print(f"  ! {os.path.basename(dirpath)} [{split}]: dropping "
                      f"{len(missing)} unlabelled frame(s): "
                      f"{[os.path.basename(frames[i]) for i in missing[:5]]}"
                      f"{' ...' if len(missing) > 5 else ''}")
                if interior:
                    print(f"  !! {len(interior)} of them are INTERIOR frames -- "
                          f"this creates a temporal gap in {os.path.basename(dirpath)}; "
                          f"T_t and Drift are computed across it")
                keep = [i for i, l in enumerate(labels) if l is not None]
                remap = {old_i: new_i for new_i, old_i in enumerate(keep)}
                frames = [frames[i] for i in keep]
                labels = [labels[i] for i in keep]
                repaired = [remap[i] for i in repaired if i in remap]
                if not frames:
                    continue
        else:
            labels = [None] * len(frames)

        seqs.append(SequenceInfo(
            name=name, split=split, index=_seq_index(name),
            frame_dir=frame_dir, label_dir=label_dir,
            frames=frames, labels=labels,
            release=_release_of(dirpath, root), prepared=prepared,
            repaired=repaired,
        ))

    if not seqs:
        raise FileNotFoundError(
            f"No {'/'.join(sorted(want))} sequences found under {root!r}. "
            "Expected folders containing a 'left_frames' subdirectory.")

    # Two sequences with the same key would silently shadow each other.
    seen: Dict[str, str] = {}
    for s in seqs:
        if s.key in seen:
            raise ValueError(
                f"duplicate sequence key {s.key!r}:\n  {seen[s.key]}\n  "
                f"{s.frame_dir}\nThis usually means a junk or backup copy of the "
                "dataset is nested under the root.")
        seen[s.key] = s.frame_dir

    seqs.sort(key=lambda s: (s.split, s.index, s.name))
    return seqs


def load_prepared(prepared_root: str,
                  splits: Sequence[str] = ("train",),
                  expect_task: Optional[str] = None,
                  expect_hw: Optional[Tuple[int, int]] = None
                  ) -> Tuple[List[SequenceInfo], Dict]:
    """Load a cache written by tools/prepare_dataset.py.

    The cached labels hold TARGET ids for one specific task at one specific
    resolution, so both are verified here rather than silently mismatching.
    """
    meta_path = os.path.join(prepared_root, "prepared_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"{prepared_root!r} is not a prepared cache (no prepared_meta.json). "
            "Build it with tools/prepare_dataset.py, or pass --root instead.")
    with open(meta_path) as f:
        meta = json.load(f)

    if expect_task and meta.get("task") != expect_task:
        raise ValueError(
            f"prepared cache was built for task={meta.get('task')!r} but "
            f"task={expect_task!r} was requested. Re-run prepare_dataset.py.")
    if expect_hw and (int(meta.get("height", -1)), int(meta.get("width", -1))) != expect_hw:
        raise ValueError(
            f"prepared cache is {meta.get('height')}x{meta.get('width')} but "
            f"{expect_hw[0]}x{expect_hw[1]} was requested. Re-run prepare_dataset.py.")

    seqs = discover_sequences(prepared_root, require_labels=True,
                              apply_repairs=False, splits=splits, prepared=True)
    return seqs, meta


def split_sequences(seqs: List[SequenceInfo],
                    val_indices: Sequence[int]) -> Tuple[List[SequenceInfo],
                                                         List[SequenceInfo]]:
    val_set = {int(v) for v in val_indices}
    present = {s.index for s in seqs}
    unknown = val_set - present
    if unknown:
        raise ValueError(
            f"requested validation sequences {sorted(unknown)} are not present. "
            f"Available: {sorted(present)} (note: EndoVis 2018 has no seq_8)")
    train = [s for s in seqs if s.index not in val_set]
    val = [s for s in seqs if s.index in val_set]
    if not train or not val:
        raise ValueError("train or validation split is empty")
    return train, val


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
class Preprocessor:
    """Resize + ImageNet normalisation. Masks always nearest-neighbour."""

    def __init__(self, height: int = 256, width: int = 320):
        if height % 32 or width % 32:
            raise ValueError("height and width must be multiples of 32 "
                             "(the VGG encoder downsamples by 32x)")
        self.h, self.w = int(height), int(width)

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] == (self.h, self.w):
            return img
        interp = cv2.INTER_AREA if img.shape[0] > self.h else cv2.INTER_LINEAR
        return cv2.resize(img, (self.w, self.h), interpolation=interp)

    def resize_mask(self, m: np.ndarray) -> np.ndarray:
        if m.shape[:2] == (self.h, self.w):
            return m
        return cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

    def normalise(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(x.transpose(2, 0, 1))


class Augmenter:
    """Geometric + photometric augmentation, OpenCV only (no extra deps).

    The photometric ranges deliberately mimic the difficult-visibility causes
    the report targets — glare, underexposure, low contrast, motion blur — so
    the base model sees them during training and the reliability estimator gets
    a realistic spread of achieved IoU to regress. Vertical flips are NOT used:
    endoscopic scenes have a consistent gravity/camera convention.
    """

    def __init__(self, hflip=0.5, scale=0.20, rotate=15.0, shift=0.06,
                 brightness=0.25, contrast=0.25, gamma=0.25,
                 blur_p=0.15, glare_p=0.10, seed: Optional[int] = None):
        self.hflip, self.scale, self.rotate, self.shift = hflip, scale, rotate, shift
        self.brightness, self.contrast, self.gamma = brightness, contrast, gamma
        self.blur_p, self.glare_p = blur_p, glare_p
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        h, w = img.shape[:2]

        if self.rng.random() < self.hflip:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        if self.scale > 0 or self.rotate > 0 or self.shift > 0:
            s = 1.0 + float(self.rng.uniform(-self.scale, self.scale))
            a = float(self.rng.uniform(-self.rotate, self.rotate))
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), a, s)
            M[0, 2] += float(self.rng.uniform(-self.shift, self.shift)) * w
            M[1, 2] += float(self.rng.uniform(-self.shift, self.shift)) * h
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Pixels rotated in from outside have no label: mark them ignore,
            # never background, or the model is trained to predict background
            # on invented content.
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=int(IGNORE_INDEX))

        x = img.astype(np.float32)
        if self.brightness > 0:
            x += 255.0 * float(self.rng.uniform(-self.brightness, self.brightness))
        if self.contrast > 0:
            f = 1.0 + float(self.rng.uniform(-self.contrast, self.contrast))
            m = float(x.mean())
            x = (x - m) * f + m
        x = np.clip(x, 0, 255)
        if self.gamma > 0:
            g = float(np.exp(self.rng.uniform(-self.gamma, self.gamma)))
            x = 255.0 * np.power(x / 255.0, g)
        x = np.clip(x, 0, 255).astype(np.uint8)

        if self.blur_p > 0 and self.rng.random() < self.blur_p:
            k = int(self.rng.choice([5, 7, 9, 11]))
            kern = np.zeros((k, k), np.float32)
            ang = float(self.rng.uniform(0, 180))
            cv2.line(kern, (0, k // 2), (k - 1, k // 2), 1.0, 1)
            kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D(
                (k / 2 - 0.5, k / 2 - 0.5), ang, 1.0), (k, k))
            ssum = kern.sum()
            if ssum > 0:
                x = cv2.filter2D(x, -1, kern / ssum)

        if self.glare_p > 0 and self.rng.random() < self.glare_p:
            gy = int(self.rng.integers(0, h))
            gx = int(self.rng.integers(0, w))
            rad = int(self.rng.integers(max(4, w // 20), max(6, w // 6)))
            overlay = x.copy()
            cv2.circle(overlay, (gx, gy), rad, (255, 255, 255), -1)
            overlay = cv2.GaussianBlur(overlay, (0, 0), rad / 3.0 + 1.0)
            alpha = float(self.rng.uniform(0.25, 0.6))
            x = cv2.addWeighted(overlay, alpha, x, 1 - alpha, 0)

        return x, mask


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
class _Base(Dataset):
    def __init__(self, sequences: List[SequenceInfo], class_map: ClassMap,
                 task: str = "binary", height: int = 256, width: int = 320):
        if not sequences:
            raise ValueError("no sequences given")
        self.sequences = sequences
        self.class_map = class_map
        self.task = task
        self.pre = Preprocessor(height, width)

        self.remap, self.target_names, self.instrument_ids = task_remap(class_map, task)
        self.num_classes = len(self.target_names)
        self._rgb_lut = class_map.rgb_lut() if class_map.encoding == "rgb" else None
        self._id_lut = class_map.id_lut() if class_map.encoding == "id" else None

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"unreadable image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path: Optional[str], prepared: bool) -> Optional[np.ndarray]:
        if path is None:
            return None
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"unreadable label: {path}")
        if prepared:
            # already canonical target ids written by tools/prepare_dataset.py
            return raw if raw.ndim == 2 else raw[:, :, 0]
        if raw.ndim == 3 and raw.shape[2] >= 3:
            raw = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
        canonical = decode_label_image(raw, self.class_map,
                                       rgb_lut=self._rgb_lut, id_lut=self._id_lut)
        out = np.full(canonical.shape, IGNORE_INDEX, dtype=np.uint8)
        known = canonical != IGNORE_INDEX
        out[known] = self.remap[canonical[known]]
        return out


class FrameDataset(_Base):
    """Shuffled frame-level view, used for Stage 1 training."""

    def __init__(self, sequences, class_map, task="binary", height=256, width=320,
                 augment: bool = False, seed: Optional[int] = None):
        super().__init__(sequences, class_map, task, height, width)
        self.augment = augment
        self._seed = seed
        self._aug: Optional[Augmenter] = None
        self.items: List[Tuple[int, int]] = [
            (si, fi) for si, s in enumerate(sequences) for fi in range(len(s))]

    def _augmenter(self) -> Augmenter:
        # One RNG per worker process, or every worker draws identical crops.
        if self._aug is None:
            info = torch.utils.data.get_worker_info()
            wid = 0 if info is None else int(info.id)
            base = 0 if self._seed is None else int(self._seed)
            self._aug = Augmenter(seed=base + 9973 * (wid + 1))
        return self._aug

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        si, fi = self.items[i]
        seq = self.sequences[si]
        img = self._read_image(seq.frames[fi])
        mask = self._read_mask(seq.labels[fi], seq.prepared)

        # Resize first: warping at 1280x1024 costs ~8x more for no benefit.
        img = self.pre.resize_image(img)
        mask = self.pre.resize_mask(mask)
        if self.augment:
            img, mask = self._augmenter()(img, mask)

        return {"image": torch.from_numpy(self.pre.normalise(img)),
                "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
                "seq": si, "frame": fi}


class SequenceDataset(_Base):
    """Strict temporal order, no augmentation — builds the Stage 2 cache."""

    def __init__(self, sequence: SequenceInfo, class_map, task="binary",
                 height=256, width=320):
        super().__init__([sequence], class_map, task, height, width)
        self.sequence = sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        img = self._read_image(self.sequence.frames[i])
        mask = self._read_mask(self.sequence.labels[i], self.sequence.prepared)
        img = self.pre.resize_image(img)
        out = {"image": torch.from_numpy(self.pre.normalise(img)), "frame": i,
               "has_mask": mask is not None}
        out["mask"] = (torch.from_numpy(
            np.ascontiguousarray(self.pre.resize_mask(mask))).long()
            if mask is not None else torch.zeros(1, dtype=torch.long))
        return out


# --------------------------------------------------------------------------- #
# class balance (used to weight the CE term)
# --------------------------------------------------------------------------- #
def class_pixel_counts(sequences: List[SequenceInfo], class_map: ClassMap,
                       task: str, height: int, width: int,
                       stride: int = 5) -> np.ndarray:
    """Pixel count per target class over every `stride`-th frame."""
    ds = FrameDataset(sequences, class_map, task, height, width, augment=False)
    counts = np.zeros(ds.num_classes + 1, dtype=np.int64)   # +1 collects ignore
    for i in range(0, len(ds), max(1, stride)):
        m = ds[i]["mask"].numpy()
        m = np.where(m == IGNORE_INDEX, ds.num_classes, m)
        counts += np.bincount(m.ravel(), minlength=ds.num_classes + 1)
    return counts[:ds.num_classes]


def inverse_sqrt_weights(counts: np.ndarray, clip: float = 12.0) -> np.ndarray:
    """Median-frequency-style weights, square-rooted and clipped.

    Plain inverse frequency puts a weight of several hundred on classes like
    suturing-needle and makes training diverge; sqrt with a clip is the usual
    stable compromise and is what the defaults use.
    """
    counts = np.asarray(counts, dtype=np.float64)
    freq = counts / max(counts.sum(), 1.0)
    nz = freq[freq > 0]
    if nz.size == 0:
        return np.ones_like(counts, dtype=np.float32)
    w = np.sqrt(np.median(nz) / np.maximum(freq, 1e-12))
    w = np.clip(w, 1.0 / clip, clip)
    w[freq == 0] = 0.0       # a class absent from training gets no gradient
    return w.astype(np.float32)


def save_manifest(path: str, sequences: List[SequenceInfo], extra: Optional[Dict] = None) -> None:
    payload = {
        "sequences": [
            {"key": s.key, "name": s.name, "split": s.split, "index": s.index,
             "release": s.release, "frames": len(s),
             "frame_dir": s.frame_dir, "label_dir": s.label_dir,
             "repaired_frames": [os.path.basename(s.frames[i]) for i in s.repaired]}
            for s in sequences],
        "total_frames": sum(len(s) for s in sequences),
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
