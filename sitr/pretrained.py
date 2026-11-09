"""
Pretrained-weight loading for the CAFormer encoders, with a HARD check.

Why this file exists: the upstream SurgeNet loader calls
`load_state_dict(..., strict=False)` on whatever the checkpoint contains. If the
key names do not line up -- a `backbone.` or `module.` prefix from DINO
training, or a {"teacher": ...} wrapper -- *nothing* loads, no error is raised,
and you train from random weights while believing you used a surgical
foundation model. Every number downstream would be quietly wrong.

`load_encoder_weights` instead:
  1. unwraps the common checkpoint containers,
  2. strips the common prefixes,
  3. counts how many of the encoder's tensors were actually matched with the
     right shape, and
  4. RAISES unless essentially all of them were (classification-head tensors
     are exempt: segmentation does not use them).

Weight sources (URLs exactly as published in TimJaspers0801/SurgeNet
load_models.py):
  surgenetxl  CAFormer-S18, DINO-pretrained on SurgeNetXL (4.7M surgical frames)
              licence CC-BY-NC-SA 4.0 -- non-commercial
  imagenet    CAFormer-S18, ImageNet-1k (sail-sg)
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

WEIGHT_URLS: Dict[str, Tuple[str, str]] = {
    # name: (url, local file name)
    "surgenetxl": (
        "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/"
        "SurgeNetXL_checkpoint_epoch0050_teacher.pth?download=true",
        "SurgeNetXL_checkpoint_epoch0050_teacher.pth"),
    "imagenet": (
        "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth",
        "caformer_s18_imagenet1k.pth"),
}

# which CAFormer activation configuration each weight family was trained with
VARIANT_OF = {"surgenetxl": "surgenet", "imagenet": "imagenet"}

_CONTAINERS = ("teacher", "student", "model", "state_dict", "model_state",
               "module")
_PREFIXES = ("module.", "backbone.", "encoder.", "model.", "metaformer.")
_EXEMPT = ("head.", "norm.")     # classifier / pooled-output norm: unused here

MIN_MATCH_FRACTION = 0.99


def _unwrap(obj):
    """Descend into {'teacher': {...}} style containers until we reach tensors."""
    for _ in range(4):
        if isinstance(obj, dict) and obj and not any(
                torch.is_tensor(v) for v in obj.values()):
            for k in _CONTAINERS:
                if k in obj and isinstance(obj[k], dict):
                    obj = obj[k]
                    break
            else:
                break
    if not isinstance(obj, dict):
        raise ValueError("checkpoint does not contain a state dict")
    return obj


def _strip(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if key.startswith(p):
                key, changed = key[len(p):], True
    return key


def fetch(source: str, path: Optional[str] = None,
          cache_dir: Optional[str] = None) -> str:
    """Return a local path to the weights, downloading once if needed."""
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--pretrained-path {path!r} does not exist")
        return path
    if source not in WEIGHT_URLS:
        raise ValueError(f"unknown weight source {source!r}; "
                         f"choose from {sorted(WEIGHT_URLS)}")
    url, fname = WEIGHT_URLS[source]
    cache_dir = cache_dir or os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, fname)
    if not os.path.isfile(dst):
        print(f"downloading {source} weights -> {dst}")
        try:
            torch.hub.download_url_to_file(url, dst, progress=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"could not download {source} weights ({e}). Download this "
                f"file in a browser:\n  {url}\nand pass it with "
                f"--pretrained-path <file>.") from e
    return dst


def load_encoder_weights(encoder: nn.Module, source: str,
                         path: Optional[str] = None,
                         cache_dir: Optional[str] = None,
                         verbose: bool = True) -> Dict[str, int]:
    """Load pretrained weights into `encoder` or raise. Returns match stats."""
    local = fetch(source, path, cache_dir)
    try:
        raw = torch.load(local, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(local, map_location="cpu")
    sd = {_strip(k): v for k, v in _unwrap(raw).items() if torch.is_tensor(v)}

    target = encoder.state_dict()
    needed = [k for k in target if not k.startswith(_EXEMPT)]
    matched, wrong_shape, missing = {}, [], []
    for k in needed:
        if k in sd:
            if tuple(sd[k].shape) == tuple(target[k].shape):
                matched[k] = sd[k]
            else:
                wrong_shape.append(f"{k}: ckpt {tuple(sd[k].shape)} vs model "
                                   f"{tuple(target[k].shape)}")
        else:
            missing.append(k)

    frac = len(matched) / max(len(needed), 1)
    stats = {"matched": len(matched), "needed": len(needed),
             "wrong_shape": len(wrong_shape), "missing": len(missing),
             "checkpoint_tensors": len(sd)}
    if frac < MIN_MATCH_FRACTION:
        sample_ckpt = list(sd)[:8]
        raise RuntimeError(
            f"pretrained weights '{source}' do NOT match the encoder: only "
            f"{len(matched)}/{len(needed)} tensors matched ({frac:.1%}).\n"
            f"  missing (first 5): {missing[:5]}\n"
            f"  wrong shape (first 3): {wrong_shape[:3]}\n"
            f"  checkpoint keys look like: {sample_ckpt}\n"
            "Refusing to continue: training on top of this would silently "
            "start from random weights.")
    encoder.load_state_dict(matched, strict=False)
    if verbose:
        print(f"pretrained '{source}': loaded {len(matched)}/{len(needed)} "
              f"encoder tensors ({frac:.1%}) from {os.path.basename(local)}")
    return stats
