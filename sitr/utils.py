"""Small shared helpers: seeding, device selection, checkpoints, logging."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(arg: str = "auto") -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_enabled(device: torch.device, requested: bool) -> bool:
    return bool(requested and device.type == "cuda")


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


def save_checkpoint(path: str, model: torch.nn.Module, meta: Dict[str, Any],
                    optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"model": model.state_dict(), "meta": meta}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        return torch.load(path, map_location=map_location)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=default)


def cosine_lr(optimizer: torch.optim.Optimizer, base_lrs, step: int,
              total_steps: int, warmup: int = 0,
              min_lr_factor: float = 0.01) -> List[float]:
    """Linear warmup then cosine decay, per parameter group.

    `base_lrs` may be a scalar or one value per group, so an encoder group with
    a reduced learning rate keeps its ratio across the whole schedule.
    """
    if np.isscalar(base_lrs):
        base_lrs = [float(base_lrs)] * len(optimizer.param_groups)
    if len(base_lrs) != len(optimizer.param_groups):
        raise ValueError(f"expected {len(optimizer.param_groups)} base lrs, "
                         f"got {len(base_lrs)}")

    if warmup and step < warmup:
        scale = (step + 1) / float(warmup)
    else:
        p = (step - warmup) / max(total_steps - warmup, 1)
        p = min(max(p, 0.0), 1.0)
        scale = min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + np.cos(np.pi * p))

    out = []
    for g, b in zip(optimizer.param_groups, base_lrs):
        g["lr"] = float(b) * float(scale)
        out.append(g["lr"])
    return out


def human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else str(n)
        n /= 1000.0
    return f"{n:.1f}T"
