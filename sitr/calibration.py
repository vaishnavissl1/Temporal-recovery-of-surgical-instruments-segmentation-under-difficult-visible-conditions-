"""
Temperature scaling for the segmentation posteriors.

Why this matters HERE more than in ordinary segmentation: the reliability
estimator's strongest indicators (q, H, M) are all read off the posterior
distribution. Neural networks are systematically overconfident, so an
uncalibrated model reports q ~ 0.95 on frames it is about to get wrong, and the
estimator has to learn around that distortion instead of using the signal.

Temperature scaling (Guo et al., 2017) fits a single scalar T on held-out data
by minimising NLL of softmax(logits / T). It cannot change the argmax, so
segmentation accuracy is EXACTLY unchanged — only the confidence values move.
That is the ideal property here: Stage 1 metrics are untouched, Stage 2 gets a
better-conditioned input.

T > 1 means the model was overconfident (the usual case). T ~ 1 means it was
already calibrated and nothing happens.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .labels import IGNORE_INDEX


@torch.no_grad()
def _collect_logits(model, loader, device, use_amp: bool, max_pixels: int,
                    channels_last: bool = False):
    """Subsample pixels across the validation set; full maps do not fit."""
    logit_chunks, target_chunks = [], []
    total = 0
    g = torch.Generator().manual_seed(0)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits, _ = model(images)
        logits = logits.float()

        B, C = logits.shape[0], logits.shape[1]
        flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        tgt = masks.reshape(-1)
        keep = tgt != IGNORE_INDEX
        flat, tgt = flat[keep], tgt[keep]
        if flat.numel() == 0:
            continue

        # cap per batch so one sequence cannot dominate the fit
        per_batch = max(1, max_pixels // max(len(loader), 1))
        if flat.shape[0] > per_batch:
            idx = torch.randperm(flat.shape[0], generator=g).to(flat.device)[:per_batch]
            flat, tgt = flat[idx], tgt[idx]

        logit_chunks.append(flat.cpu())
        target_chunks.append(tgt.cpu())
        total += flat.shape[0]
        if total >= max_pixels:
            break

    if not logit_chunks:
        return None, None
    return torch.cat(logit_chunks), torch.cat(target_chunks)


def fit_temperature(model, loader, device, use_amp: bool = False,
                    max_pixels: int = 2_000_000, max_iter: int = 100,
                    channels_last: bool = False) -> Optional[float]:
    """Fit a single temperature on the validation loader. Returns T, or None.

    Uses LBFGS on NLL, which is the standard recipe and converges in a few
    dozen evaluations on a couple of million sampled pixels.
    """
    was_training = model.training
    model.eval()
    logits, targets = _collect_logits(model, loader, device, use_amp,
                                      max_pixels, channels_last)
    if was_training:
        model.train()
    if logits is None or logits.shape[0] < 1000:
        return None

    log_t = torch.zeros(1, requires_grad=True)          # T = exp(log_t) > 0
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(log_t), targets)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:  # noqa: BLE001
        return None

    t = float(torch.exp(log_t).item())
    if not (0.05 < t < 20.0):        # a wild value means the fit diverged
        return None
    return t


def nll_and_ece(logits: torch.Tensor, targets: torch.Tensor,
                temperature: float = 1.0, bins: int = 15):
    """Negative log-likelihood and expected calibration error, for reporting."""
    scaled = logits / float(temperature)
    nll = float(F.cross_entropy(scaled, targets))
    probs = torch.softmax(scaled, dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == targets).float()

    ece, n = 0.0, conf.numel()
    edges = torch.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += float(m.sum()) / n * abs(float(correct[m].mean() - conf[m].mean()))
    return nll, ece


@torch.no_grad()
def calibration_report(model, loader, device, temperature: float,
                       use_amp: bool = False, max_pixels: int = 1_000_000,
                       channels_last: bool = False):
    """NLL/ECE before and after scaling, so the effect is reported not assumed."""
    logits, targets = _collect_logits(model, loader, device, use_amp,
                                      max_pixels, channels_last)
    if logits is None:
        return {}
    nll0, ece0 = nll_and_ece(logits, targets, 1.0)
    nll1, ece1 = nll_and_ece(logits, targets, temperature)
    return {"temperature": float(temperature),
            "nll_before": nll0, "nll_after": nll1,
            "ece_before": ece0, "ece_after": ece1,
            "pixels": int(logits.shape[0])}
