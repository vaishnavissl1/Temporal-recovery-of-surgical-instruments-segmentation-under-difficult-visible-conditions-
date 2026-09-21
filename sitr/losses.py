"""Stage 1 training objectives — report eqs. (27)-(29)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import IGNORE_INDEX


class SoftDiceLoss(nn.Module):
    """Eq. (28): soft Dice averaged over classes, ignoring `ignore_index` pixels.

    Computed on probabilities, so it stays differentiable; ignored pixels are
    zeroed in both the prediction and the one-hot target so they contribute to
    neither the intersection nor the union.
    """

    def __init__(self, eps: float = 1.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        valid = (target != self.ignore_index)
        safe = torch.where(valid, target, torch.zeros_like(target))
        onehot = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).to(probs.dtype)

        vmask = valid.unsqueeze(1).to(probs.dtype)
        probs = probs * vmask
        onehot = onehot * vmask

        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss — for thin structures that Dice under-weights.

    Dice weights false positives and false negatives equally. On a 2-4 px wide
    clasper tip at 256x320, missing it entirely costs almost nothing in Dice
    because the class is tiny, so the model learns to drop it. Tversky's beta
    (weight on false negatives) > alpha penalises exactly that, and the focal
    exponent concentrates gradient on the classes still being got wrong.

    alpha=0.3, beta=0.7, gamma=0.75 are the standard settings from the medical
    segmentation literature and are what the defaults use.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, eps: float = 1.0,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        valid = (target != self.ignore_index)
        safe = torch.where(valid, target, torch.zeros_like(target))
        onehot = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).to(probs.dtype)
        vmask = valid.unsqueeze(1).to(probs.dtype)
        probs, onehot = probs * vmask, onehot * vmask

        dims = (0, 2, 3)
        tp = (probs * onehot).sum(dims)
        fp = (probs * (1 - onehot)).sum(dims)
        fn = ((1 - probs) * onehot).sum(dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return torch.pow(1.0 - tversky, self.gamma).mean()


class SegLoss(nn.Module):
    """Eq. (29) plus an optional Focal-Tversky term.

    L_seg = lambda_CE * L_CE + lambda_Dice * L_Dice + lambda_Tversky * L_FT

    Setting lambda_tversky=0 gives exactly the report's eq. (29).
    """

    def __init__(self, lambda_ce: float = 1.0, lambda_dice: float = 1.0,
                 lambda_tversky: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.lambda_ce = float(lambda_ce)
        self.lambda_dice = float(lambda_dice)
        self.lambda_tversky = float(lambda_tversky)
        self.ce = nn.CrossEntropyLoss(weight=class_weights,
                                      ignore_index=ignore_index)
        self.dice = SoftDiceLoss(ignore_index=ignore_index)
        self.tversky = (FocalTverskyLoss(ignore_index=ignore_index)
                        if self.lambda_tversky > 0 else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = self.ce(logits, target)
        dc = self.dice(logits, target)
        total = self.lambda_ce * ce + self.lambda_dice * dc
        parts = {"ce": float(ce.detach()), "dice": float(dc.detach()),
                 "tversky": 0.0}
        if self.tversky is not None:
            tv = self.tversky(logits, target)
            total = total + self.lambda_tversky * tv
            parts["tversky"] = float(tv.detach())
        return total, parts


class ReliabilityLoss(nn.Module):
    """Eq. (30): mean squared error between r_t and r*_t = IoU(S_t, Y_t)."""

    def forward(self, r: torch.Tensor, r_star: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(r, r_star)
