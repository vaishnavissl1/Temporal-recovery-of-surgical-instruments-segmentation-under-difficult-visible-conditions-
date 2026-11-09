"""Wrap a fitted scikit-learn regressor so it drops into every place a
ReliabilityMLP is used (runtime.ReleaseBundle, SelectiveInference): same call
signature (tensor [B, D] -> tensor [B] in [0, 1]) and the same `in_dim`."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SklearnReliability(nn.Module):
    def __init__(self, estimator, in_dim: int):
        super().__init__()
        self.estimator = estimator
        self.in_dim = int(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().float().cpu().numpy().reshape(-1, self.in_dim)
        y = np.clip(self.estimator.predict(x_np.astype(np.float64)), 0.0, 1.0)
        return torch.as_tensor(y, dtype=torch.float32, device=x.device)

    @torch.no_grad()
    def predict(self, feats):
        return float(self.forward(torch.as_tensor(feats).view(1, -1)).item())
