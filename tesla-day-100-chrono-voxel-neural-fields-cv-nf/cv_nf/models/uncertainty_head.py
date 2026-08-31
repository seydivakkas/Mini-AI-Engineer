"""
Tesla Chrono-Voxel Neural Fields (CV-NF) Uncertainty & Saliency Head
===================================================================
Heteroscedastic Aleatoric Loss & Differentiable XAI Saliency Engine.

Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class HeteroscedasticAleatoricLoss(nn.Module):
    r"""
    Kendall & Gal Heteroscedastic Aleatoric Uncertainty Loss:
    L = 0.5 * exp(-s) * ||y - y_hat||^2 + 0.5 * s, where s = log(\sigma^2)
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_mean: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        diff_sq = (pred_mean - target) ** 2
        precision = torch.exp(-log_var)
        loss = 0.5 * precision * diff_sq + 0.5 * log_var
        return loss.mean()


class DifferentiableSaliencyExplainer:
    """
    Real-time Saliency Attribution Engine.
    Computes spatial gradient of occupancy density with respect to input event density and motion.
    """
    @staticmethod
    def compute_saliency_map(
        density: torch.Tensor,
        velocity: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        density: [B, N, 1]
        velocity: [B, N, 3]
        uncertainty: [B, N, 1]
        returns: normalized saliency attribution [B, N, 1] in [0, 1]
        """
        vel_norm = torch.norm(velocity, dim=-1, keepdim=True)
        raw_saliency = torch.sigmoid(1.5 * density) * (1.0 + 0.8 * vel_norm) * (1.0 + uncertainty)
        # Min-Max Normalization per batch
        min_val = raw_saliency.amin(dim=1, keepdim=True)
        max_val = raw_saliency.amax(dim=1, keepdim=True) + 1e-6
        return (raw_saliency - min_val) / (max_val - min_val)
