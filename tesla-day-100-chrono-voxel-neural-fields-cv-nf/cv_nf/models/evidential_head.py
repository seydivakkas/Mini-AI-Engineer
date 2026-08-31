"""
Tesla Chrono-Voxel Neural Fields (CV-NF) — Normal-Inverse-Gamma (NIG) Evidential Deep Learning
=============================================================================================
Decomposes Continuous-Time 4D Occupancy predictions into:
  1. Mean Target Prediction (gamma)
  2. Aleatoric Uncertainty (Sensor/Atmospheric Noise): sigma^2 = beta / (alpha - 1)
  3. Epistemic Uncertainty (OOD / AI Model Ignorance): Var[mu] = beta / (v * (alpha - 1))

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class NormalInverseGammaEvidentialHead(nn.Module):
    """
    Evidential Deep Learning (EDL) Regression Head parameterized by NIG distribution.
    Outputs 4 parameters per voxel/query: (gamma, v, alpha, beta).
    """
    def __init__(self, in_features: int = 64):
        super().__init__()
        self.fc_dense = nn.Linear(in_features, 64)
        self.fc_evidence = nn.Linear(64, 4)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing evidential hyper-parameters:
          gamma: Predicted Mean in [-inf, +inf]
          v: Virtual evidence count > 0 (Softplus + 1e-4)
          alpha: Shape parameter > 1.0 (Softplus + 1.0 + 1e-4)
          beta: Scale parameter > 0 (Softplus + 1e-4)
        """
        h = F.relu(self.fc_dense(x))
        out = self.fc_evidence(h)

        gamma = out[..., 0:1]
        v = F.softplus(out[..., 1:2]) + 1e-4
        alpha = F.softplus(out[..., 2:3]) + 1.0 + 1e-4
        beta = F.softplus(out[..., 3:4]) + 1e-4

        # Mathematical Uncertainty Decompositions
        # 1. Aleatoric Variance (Noise inherent in camera/environment):
        aleatoric = beta / (alpha - 1.0)

        # 2. Epistemic Variance (Model lack of knowledge / Out-Of-Distribution):
        epistemic = beta / (v * (alpha - 1.0))

        # 3. Total Predictive Uncertainty:
        total_uncertainty = aleatoric + epistemic

        return {
            "gamma": gamma,
            "v": v,
            "alpha": alpha,
            "beta": beta,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": total_uncertainty
        }

    @staticmethod
    def evidential_nig_loss(
        y_true: torch.Tensor,
        gamma: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        coeff: float = 0.05
    ) -> torch.Tensor:
        """
        Computes the Maximum Likelihood Estimation of Student-t marginal distribution
        plus Evidence Regularization penalty:
          L_NIG = L_NLL + coeff * L_reg
        """
        two_beta_v = 2.0 * beta * (1.0 + v)
        diff_sq = (y_true - gamma) ** 2
        
        # Negative Log-Likelihood of Student-t
        nll = 0.5 * torch.log(torch.tensor(3.14159265)) \
            - 0.5 * torch.log(v) \
            + alpha * torch.log(two_beta_v) \
            - torch.lgamma(alpha) \
            + torch.lgamma(alpha + 0.5) \
            + (alpha + 0.5) * torch.log(1.0 + (v * diff_sq) / two_beta_v)

        # Regularizer penalizing confident incorrect predictions
        error = torch.abs(y_true - gamma)
        reg = error * (2.0 * v + alpha)

        return torch.mean(nll + coeff * reg)
