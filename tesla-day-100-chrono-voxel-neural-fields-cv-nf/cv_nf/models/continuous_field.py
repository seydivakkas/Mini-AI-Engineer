r"""
Tesla Chrono-Voxel Neural Fields (CV-NF) Continuous Neural Field Core
====================================================================
This module implements the Continuous-Time Implicit 4D Neural Radiance &
Occupancy Flow Field. Given an arbitrary spatial coordinate x in R^3 and
microsecond query timestamp t in R, it predicts density sigma, velocity flow v,
RGB color c, and aleatoric uncertainty u.

Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class MultiResolutionFourierEncoding(nn.Module):
    """
    Continuous Multi-Resolution Harmonic / Fourier Embedding for 4D Coordinates (x, y, z, t).
    """
    def __init__(self, in_dim: int = 4, num_frequencies: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [..., in_dim]
        returns: [..., in_dim * 2 * num_frequencies]
        """
        bands = self.freq_bands.to(coords.device)
        # Shape: [..., in_dim, num_frequencies]
        scaled = coords.unsqueeze(-1) * bands
        # Sin and Cos harmonics
        sin_part = torch.sin(scaled)
        cos_part = torch.cos(scaled)
        encoded = torch.cat([sin_part, cos_part], dim=-1)
        return encoded.flatten(start_dim=-2)


class ChronoVoxelNeuralField(nn.Module):
    r"""
    Continuous-Time Implicit 4D Occupancy & Motion Flow Field (CV-NF).
    F_theta: (\gamma(x), \gamma(t), S(x, t), I_RGB) -> (\sigma, v, c, u)
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_frequencies: int = 6,
        rgb_feat_dim: int = 32,
        event_feat_dim: int = 16
    ):
        super().__init__()
        self.encoder = MultiResolutionFourierEncoding(in_dim=4, num_frequencies=num_frequencies)
        encoded_dim = 4 * 2 * num_frequencies

        # Fusion MLP
        in_features = encoded_dim + rgb_feat_dim + event_feat_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Multi-Task Prediction Heads
        self.density_head = nn.Linear(hidden_dim, 1)          # Volume Density \sigma >= 0
        self.velocity_head = nn.Linear(hidden_dim, 3)         # 3D Motion Flow Vector v in m/s
        self.color_head = nn.Linear(hidden_dim, 3)            # RGB Neural Radiance [0, 1]
        self.uncertainty_head = nn.Linear(hidden_dim, 1)      # Heteroscedastic Aleatoric Uncertainty \log(\sigma^2)

    def forward(
        self,
        spatial_xyz: torch.Tensor,
        query_t: torch.Tensor,
        event_surface_feat: torch.Tensor,
        rgb_context_feat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        spatial_xyz: [B, N_pts, 3] (Continuous 3D coordinates)
        query_t: [B, N_pts, 1] (Microsecond continuous timestamps)
        event_surface_feat: [B, N_pts, event_feat_dim] (ASTES continuous kernel embedding)
        rgb_context_feat: [B, N_pts, rgb_feat_dim] (Sparse Cross-Attention context)
        """
        # 1. 4D Coordinate Harmonic Encoding
        coords_4d = torch.cat([spatial_xyz, query_t], dim=-1)
        encoded_4d = self.encoder(coords_4d)

        # 2. Multimodal Fusion
        fused = torch.cat([encoded_4d, rgb_context_feat, event_surface_feat], dim=-1)
        features = self.backbone(fused)

        # 3. Predict Density (Softplus for positive density), Flow, Color, Uncertainty
        raw_sigma = self.density_head(features)
        sigma = F.softplus(raw_sigma)

        velocity = self.velocity_head(features)
        color = torch.sigmoid(self.color_head(features))
        log_var = self.uncertainty_head(features)
        uncertainty = torch.exp(0.5 * log_var)  # Standard deviation estimate

        return {
            "density": sigma,
            "velocity": velocity,
            "color": color,
            "uncertainty": uncertainty,
            "log_variance": log_var
        }
