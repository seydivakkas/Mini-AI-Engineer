"""
Tesla CV-NF Unit Tests: ASTES & Continuous Neural Field (PyTest)
==============================================================
Tests continuous 4D coordinate harmonic encodings, implicit field outputs,
and explainable saliency bounds.

Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
All Rights Reserved.
"""

import pytest
import torch
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from cv_nf.models.continuous_field import ChronoVoxelNeuralField, MultiResolutionFourierEncoding
from cv_nf.models.cross_attention_fusion import SparseLinearCrossAttention
from cv_nf.models.uncertainty_head import DifferentiableSaliencyExplainer, HeteroscedasticAleatoricLoss


def test_continuous_harmonic_encoding():
    """Validates multi-resolution Fourier encoding dimensions."""
    encoder = MultiResolutionFourierEncoding(in_dim=4, num_frequencies=6)
    coords = torch.randn(2, 100, 4)
    encoded = encoder(coords)

    # 4 coords * 2 (sin/cos) * 6 freqs = 48 dims
    assert encoded.shape == (2, 100, 48)


def test_chrono_voxel_neural_field_forward():
    """Validates density, velocity flow, color, and uncertainty head shapes and bounds."""
    model = ChronoVoxelNeuralField(hidden_dim=64, num_frequencies=4, rgb_feat_dim=16, event_feat_dim=8)
    model.eval()

    B, N_pts = 2, 50
    xyz = torch.randn(B, N_pts, 3)
    t = torch.full((B, N_pts, 1), 0.016)
    event_feat = torch.randn(B, N_pts, 8)
    rgb_feat = torch.randn(B, N_pts, 16)

    with torch.no_grad():
        out = model(xyz, t, event_feat, rgb_feat)

    assert out["density"].shape == (B, N_pts, 1)
    assert (out["density"] >= 0.0).all() # Volume density is non-negative
    assert out["velocity"].shape == (B, N_pts, 3)
    assert out["color"].shape == (B, N_pts, 3)
    assert (out["color"] >= 0.0).all() and (out["color"] <= 1.0).all()
    assert out["uncertainty"].shape == (B, N_pts, 1)
    assert (out["uncertainty"] > 0.0).all()


def test_sparse_cross_attention_and_saliency():
    """Validates sparse cross-attention fusion and XAI saliency map normalization."""
    attn = SparseLinearCrossAttention(query_dim=16, key_dim=16, embed_dim=16, num_heads=2)
    q = torch.randn(2, 30, 16)
    kv = torch.randn(2, 80, 16)
    fused = attn(q, kv)
    assert fused.shape == (2, 30, 16)

    density = torch.rand(2, 30, 1) * 5.0
    vel = torch.randn(2, 30, 3)
    unc = torch.rand(2, 30, 1) * 0.5
    saliency = DifferentiableSaliencyExplainer.compute_saliency_map(density, vel, unc)

    assert saliency.shape == (2, 30, 1)
    assert saliency.min() >= 0.0 and saliency.max() <= 1.0 + 1e-5
