"""
Tesla CV-NF Unit Tests: Self-Supervised Loss & Raymarching (PyTest)
==================================================================
Tests Differentiable Raymarching, SSIM metric, Inverse Warping, and
Autograd backpropagation for end-to-end self-supervised training.

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

from cv_nf.engine.self_supervised_loss import SSIMLoss, SelfSupervisedPhotometricEngine


def test_ssim_loss_computation():
    """Validates SSIM loss returns 0.0 for identical images and is bounded in [0, 1]."""
    ssim = SSIMLoss(window_size=7, channel=3)
    img = torch.rand(2, 3, 32, 32)
    loss_identical = ssim(img, img)
    assert loss_identical.item() < 1e-4

    img_diff = torch.rand(2, 3, 32, 32)
    loss_diff = ssim(img, img_diff)
    assert 0.0 <= loss_diff.item() <= 1.0


def test_differentiable_raymarching_and_autograd():
    """Validates raymarching shapes and backpropagation gradient flow."""
    engine = SelfSupervisedPhotometricEngine()

    B, H, W, N_samples = 1, 8, 8, 16
    sigmas = torch.rand(B, H, W, N_samples, requires_grad=True)
    colors = torch.rand(B, H, W, N_samples, 3, requires_grad=True)
    z_vals = torch.linspace(0.5, 50.0, N_samples).expand(B, H, W, N_samples)

    rgb, depth = engine.differentiable_raymarch(sigmas, colors, z_vals)

    assert rgb.shape == (B, H, W, 3)
    assert depth.shape == (B, H, W)

    # Test backpropagation
    dummy_loss = rgb.sum() + depth.sum()
    dummy_loss.backward()

    assert sigmas.grad is not None
    assert colors.grad is not None
    assert not torch.isnan(sigmas.grad).any()
