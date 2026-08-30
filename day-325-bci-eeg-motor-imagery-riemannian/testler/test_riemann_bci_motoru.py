"""
Day 325: Brain-Computer Interface (BCI) & Riemannian Geometry on EEG
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.riemann_bci_motoru import (
    EEGMotorImageryGenerator,
    CovarianceEstimator,
    RiemannianGeometryEngine,
    RiemannianMDMClassifier,
    TangentSpaceClassifier,
)


def test_eeg_motor_imagery_generator_shape():
    """
    EEG Sinyal üretecinin boyut doğrulaması.
    """
    gen = EEGMotorImageryGenerator(num_channels=4, sampling_rate=250, trial_duration_sec=1.0)
    x_eeg, y = gen.uret_eeg_deneyleri(num_trials_per_class=5)
    
    assert x_eeg.shape == (15, 4, 250)  # 3 sınıftan 5'er trial = 15
    assert len(y) == 15


def test_covariance_estimator_spd():
    """
    SCM Kovaryans matrisinin Simetrik Pozitif Tanımlı (SPD) olduğunu doğrulama.
    """
    x_sample = np.random.randn(6, 200)
    scm = CovarianceEstimator.hesapla_scm(x_sample)
    
    # 1. Simetri Testi: S = S^T
    assert np.allclose(scm, scm.T, atol=1e-6)
    
    # 2. Pozitif Tanımlılık Testi: Özdeğerler > 0
    eigvals = np.linalg.eigvalsh(scm)
    assert np.all(eigvals > 0.0)


def test_riemannian_distance_properties():
    """
    Affine-Invariant Riemannian Metric (AIRM) Metrik Özellikleri Testi.
    """
    np.random.seed(42)
    s1 = CovarianceEstimator.hesapla_scm(np.random.randn(4, 100))
    s2 = CovarianceEstimator.hesapla_scm(np.random.randn(4, 100))
    
    d12 = RiemannianGeometryEngine.riemannian_distance(s1, s2)
    d21 = RiemannianGeometryEngine.riemannian_distance(s2, s1)
    d11 = RiemannianGeometryEngine.riemannian_distance(s1, s1)
    
    # Simetri
    assert pytest.approx(d12, abs=1e-5) == d21
    # Özdeşlerin Mesafesi Sıfırdır
    assert pytest.approx(d11, abs=1e-5) == 0.0
    # Pozitiflik
    assert d12 > 0.0


def test_tangent_space_projection_dimension():
    """
    Teğet Uzayı Vektörleştirmesinin Vektör Boyutunu Doğrulama: C * (C + 1) / 2
    """
    channels = 6
    s = CovarianceEstimator.hesapla_scm(np.random.randn(channels, 100))
    mean_s = np.eye(channels)
    
    vec = RiemannianGeometryEngine.tangent_space_projection(s, mean_s)
    expected_dim = int(channels * (channels + 1) / 2)  # 6*7/2 = 21
    assert vec.shape == (expected_dim,)


def test_frechet_mean_convergence():
    """
    Frechet Ortalamasının Kurala Uygun Şekilde Hesaplanması ve SPD Kalması.
    """
    sigmas = [CovarianceEstimator.hesapla_scm(np.random.randn(4, 100)) for _ in range(5)]
    f_mean = RiemannianGeometryEngine.frechet_mean(sigmas, max_iter=10)
    
    assert f_mean.shape == (4, 4)
    eigvals = np.linalg.eigvalsh(f_mean)
    assert np.all(eigvals > 0.0)
