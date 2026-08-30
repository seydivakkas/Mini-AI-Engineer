"""
Day 361: Optical Matrix Multiplication with Mach-Zehnder Interferometer (MZI) Photonic Mesh
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

from src.mzi_photonic_mesh_motoru import (
    MZICell,
    ClementsMZIMesh,
    PhotonicMatrixMultiplier,
    PhotonicInferenceSimulator,
)
from src.mzi_profilleyici import MZIProfilleyici


def test_mzi_cell_unitary_property():
    r"""
    2x2 MZI Hücresi Üniterlik (T^\dagger T = I) Testi.
    """
    cell = MZICell(theta=0.785, phi=1.25)
    t_mat = cell.transfer_matrix()
    identity_approx = t_mat.conj().T @ t_mat
    np.testing.assert_allclose(identity_approx, np.eye(2), atol=1e-5)


def test_clements_mzi_mesh_unitary():
    """
    4x4 Clements Fotonik Ağ Üniterlik Testi.
    """
    mesh = ClementsMZIMesh(dim=4)
    u_mat = mesh.compute_mesh_unitary()
    assert u_mat.shape == (4, 4)
    identity_approx = u_mat.conj().T @ u_mat
    np.testing.assert_allclose(identity_approx, np.eye(4), atol=1e-5)


def test_photonic_matrix_multiplier_gemm():
    """
    SVD Tabanlı Optik GEMM Matris Çarpım Testi.
    """
    mult = PhotonicMatrixMultiplier(dim=4)
    w_mat = np.array([
        [1.0, 0.5, -0.2, 0.1],
        [0.0, 1.2, 0.3, -0.5],
        [0.4, -0.1, 0.9, 0.2],
        [-0.3, 0.2, 0.1, 0.8]
    ])
    mult.load_weight_matrix(w_mat)
    
    x = np.array([1.0, 0.5, -0.5, 2.0])
    y_opt = mult.optical_gemm(x)
    assert y_opt.shape == (4,)
    assert not np.isnan(y_opt).any()


def test_mzi_profiler_metrics():
    """
    MZI Fotonik Profilleyici Metrik Testi.
    """
    mock_res = {
        "fidelity_score": 98.5,
        "energy_savings_ratio": 480.0,
        "photonic_latency_ps": 11.66
    }
    metrics = MZIProfilleyici.profille(mock_res)
    assert metrics["fidelity_score"] == 98.5
    assert metrics["energy_savings_ratio"] == 480.0
    assert metrics["photonic_readiness"] > 98.0
