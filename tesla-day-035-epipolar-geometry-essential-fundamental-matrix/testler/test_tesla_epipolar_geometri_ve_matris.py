"""
Tesla Epipolar Geometri Birim Testleri (PyTest)
==============================================
Bu test paketi; Essential Matrisi (E), Fundamental Matrisi (F),
8-Nokta SVD algoritmasını ve Sampson hata kısıtını test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_epipolar_geometri_ve_matris import TeslaEpipolarCalibrator


def test_skew_symmetric_matris_ozellikleri():
    """Vektörün kendisiyle çarpraz çarpımının sıfır olduğu ([t]x @ t = 0) test edilir."""
    t = np.array([1.5, -2.0, 3.0])
    tx = TeslaEpipolarCalibrator.skew_symmetric(t)

    # Antisimetri: tx.T == -tx
    assert np.allclose(tx.T, -tx)
    # Kendisiyle çarpımı sıfır
    assert np.allclose(tx @ t, np.zeros(3))


def test_epipolar_kisit_denklemi():
    """Doğru eşleşen noktalar için x2^T @ F @ x1 kısıtının sıfıra yakın olduğu test edilir."""
    K = np.array([[1000.0, 0, 500.0], [0, 1000.0, 500.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.5, 0.0, 0.0])

    F = TeslaEpipolarCalibrator.compute_fundamental_matrix(K, K, R, t)

    # 3D bir nokta: P = (1.0, 0.5, 5.0)
    P = np.array([1.0, 0.5, 5.0])
    # Cam 1 izdüşümü: (u1, v1)
    u1 = K[0, 0] * (P[0]/P[2]) + K[0, 2]
    v1 = K[1, 1] * (P[1]/P[2]) + K[1, 2]
    x1 = np.array([u1, v1, 1.0])

    # Cam 2 izdüşümü: P2 = P - t = (0.5, 0.5, 5.0)
    P2 = P - t
    u2 = K[0, 0] * (P2[0]/P2[2]) + K[0, 2]
    v2 = K[1, 1] * (P2[1]/P2[2]) + K[1, 2]
    x2 = np.array([u2, v2, 1.0])

    # Epipolar Kısıt: x2.T @ F @ x1 = 0
    epipolar_val = float(x2.T @ F @ x1)
    assert abs(epipolar_val) < 1e-4


def test_8_nokta_svd_rank2_zorlamasi():
    """8-nokta algoritmasının ürettiği Fundamental matrisin determinantının 0 olduğu test edilir."""
    pts1 = np.random.uniform(100, 800, size=(10, 2))
    pts2 = pts1 + np.random.normal(0, 2, size=(10, 2))

    F_est = TeslaEpipolarCalibrator.estimate_fundamental_8point(pts1, pts2)

    assert F_est.shape == (3, 3)
    assert abs(np.linalg.det(F_est)) < 1e-6  # Rank-2 zorlaması det(F) = 0
    assert np.linalg.matrix_rank(F_est) == 2
