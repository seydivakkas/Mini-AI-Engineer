"""
Tesla Derinlik ve Optik Akış Birim Testleri (PyTest)
===================================================
Bu test paketi; Disparity derinlik hesabını ($Z=fB/d$), karesel belirsizliği,
Time-to-Contact (TTC) formülasyonunu ve Lucas-Kanade optik akışını test eder.

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

from src.tesla_derinlik_ve_optik_akis import TeslaDepthAndOpticalFlowEstimator


def test_disparity_to_depth_donusumu():
    """f = 1200 px, B = 0.5 m, d = 24 px iken Z = 25.0 m çıktığı test edilir."""
    estimator = TeslaDepthAndOpticalFlowEstimator(focal_length_px=1200.0, baseline_m=0.50)
    disp = np.array([24.0])
    depth = estimator.disparity_to_depth(disp)
    assert np.isclose(depth[0], 25.0)


def test_karesel_derinlik_belirsizligi():
    """Uzak mesafelerdeki derinlik belirsizliğinin yakın mesafeden karesel olarak büyük olduğu test edilir."""
    estimator = TeslaDepthAndOpticalFlowEstimator(focal_length_px=1200.0, baseline_m=0.50)
    d_near = np.array([10.0])
    d_far = np.array([50.0])

    unc_near = estimator.compute_depth_uncertainty(d_near)
    unc_far = estimator.compute_depth_uncertainty(d_far)

    # 50m'deki hata 10m'deki hatanın 25 katı olmalıdır (50^2 / 10^2 = 25)
    assert np.isclose(unc_far[0] / unc_near[0], 25.0)


def test_time_to_contact_hesabi():
    """30 metre mesafede 10 m/s yaklaşan araç için TTC = 3.0 saniye olduğu test edilir."""
    estimator = TeslaDepthAndOpticalFlowEstimator()
    ttc = estimator.compute_time_to_contact(depth_m=30.0, rel_speed_mps=10.0)
    assert np.isclose(ttc, 3.0)


def test_lucas_kanade_hareket_kestirimi():
    """Yapay ötelenen bir görüntü yamasında hareket yönünün doğru kestirildiği test edilir."""
    estimator = TeslaDepthAndOpticalFlowEstimator()
    # 2D Gauss Yaması
    y, x = np.mgrid[-10:11, -10:11]
    patch1 = np.exp(-(x**2 + y**2) / 20.0) * 255.0
    # 1 piksel sağa, 1 piksel aşağı öteleme
    patch2 = np.roll(patch1, shift=(1, 1), axis=(0, 1))

    vx, vy = estimator.estimate_lucas_kanade_flow(patch1, patch2)
    assert vx > 0.0
    assert vy > 0.0
