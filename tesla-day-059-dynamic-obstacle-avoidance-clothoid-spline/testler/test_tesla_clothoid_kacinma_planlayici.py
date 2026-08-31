"""
Tesla Clothoid ve Dinamik Kaçınma Birim Testleri (PyTest)
==========================================================
Bu test paketi; Eğrilik değişim hızı güvenliğini, Clothoid Fresnel
dilim üretimini ve dinamik engelden kaçınma güvenlik mesafesini test eder.

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

from src.tesla_clothoid_kacinma_planlayici import TeslaClothoidAvoidancePlanner


def test_egrilik_degisim_hizi_guvenligi():
    """Direksiyon dönüş hız sınırını aşmayan eğrilik değişiminin güvenli onaylandığı test edilir."""
    planner = TeslaClothoidAvoidancePlanner(wheelbase_m=2.875, max_steer_rate_rad_s=0.60)

    # Güvenli küçük eğrilik değişimi: 0.02'den 0.03'e 5 metrede
    is_safe = planner.is_curvature_rate_safe(kappa_curr=0.02, kappa_next=0.03, ds=5.0, speed_mps=20.0)
    assert is_safe is True

    # Tehlikeli ani eğrilik sıçraması: 0.0'dan 0.15'e 0.5 metrede
    is_unsafe = planner.is_curvature_rate_safe(kappa_curr=0.0, kappa_next=0.15, ds=0.5, speed_mps=20.0)
    assert is_unsafe is False


def test_clothoid_dilim_uretimi():
    """Clothoid diliminin doğrusal eğrilik ve sürekli pozisyon ürettiği test edilir."""
    planner = TeslaClothoidAvoidancePlanner()
    seg = planner.generate_clothoid_segment(s_total_m=20.0, kappa_start=0.0, kappa_end=0.05, num_points=25)

    assert len(seg["x"]) == 25
    assert len(seg["curvature_kappa"]) == 25
    assert np.isclose(seg["curvature_kappa"][0], 0.0)
    assert np.isclose(seg["curvature_kappa"][-1], 0.05)


def test_engelden_kacinma_manevra_guvenligi():
    """Manevranın engele en az 1.5 metre güvenlik mesafesi bıraktığı test edilir."""
    planner = TeslaClothoidAvoidancePlanner()
    res = planner.plan_obstacle_avoidance_maneuver(obstacle_x_m=35.0, obstacle_y_m=0.0)

    assert res["is_safe"] is True
    assert res["min_clearance_m"] >= 1.5
    assert len(res["x_traj"]) == 100
