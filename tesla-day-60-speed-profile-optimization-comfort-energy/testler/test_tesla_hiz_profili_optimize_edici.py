"""
Tesla Hız Profili Optimizasyonu Birim Testleri (PyTest)
========================================================
Bu test paketi; Maksimum viraj hızı tespitini, İleri-Geri geçişli
hız sınırlamasını ve yanal ivme konfor kısıtlarını test eder.

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

from src.tesla_hiz_profili_optimize_edici import TeslaSpeedProfileOptimizer


def test_maksimum_viraj_hizi_hesabi():
    """Düz yolda maksimum hız, virajda ise eğriliğe bağlı güvenli hız üretildiği test edilir."""
    optimizer = TeslaSpeedProfileOptimizer(max_speed_mps=33.33, max_lat_accel_mps2=2.0)

    # Düz Yol (kappa = 0)
    v_straight = optimizer.max_safe_cornering_speed(0.0)
    assert np.isclose(v_straight, 33.33)

    # Keskin Viraj (kappa = 0.04 -> R = 25m) -> v = sqrt(2.0 / 0.04) = sqrt(50) ~= 7.07 m/s (25.4 km/h)
    v_corner = optimizer.max_safe_cornering_speed(0.04)
    assert np.isclose(v_corner, np.sqrt(50.0))


def test_ileri_geri_hiz_optimizasyonu():
    """Viraja yaklaşırken aracın önceden fren yaptığı ve yanal ivmenin 2.0 m/s^2 altında kaldığı test edilir."""
    optimizer = TeslaSpeedProfileOptimizer()
    res = optimizer.optimize_speed_profile(track_length_m=200.0, curve_start_m=70.0, curve_end_m=130.0, curve_kappa=0.04)

    assert res["is_comfortable"] is True
    assert res["min_corner_speed_mps"] < res["max_straight_speed_mps"]
    assert res["regen_energy_kj"] > 0.0


def test_dizi_ve_profil_boyut_tutarliligi():
    """Hız, boyuna ivme ve yanal ivme dizilerinin 100 nokta boyutunda olduğu test edilir."""
    optimizer = TeslaSpeedProfileOptimizer()
    res = optimizer.optimize_speed_profile(num_points=100)

    assert len(res["s_array"]) == 100
    assert len(res["optimized_speed_mps"]) == 100
    assert len(res["lateral_acc_mps2"]) == 100
