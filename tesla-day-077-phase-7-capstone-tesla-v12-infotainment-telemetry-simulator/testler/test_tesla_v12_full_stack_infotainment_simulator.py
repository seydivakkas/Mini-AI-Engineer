"""
Tesla Faz 7 Capstone Birim Testleri (PyTest)
============================================
Bu test paketi; tam yığın Tesla V12 Infotainment ve Telemetri simülatörünün
9 alt sisteminin senkronizasyonunu, 3D projeksiyonunu ve genel sağlığını test eder.

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

from src.tesla_v12_full_stack_infotainment_simulator import TeslaV12FullStackInfotainmentSimulator


def test_tam_yigin_infotainment_dongusu():
    """Tüm alt sistemlerin senkronize bir döngüde başarıyla çalıştığı test edilir."""
    sim = TeslaV12FullStackInfotainmentSimulator()
    res = sim.step_infotainment_cycle(
        speed_kmh=90.0,
        battery_pct=75.0,
        obstacle_3d=(1.5, 20.0, 0.0),
        phone_uwb_tof_ns=4.5
    )

    assert res["speed_kmh"] == 90.0
    assert res["battery_pct"] == 75.0
    assert res["fsd_engaged"] is True
    assert res["capstone_all_systems_go"] is True


def test_3d_gpu_projeksiyonu():
    """3D dünya koordinatlarının geçerli ekran piksel alanına (1920x1080) düştüğü test edilir."""
    sim = TeslaV12FullStackInfotainmentSimulator()
    res = sim.step_infotainment_cycle(obstacle_3d=(0.0, 20.0, 0.0))

    # Merkez engel tam ekranın ortasında (u ~ 960, v ~ 540) olmalıdır
    assert np.isclose(res["screen_proj_u"], 960.0, atol=1.0)
    assert np.isclose(res["screen_proj_v"], 540.0, atol=1.0)


def test_uwb_kilit_entegrasyonu():
    """UWB mesafesi 2m'yi aşınca kapının kilitlendiği test edilir."""
    sim = TeslaV12FullStackInfotainmentSimulator()
    res_far = sim.step_infotainment_cycle(phone_uwb_tof_ns=30.0)  # 9 metre

    assert res_far["door_locked"] is True
    assert res_far["capstone_all_systems_go"] is False  # Telefon uzakta olduğu için
