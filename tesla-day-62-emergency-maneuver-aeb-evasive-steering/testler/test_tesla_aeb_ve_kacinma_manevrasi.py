"""
Tesla AEB ve Acil Kaçınma Birim Testleri (PyTest)
=================================================
Bu test paketi; Acil durma mesafesi formülasyonunu, Euro-NCAP AEB kademelerini
ve Acil Kaçınma Direksiyonu (AES) karar mantığını test eder.

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

from src.tesla_aeb_ve_kacinma_manevrasi import TeslaAEBController, AEBLevel


def test_acil_durma_mesafesi_hesabi():
    """v = 20 m/s (72 km/h) için durma mesafesinin d = 20*0.2 + 400/(2*9) = 4 + 22.22 = 26.22m olduğu test edilir."""
    controller = TeslaAEBController(max_aeb_decel_mps2=9.0, system_delay_s=0.20)
    d_stop = controller.compute_emergency_stopping_distance(speed_mps=20.0)

    beklenen_d = (20.0 * 0.20) + ((20.0 ** 2) / 18.0)
    assert np.isclose(d_stop, beklenen_d)


def test_tam_aeb_tetikleme():
    """TTC <= 1.0s olduğunda FULL_AEB seviyesi ve -9.0 m/s^2 fren ivmesi üretildiği test edilir."""
    controller = TeslaAEBController()
    res = controller.evaluate_aeb_trigger(
        ego_speed_mps=20.0,
        dist_to_obstacle_m=18.0,
        rel_speed_mps=20.0,
        is_adjacent_lane_clear=False
    )

    assert res["aeb_level"] == AEBLevel.FULL_AEB.value
    assert res["target_acc_mps2"] == -9.0
    assert res["is_emergency"] is True


def test_acil_kacinma_direksiyonu_tetikleme():
    """Durma mesafesi yetersizken yan şerit boşsa EVASIVE_STEER devreye girdiği test edilir."""
    controller = TeslaAEBController()
    # Mesafe durma mesafesinin %75'inden az (örn. 15m < 26.22*0.75=19.6m) ve yan şerit açık
    res = controller.evaluate_aeb_trigger(
        ego_speed_mps=20.0,
        dist_to_obstacle_m=15.0,
        rel_speed_mps=20.0,
        is_adjacent_lane_clear=True
    )

    assert res["aeb_level"] == AEBLevel.EVASIVE_STEER.value
    assert res["target_steer_rad"] > 0.0
    assert res["is_emergency"] is True
