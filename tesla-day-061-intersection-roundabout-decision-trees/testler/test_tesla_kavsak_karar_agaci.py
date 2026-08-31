"""
Tesla Kavşak Karar Ağacı Birim Testleri (PyTest)
=================================================
Bu test paketi; Time-To-Collision (TTC) hesaplamasını, Gap Acceptance
karar mantığını ve döner kavşak durum geçişlerini test eder.

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

from src.tesla_kavsak_karar_agaci import TeslaIntersectionDecisionTree, RoundaboutState


def test_can_enter_intersection_ttc_esigi():
    """TTC >= 3.5s için True, TTC < 3.5s için False döndüğü test edilir."""
    tree = TeslaIntersectionDecisionTree(min_ttc_safe_s=3.5)

    # 40 metrede 10 m/s hızla yaklaşan araç -> TTC = 4.0s >= 3.5s -> Güvenli (True)
    assert tree.can_enter_intersection(dist_to_approaching_m=40.0, approaching_speed_mps=10.0) is True

    # 30 metrede 10 m/s hızla yaklaşan araç -> TTC = 3.0s < 3.5s -> Tehlikeli / Bekle (False)
    assert tree.can_enter_intersection(dist_to_approaching_m=30.0, approaching_speed_mps=10.0) is False


def test_doner_kavsak_yol_verme_durumu():
    """Kavşakta yaklaşan kritik araç varken YIELDING durumuna geçtiği test edilir."""
    tree = TeslaIntersectionDecisionTree(min_ttc_safe_s=3.5)
    vehs = [{"id": 1, "dist_m": 25.0, "speed_mps": 10.0}]  # TTC = 2.5s

    res = tree.evaluate_roundabout_scenario(ego_dist_to_yield_line_m=5.0, circulating_vehicles=vehs)
    assert res["state"] == RoundaboutState.YIELDING.value
    assert res["can_enter"] is False
    assert res["target_acc_mps2"] < 0.0


def test_doner_kavsak_giris_onayi_durumu():
    """Kavşakta tüm araçlar güvenli mesafedeyken ENTERING durumuna geçtiği test edilir."""
    tree = TeslaIntersectionDecisionTree(min_ttc_safe_s=3.5)
    vehs = [{"id": 1, "dist_m": 60.0, "speed_mps": 10.0}]  # TTC = 6.0s

    res = tree.evaluate_roundabout_scenario(ego_dist_to_yield_line_m=5.0, circulating_vehicles=vehs)
    assert res["state"] == RoundaboutState.ENTERING.value
    assert res["can_enter"] is True
    assert res["target_acc_mps2"] > 0.0
