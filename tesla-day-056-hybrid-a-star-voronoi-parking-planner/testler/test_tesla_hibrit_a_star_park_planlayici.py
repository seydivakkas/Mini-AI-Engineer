"""
Tesla Hibrit A* Park Planlayıcı Birim Testleri (PyTest)
========================================================
Bu test paketi; Kinematik bisiklet adımı geçişini, Voronoi engel
maliyet fonksiyonunu ve paralel park yörünge başarı kriterlerini test eder.

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

from src.tesla_hibrit_a_star_park_planlayici import TeslaHybridAStarParkPlanner


def test_kinematik_bisiklet_adimi_duz_ve_donus():
    """Direksiyon 0 iken düz ilerleme, direksiyon açılı iken yaw değişiminin gerçekleştiği test edilir."""
    planner = TeslaHybridAStarParkPlanner(wheelbase_m=2.875, dt_s=0.1)
    s0 = np.array([0.0, 0.0, 0.0])

    # Düz İlerleme
    s_straight = planner.step_kinematic_bicycle(s0, velocity_mps=10.0, steer_rad=0.0)
    assert np.isclose(s_straight[0], 1.0)
    assert np.isclose(s_straight[1], 0.0)
    assert np.isclose(s_straight[2], 0.0)

    # Sağa Dönüş
    s_turn = planner.step_kinematic_bicycle(s0, velocity_mps=10.0, steer_rad=0.2)
    assert s_turn[2] > 0.0  # Pozitif yaw açısı


def test_voronoi_engel_maliyeti():
    """Engele yakın konumun uzak konuma göre çok daha yüksek maliyet ürettiği test edilir."""
    planner = TeslaHybridAStarParkPlanner()
    obstacles = np.array([[5.0, 0.0]])

    cost_close = planner.compute_voronoi_obstacle_cost(np.array([4.8, 0.0]), obstacles)  # 0.2m mesafe -> Çarpışma riski
    cost_far = planner.compute_voronoi_obstacle_cost(np.array([2.0, 0.0]), obstacles)    # 3.0m mesafe

    assert cost_close > cost_far
    assert cost_close == 1000.0


def test_otonom_paralel_park_basarisi():
    """Yörüngenin başarıyla park cebine ulaştığı ve son hatanın eşik altında kaldığı test edilir."""
    planner = TeslaHybridAStarParkPlanner()
    res = planner.plan_parallel_parking_trajectory()

    assert res["success"] is True
    assert res["final_pos_error_m"] < 0.4
    assert res["final_yaw_error_deg"] < 5.0
    assert len(res["trajectory"]) == 36
