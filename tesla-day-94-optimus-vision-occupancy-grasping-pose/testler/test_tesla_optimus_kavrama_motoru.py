"""
Tesla Optimus Görsel Kavrama Birim Testleri (PyTest)
====================================================
Bu test paketi; mikro-voksel doluluk üretimini, 6-DoF kavrama pozunu
ve dokunsal kuvvet regülasyonunu test eder.

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

from src.tesla_optimus_kavrama_motoru import TeslaOptimusVisionGraspEngine


def test_mikro_voksel_ve_kavrama_pozu():
    """3D mikro-vokselden 6-DoF SE(3) kavrama duruşunun başarıyla çıkarıldığı test edilir."""
    engine = TeslaOptimusVisionGraspEngine()
    grid = engine.generate_micro_occupancy_grid(target_object="4680_BATTERY_CELL")

    assert np.sum(grid) > 0.0

    pose_res = engine.estimate_6dof_grasp_pose(grid)
    assert pose_res["success"] is True
    assert len(pose_res["p_grasp_m"]) == 3
    assert pose_res["confidence_score"] > 0.95


def test_dokunsal_kuvvet_kontrolu_yumurta():
    """Kırılgan yumurtanın 2.4 N kuvvetle güvenle tutulduğu (kırılmadığı/düşürülmediği) test edilir."""
    engine = TeslaOptimusVisionGraspEngine()

    # İdeal tutuş (2.0 mm parmak esnemesi -> 2.4 N)
    res_safe = engine.regulate_tactile_grip_force(finger_displacement_mm=2.0, object_type="DELICATE_EGG")
    assert res_safe["is_safe_grip"] is True
    assert res_safe["is_crushed"] is False
    assert res_safe["is_dropped"] is False

    # Aşırı baskı (4.0 mm esneme -> 4.8 N -> Kırılma)
    res_crush = engine.regulate_tactile_grip_force(finger_displacement_mm=4.0, object_type="DELICATE_EGG")
    assert res_crush["is_crushed"] is True
    assert res_crush["is_safe_grip"] is False


def test_dokunsal_kuvvet_kontrolu_pil_hucresi():
    """4680 pil hücresinin sağlam kavrandığı test edilir."""
    engine = TeslaOptimusVisionGraspEngine()
    res_bat = engine.regulate_tactile_grip_force(finger_displacement_mm=10.0, object_type="4680_BATTERY_CELL")
    assert res_bat["is_safe_grip"] is True
    assert res_bat["measured_force_n"] == 12.0
