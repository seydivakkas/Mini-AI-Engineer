"""
Tesla Gölge Modu ve Veri Motoru Birim Testleri (PyTest)
=======================================================
Bu test paketi; İnsan-Gölge model uyuşmazlık tetikleyicilerini,
uç klip paketlemesini ve A/B Z-testi istatistiksel anlamlılığını test eder.

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

from src.tesla_golge_modu_ve_veri_motoru import TeslaShadowModeDataEngine


def test_direksiyon_uyusmazlik_tetikleyici():
    """Direksiyon açısı farkı 5 derecenin üzerindeyken tetikleyicinin aktif olduğu test edilir."""
    engine = TeslaShadowModeDataEngine(steering_thresh_deg=5.0)

    # 6.0 derece fark -> Tetiklenmeli
    res_trig = engine.check_discrepancy_and_trigger(human_steering_deg=6.0, shadow_steering_deg=0.0, human_accel_mps2=0.0, shadow_accel_mps2=0.0)
    # 2.0 derece fark -> Tetiklenmemeli
    res_safe = engine.check_discrepancy_and_trigger(human_steering_deg=2.0, shadow_steering_deg=0.0, human_accel_mps2=0.0, shadow_accel_mps2=0.0)

    assert res_trig["is_triggered"] is True
    assert res_trig["clip_package"] is not None
    assert res_safe["is_triggered"] is False


def test_fren_ve_serit_uyusmazlik_tetikleyici():
    """Fren farkı 1.5 m/s^2 veya şerit komutu zıt olduğunda tetiklendiği test edilir."""
    engine = TeslaShadowModeDataEngine(accel_thresh_mps2=1.5)

    res_accel = engine.check_discrepancy_and_trigger(human_steering_deg=0.0, shadow_steering_deg=0.0, human_accel_mps2=-2.0, shadow_accel_mps2=0.0)
    res_lane = engine.check_discrepancy_and_trigger(human_steering_deg=0.0, shadow_steering_deg=0.0, human_accel_mps2=0.0, shadow_accel_mps2=0.0, human_lane_action="LEFT", shadow_lane_action="KEEP")

    assert res_accel["is_triggered"] is True
    assert res_lane["is_triggered"] is True


def test_ab_test_mpi_anlamlilik():
    """Daha az müdahale alan model B'nin MPI'sinin daha yüksek ve p < 0.05 olduğu test edilir."""
    engine = TeslaShadowModeDataEngine()
    ab_res = engine.evaluate_ab_test_significance(
        interventions_model_a=50, miles_model_a=10000.0,
        interventions_model_b=15, miles_model_b=10000.0
    )

    assert ab_res["mpi_model_b"] > ab_res["mpi_model_a"]
    assert ab_res["statistically_significant"] is True
    assert ab_res["p_value"] < 0.05
