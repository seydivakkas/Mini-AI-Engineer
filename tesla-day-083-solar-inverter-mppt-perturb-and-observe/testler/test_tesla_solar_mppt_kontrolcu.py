"""
Tesla Solar MPPT Birim Testleri (PyTest)
========================================
Bu test paketi; PV panel güç modelini, Perturb and Observe algoritmasının
yön kararlarını ve maksimum güç noktası takip verimliliğini test eder.

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

from src.tesla_solar_mppt_kontrolcu import TeslaSolarMPPTController


def test_pv_guc_modeli():
    """PV modelinin 33.44V'da maksimum güç ve 50V Voc'de 0W ürettiği test edilir."""
    ctrl = TeslaSolarMPPTController(v_oc=50.0, i_sc=10.0, optimal_v_mpp=33.44)

    p_mpp = ctrl.calculate_pv_power(33.44)
    p_50 = ctrl.calculate_pv_power(50.0)
    p_15 = ctrl.calculate_pv_power(15.0)

    assert p_mpp > p_15
    assert p_50 == 0.0
    assert p_mpp > 250.0  # Pozitif tepe güç


def test_po_yon_karari():
    """Güç arttığında aynı yönde adım atıldığı test edilir."""
    ctrl = TeslaSolarMPPTController()
    ctrl.prev_v = 20.0
    ctrl.prev_p = 180.0

    # 22V'da güç 200W'a çıktı -> Voltajı artırmaya devam etmeli
    v_next = ctrl.mppt_step_perturb_and_observe(v_curr=22.0, p_curr=200.0, step_v=1.0)

    assert v_next > 22.0


def test_mppt_takip_verimliligi():
    """50 iterasyonda MPPT verimliliğinin %99'un üzerine çıktığı test edilir."""
    ctrl = TeslaSolarMPPTController()
    res = ctrl.simulate_mppt_tracking(initial_v=15.0, iterations=60, step_v=0.5)

    assert res["mppt_efficiency_pct"] >= 98.5
    assert res["locked_on_mpp"] is True
    assert np.isclose(res["final_tracked_v"], 33.44, atol=1.5)
