"""
Day 384: Unit Tests for Autonomous Chemical Reactor Control with Real-Time NMR Feedback
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chemical_reactor_nmr_motoru import (
    CSTRReactorState,
    NMRSpectrum,
    NMRSpectrometerModel,
    CSTRKineticsEngine,
    ReactorAdaptiveController,
    ChemicalReactorBenchmark
)


def test_nmr_spectrum_generation_and_peaks():
    """Çevrimiçi NMR simülatörünün Lorentzian piklerini ve SNR değerini doğru ürettiğini test eder."""
    nmr = NMRSpectrometerModel(num_points=300)
    spec = nmr.generate_spectrum(c_a=1.5, c_b=1.2, c_c=0.8, c_d=0.1)

    assert len(spec.ppm_axis) == 300
    assert spec.snr_db > 10.0
    # 4.8 ppm (C piki) civarında sinyal şiddeti pozitif olmalıdır
    mask_c = np.abs(spec.ppm_axis - 4.8) < 0.2
    assert np.max(spec.intensity[mask_c]) > 0.5


def test_cstr_kinetics_rk4_mass_balance():
    """CSTR kinetik RK4 çözücüsünün derişimleri pozitif tuttuğunu ve tepkimeyi ilerlettiğini test eder."""
    kinetics = CSTRKineticsEngine(volume_l=10.0)
    s0 = CSTRReactorState(temp_k=335.0, jacket_temp_k=325.0, c_a=1.5, c_b=1.8, c_c=0.0, c_d=0.0)

    s1 = kinetics.step_rk4(s0, dt_min=0.1)
    assert s1.c_c > 0.0, "Tepkime sonucunda ürün C oluşmalıdır."
    assert s1.c_a < s0.c_a, "Reaktif A tüketilmelidir."
    assert s1.temp_k > 0.0


def test_reactor_adaptive_controller_cooling():
    """Uyarlamalı kontrolcünün yüksek sıcaklıkta acil ceket soğutması yaptığını test eder."""
    ctrl = ReactorAdaptiveController(target_temp_k=338.0, critical_temp_k=360.0)
    high_temp_state = CSTRReactorState(temp_k=358.0)
    nmr_feedback = {"C": 1.2, "D": 0.1}

    jacket_t, flow = ctrl.compute_control_actions(high_temp_state, nmr_feedback)
    assert jacket_t <= 285.0, "Kritik sıcaklığa yaklaşıldığında acil soğutma uygulanmalıdır."


def test_tam_chemical_reactor_benchmark():
    """Tam otonom kimyasal reaktör ve NMR geri bildirim benchmarkını test eder."""
    bench = ChemicalReactorBenchmark()
    res = bench.kos(num_steps=30)

    assert res["num_steps"] == 30
    assert res["final_yield_pct"] > 30.0, "Hedef ürün verimi pozitif ve anlamlı olmalıdır."
    assert res["thermal_runaway_safe"] is True, "Termal kaçak meydana gelmemelidir."
    assert res["max_reactor_temp_k"] < 360.0
