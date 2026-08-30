"""
Day 369: Mixed-Signal ADC/DAC Power Optimization for Analog AI Accelerators
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.adc_dac_optimizasyon_motoru import (
    SuccessiveApproximationADC,
    PulseWidthModulationDAC,
    AdaptiveMixedSignalCrossbar,
    ADCDACPowerBenchmark,
)
from src.adc_dac_profilleyici import ADCDACProfilleyici


def test_sar_adc_quantization():
    """
    SAR ADC Kuantalama ve Güç Modeli Testi.
    """
    adc = SuccessiveApproximationADC(resolution_bits=6, v_ref=1.0)
    sig = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    levels, rec = adc.quantize(sig)
    
    assert len(levels) == 5
    assert (rec <= 1.0).all()
    assert adc.power_uw > 0.0


def test_pwm_dac_voltage_range():
    """
    PWM DAC Giriş Modülatörü Testi.
    """
    dac = PulseWidthModulationDAC(resolution_bits=4, v_max=0.8)
    dig = np.array([0, 5, 10, 15])
    v_out = dac.convert(dig)
    
    assert v_out[0] == 0.0
    assert v_out[-1] == pytest.approx(0.8, abs=0.01)


def test_adaptive_mixed_signal_crossbar_power_savings():
    """
    Adaptif Kolon Kapılama Güç Tasarrufu Testi.
    """
    crossbar = AdaptiveMixedSignalCrossbar(rows=8, cols=8)
    x = np.random.randint(0, 16, 8)
    res = crossbar.compute_fixed_vs_adaptive(x)
    
    assert res["power_saving_pct"] > 50.0 # %50'den fazla tasarruf
    assert res["cosine_similarity"] > 0.95


def test_adc_dac_profiler_metrics():
    """
    ADC/DAC Profilleyici Metrik Testi.
    """
    mock_res = {
        "fixed_power_mw": 4.5,
        "adaptive_power_mw": 1.4,
        "power_saving_pct": 68.8,
        "cosine_similarity": 0.995
    }
    metrics = ADCDACProfilleyici.profille(mock_res)
    assert metrics["power_saving_pct"] == 68.8
    assert metrics["mixed_signal_readiness"] > 98.0
