"""
Day 390: Unit Tests for Climate Engineering & Carbon Capture Optimization
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carbon_capture_causal_motoru import (
    DACReactorUnit,
    LangmuirAdsorptionModel,
    CausalInferenceEngine,
    CarbonCaptureBenchmark
)


def test_langmuir_adsorption_isotherm():
    """Langmuir adsorpsiyon modelinin nem sinerjisini ve CO2 tutunmasını doğru hesapladığını test eder."""
    model = LangmuirAdsorptionModel()
    q_dry = model.compute_adsorbed_co2(p_co2_kpa=0.042, temp_k=298.15, humidity=0.20)
    q_humid = model.compute_adsorbed_co2(p_co2_kpa=0.042, temp_k=298.15, humidity=0.80)

    assert q_dry > 0.0
    assert q_humid > q_dry, "Nemli ortamda katı amin adsorpsiyonu artmalıdır."


def test_desorption_thermal_energy():
    """Termal desorpsiyon enerji gereksiniminin pozitif ve sıcaklıkla orantılı olduğunu test eder."""
    model = LangmuirAdsorptionModel()
    energy_low = model.compute_desorption_energy_mj(adsorbed_mol=2.0, regen_temp_k=358.15)
    energy_high = model.compute_desorption_energy_mj(adsorbed_mol=2.0, regen_temp_k=378.15)

    assert energy_low > 0.0
    assert energy_high > energy_low


def test_causal_inference_engine_intervention():
    """Nedensel müdahale motorunun neme göre optimum desorpsiyon sıcaklığı belirlediğini test eder."""
    causal = CausalInferenceEngine()
    temp_dry = causal.evaluate_interventions(ambient_temp_k=295.0, humidity=0.20)
    temp_humid = causal.evaluate_interventions(ambient_temp_k=295.0, humidity=0.85)

    assert temp_dry >= 358.0
    assert temp_humid > temp_dry


def test_tam_carbon_capture_benchmark():
    """Tam endüstriyel doğrudan havadan karbon yakalama benchmarkını test eder."""
    bench = CarbonCaptureBenchmark(num_units=50)
    res = bench.kos(num_days=10)

    assert res["total_co2_captured_tons"] > 0.0
    assert res["specific_energy_consumption_mwh_ton"] <= 2.50
    assert res["capture_efficiency_pct"] > 85.0
    assert res["levelized_cost_usd_ton"] < 150.0
