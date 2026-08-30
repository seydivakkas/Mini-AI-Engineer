"""
Day 359: Extreme-Temperature Adaptive Neural Scaling & Dynamic Voltage/Frequency Scaling (DVFS)
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

from src.thermal_scaling_ai_motoru import (
    ThermalOperatingMode,
    AvionicsThermalDieSimulator,
    ElasticNeuralScalingModel,
    DynamicThermalGovernorAgent,
    ExtremeTemperatureFlightMission,
)
from src.thermal_profilleyici import ThermalProfilleyici


def test_avionics_thermal_die_simulator():
    """
    Termal RC Çip Simülatörü Testi.
    """
    sim = AvionicsThermalDieSimulator()
    t_die, p_tot = sim.step_thermal(clock_ghz=1.2, model_load_ratio=1.0, t_ambient=30.0, dt=1.0)
    assert t_die > 35.0
    assert p_tot > 10.0


def test_elastic_neural_scaling_model():
    """
    Elastik Nöral Model Ölçekleme Testi.
    """
    acc_full, load_full = ElasticNeuralScalingModel.infer(np.zeros(5), ThermalOperatingMode.FULL_POWER_HIGH_PERF)
    acc_crit, load_crit = ElasticNeuralScalingModel.infer(np.zeros(5), ThermalOperatingMode.CRITICAL_HEAT_SURVIVAL)
    
    assert acc_full > acc_crit
    assert load_full > load_crit
    assert load_crit < 0.30


def test_dynamic_thermal_governor_agent():
    """
    Termal Yönetici DVFS Karar Testi.
    """
    gov = DynamicThermalGovernorAgent(t_warm=60.0, t_critical=85.0)
    mode1, clk1 = gov.select_mode_and_clock(40.0)
    mode2, clk2 = gov.select_mode_and_clock(70.0)
    mode3, clk3 = gov.select_mode_and_clock(90.0)
    
    assert mode1 == ThermalOperatingMode.FULL_POWER_HIGH_PERF
    assert clk1 == 1.20
    assert mode2 == ThermalOperatingMode.WARM_BALANCED
    assert clk2 == 0.80
    assert mode3 == ThermalOperatingMode.CRITICAL_HEAT_SURVIVAL
    assert clk3 == 0.40


def test_thermal_profiler_metrics():
    """
    Termal Profilleyici Metrik Testi.
    """
    mock_res = {
        "max_ai_temp": 82.5,
        "survived_mission": True
    }
    metrics = ThermalProfilleyici.profille(mock_res)
    assert metrics["survived_mission"] is True
    assert metrics["overheat_prevention_score"] == 100.0
    assert metrics["thermal_survival_readiness"] > 95.0
