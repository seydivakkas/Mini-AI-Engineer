"""
Day 397: Unit Tests for Quantum-Assisted Neural PDE Ocean-Climate Solver
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ocean_climate_pde_motoru import (
    OceanGridState,
    FourierNeuralOperatorPDE,
    AMOCStabilityAnalyzer,
    QuantumAcceleratedClimateBenchmark
)


def test_fourier_neural_operator_step():
    """Fourier Nöral Operatörünün enerji korunumunu ve şekil doğruluğunu test eder."""
    fno = FourierNeuralOperatorPDE(modes=8, width=16)
    temp = np.ones((32, 64)) * 15.0
    sal = np.ones((32, 64)) * 35.0

    new_t, new_s, err = fno.solve_step(temp, sal, dt_years=1.0)
    assert new_t.shape == (32, 64)
    assert new_s.shape == (32, 64)
    assert err < 0.05


def test_amoc_stability_analyzer_weakening():
    """Tatlı su deşarjının AMOC debisini zayıflattığını test eder."""
    analyzer = AMOCStabilityAnalyzer(baseline_amoc_sv=18.5)
    amoc_1950 = analyzer.compute_amoc_strength(year=1950, freshwater_flux_sv=0.0)
    amoc_2050 = analyzer.compute_amoc_strength(year=2050, freshwater_flux_sv=0.15)

    assert amoc_1950 > amoc_2050
    assert amoc_2050 > 2.0


def test_ocean_grid_state_initialization():
    """Okyanus grid hücresi veri yapısını test eder."""
    state = OceanGridState(lat=45.0, lon=-30.0, depth_m=100.0, temperature_c=12.5, salinity_psu=35.2, velocity_u=0.1, velocity_v=0.05, streamfunction_sv=16.0)
    assert state.temperature_c == 12.5
    assert state.streamfunction_sv == 16.0


def test_tam_quantum_climate_benchmark():
    """Tam kuantum destekli iklim benchmarkını test eder."""
    bench = QuantumAcceleratedClimateBenchmark(simulation_years=50)
    res = bench.kos()

    assert res["simulation_years"] == 50
    assert res["speedup_vs_fortran"] >= 1000.0
    assert res["avg_energy_conservation_error_pct"] < 0.05
    assert len(res["amoc_timeline"]) == 50
