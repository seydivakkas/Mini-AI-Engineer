"""
Day 385: Unit Tests for Sub-Millimeter Precision Microsurgery Robot
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from microsurgery_robot_motoru import (
    NeedleState,
    TremorCancellationKalmanFilter,
    VascularAnastomosisPlanner,
    ImpedanceForceFeedbackController,
    MicrosurgeryBenchmark
)


def test_tremor_kalman_filter_attenuation():
    """Kalman filtresinin 10 Hz el titremesini etkili şekilde sönümlediğini test eder."""
    flt = TremorCancellationKalmanFilter(dt_s=0.01, f_tremor_hz=10.0, r_pole=0.60)
    
    # 10 Hz titreme sinyali
    raw_signals = [0.15 * np.sin(2.0 * np.pi * 10.0 * (i * 0.01)) for i in range(100)]
    filtered_signals = [flt.filter_position(s) for s in raw_signals]

    raw_var = np.var(raw_signals[30:])
    filt_var = np.var(filtered_signals[30:])
    assert filt_var < raw_var * 0.35, "Filtre titreme varyansını en az %65 düşürmelidir."


def test_vascular_anastomosis_planner_trajectory():
    """Vasküler anastomoz planlayıcısının 3B kavisli dikiş yolu ürettiğini test eder."""
    planner = VascularAnastomosisPlanner(vessel_radius_mm=0.4, needle_radius_mm=1.2)
    path = planner.generate_stitch_trajectory(num_points=50)

    assert len(path) == 50
    assert isinstance(path[0], np.ndarray)
    assert path[0].shape == (3,)
    assert path[-1][2] > path[0][2]


def test_impedance_controller_force_safety():
    """Empedans kontrolcüsünün delinme sonrası kuvveti düşürdüğünü ve yırtılmayı önlediğini test eder."""
    ctrl = ImpedanceForceFeedbackController(max_safe_force_n=0.25)
    
    f1, s1, p1 = ctrl.compute_interaction(np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0]), np.zeros(3))
    assert not p1
    assert f1 > 0.0

    f2, s2, p2 = ctrl.compute_interaction(np.array([0.0, 0.0, 0.0]), np.array([1.5, 0.0, 0.0]), np.zeros(3))
    assert bool(p2) is True
    assert f2 < 0.25, "Kuvvet güvenli sınırı aşmamalıdır."


def test_tam_microsurgery_benchmark():
    """Tam mikro-cerrahi anastomoz benchmarkını test eder."""
    bench = MicrosurgeryBenchmark()
    res = bench.kos(num_steps=100)

    assert res["num_steps"] == 100
    assert res["avg_positioning_error_um"] < 50.0, "Milimetre-altı konumlandırma hassasiyeti korunmalıdır."
    assert res["tissue_integrity_safe"] is True
    assert res["tremor_attenuation_pct"] > 40.0
