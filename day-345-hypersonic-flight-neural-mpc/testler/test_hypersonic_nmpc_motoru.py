"""
Day 345: Hypersonic Flight Neural Model Predictive Control (Neural MPC)
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

from src.hypersonic_nmpc_motoru import (
    HypersonicAeroDynamics,
    NeuralDynamicsSurrogate,
    HighSpeedNeuralMPC,
)
from src.nmpc_profilleyici import NMPCProfilleyici


def test_hypersonic_derivatives_shape():
    """
    Hipersonik Türev Hesaplama Vektör Boyut Testi.
    """
    aero = HypersonicAeroDynamics()
    state = np.array([1800.0, 0.0, 0.05, 0.0])
    derivs = aero.compute_derivatives(state, delta_e=0.0)
    
    assert len(derivs) == 4
    assert derivs[0] < 0.0 # Sürükleme kuvvetinden dolayı hız azalır


def test_neural_dynamics_surrogate_step():
    """
    Nöral Dinamik Vekili İleri Adım Testi.
    """
    aero = HypersonicAeroDynamics()
    surrogate = NeuralDynamicsSurrogate(aero)
    state_0 = np.array([1800.0, 0.0, 0.05, 0.0])
    
    next_state = surrogate.predict_next_state(state_0, delta_e=0.02, dt=0.02)
    assert len(next_state) == 4
    assert next_state[0] > 0.0


def test_high_speed_neural_mpc_optimization():
    """
    Nöral MPC Ufuk Optimizasyon Testi.
    """
    aero = HypersonicAeroDynamics()
    surrogate = NeuralDynamicsSurrogate(aero)
    nmpc = HighSpeedNeuralMPC(surrogate, horizon=10, dt=0.02)
    
    state = np.array([1800.0, 0.0, np.radians(2.0), 0.0])
    target_alpha = np.radians(5.0)
    
    opt_res = nmpc.optimize_control(state, target_alpha)
    assert "optimal_delta_e_rad" in opt_res
    assert abs(opt_res["optimal_delta_e_deg"]) <= 20.0
    assert opt_res["cost"] < 1000.0


def test_nmpc_profiler_metrics():
    """
    Hipersonik NMPC Profilleyici Testi.
    """
    metrics = NMPCProfilleyici.profille(
        mean_alpha_error_deg=0.05,
        max_elevon_deg=12.0,
        mean_solve_time_ms=0.2
    )
    
    assert metrics["tracking_score"] == 100.0
    assert metrics["stability_score"] == 100.0
    assert metrics["flight_safety_score"] > 95.0
