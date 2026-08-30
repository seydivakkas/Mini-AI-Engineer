"""
Day 392: Unit Tests for Nuclear Fusion Plasma Control & Tokamak Deep RL
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fusion_tokamak_rl_motoru import (
    TokamakPlasmaState,
    GradShafranovMHDEquilibrium,
    TokamakMultiCoilEnvironment,
    PPOPlasmaRLController,
    FusionTokamakBenchmark
)


def test_grad_shafranov_equilibrium_grid():
    """Grad-Shafranov çözücüsünün geçerli 2B manyetik akı ızgarası ürettiğini test eder."""
    solver = GradShafranovMHDEquilibrium(R0=6.2, a=2.0)
    R_grid, Z_grid, psi_grid = solver.solve_equilibrium_flux_grid(grid_size=30)

    assert R_grid.shape == (30, 30)
    assert Z_grid.shape == (30, 30)
    assert psi_grid.shape == (30, 30)
    assert np.max(psi_grid) <= 1.0


def test_tokamak_multi_coil_vde_drift():
    """Kontrol uygulanmadığında plazmanın dikey olarak kararsız sürüklendiğini test eder."""
    env = TokamakMultiCoilEnvironment()
    state = TokamakPlasmaState(
        t_ms=0.0, R_p_m=6.20, Z_p_m=0.01, I_p_MA=15.0,
        beta_N=2.4, q_95=3.4, elongation_kappa=1.75, triangularity_delta=0.35
    )
    # Sıfır voltaj uygula
    zero_voltages = np.zeros(12)
    next_state = env.step(state, zero_voltages)

    # Kararsız büyüme: Z_p büyümelidir
    assert abs(next_state.Z_p_m) > abs(state.Z_p_m)


def test_ppo_controller_stabilization_action():
    """Deep RL kontrolcünün dikey sapmaya karşı düzeltici voltaj ürettiğini test eder."""
    controller = PPOPlasmaRLController()
    state_up = TokamakPlasmaState(
        t_ms=0.0, R_p_m=6.20, Z_p_m=0.02, I_p_MA=15.0,
        beta_N=2.4, q_95=3.4, elongation_kappa=1.75, triangularity_delta=0.35
    )
    voltages = controller.compute_action(state_up, prev_z=0.019)

    # Yukarı kayan plazmayı aşağı çekmek için üst bobinler pozitif / alt negatif olmalıdır
    assert voltages.shape == (12,)
    assert voltages[0] > 0.0
    assert voltages[6] < 0.0


def test_tam_fusion_tokamak_benchmark():
    """Tam tokamak füzyon plazma kararlılık benchmarkını test eder."""
    bench = FusionTokamakBenchmark(steps=500)
    res = bench.kos()

    assert res["total_steps"] == 500
    assert res["vde_avoidance_success_pct"] == 100.0
    assert res["rms_vertical_error_mm"] < 5.0
    assert res["max_coil_voltage_kv"] <= 10.0
