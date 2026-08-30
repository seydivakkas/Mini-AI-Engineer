"""
Day 370: Reinforcement Learning-Based Thermal-Aware AI Chip Floorplanning
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

from src.thermal_floorplanning_rl_motoru import (
    ChipMacro,
    SiliconThermalDieGrid,
    RLMacroPlacerAgent,
    AIFloorplanningBenchmark,
)
from src.floorplanning_profilleyici import FloorplanningProfilleyici


def test_chip_macro_initialization():
    """
    Çip Makro Blok Başlatma Testi.
    """
    macro = ChipMacro("TensorCore_0", width=4, height=4, power_w=15.0, net_id=1)
    assert macro.name == "TensorCore_0"
    assert macro.w == 4
    assert macro.power == 15.0


def test_silicon_thermal_die_grid_computation():
    """
    Silikon Izgara 2B Sıcaklık Modeli Testi.
    """
    grid = SiliconThermalDieGrid(grid_size=10, t_ambient_c=35.0)
    m = ChipMacro("Core", 2, 2, 10.0)
    m.pos_x, m.pos_y = 4, 4
    t_map, t_peak, hpwl, ov = grid.compute_thermal_map_and_hpwl([m])
    
    assert t_map.shape == (10, 10)
    assert t_peak > 35.0
    assert ov == 0


def test_rl_macro_placer_agent_cooling():
    """
    RL Isı-Farkında Sıcaklık Düşüş Testi.
    """
    benchmark = AIFloorplanningBenchmark()
    res = benchmark.run_benchmark()
    
    assert res["t_peak_rl"] < res["t_peak_naive"]
    assert res["temp_reduction_c"] > 15.0 # En az 15 derece soğuma
    assert res["overlaps"] == 0


def test_floorplanning_profiler_metrics():
    """
    Floorplanning Profilleyici Metrik Testi.
    """
    mock_res = {
        "t_peak_naive": 104.5,
        "t_peak_rl": 78.2,
        "temp_reduction_c": 26.3,
        "hpwl_saving_pct": 24.5,
        "overlaps": 0
    }
    metrics = FloorplanningProfilleyici.profille(mock_res)
    assert metrics["overlap_score"] == 100.0
    assert metrics["floorplanning_readiness"] > 98.0
