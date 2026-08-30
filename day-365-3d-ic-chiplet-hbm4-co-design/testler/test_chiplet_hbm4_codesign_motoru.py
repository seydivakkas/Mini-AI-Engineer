"""
Day 365: 3D-IC Chiplet Architecture & HBM4 Memory Co-Design
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

from src.chiplet_hbm4_codesign_motoru import (
    ThroughSiliconViaLink,
    HBM4MemoryStack,
    ChipletComputeTile,
    ThreeDICCoDesignSimulator,
)
from src.chiplet_profilleyici import ChipletProfilleyici


def test_through_silicon_via_link_latency():
    """
    3D Dikey Silikon Geçiş (TSV) Gecikme Testi.
    """
    tsv = ThroughSiliconViaLink()
    assert tsv.latency_ps < 1.0 # 1 ps altı
    assert tsv.energy_pj_per_bit < 1.0


def test_hbm4_memory_stack_bandwidth():
    """
    HBM4 2048-Bit Bellek Bant Genişliği Testi.
    """
    hbm4 = HBM4MemoryStack(num_stacks=4, bus_width_bits=2048, pin_speed_gbps=8.0)
    assert hbm4.bw_per_stack_tb_s == pytest.approx(2.048, abs=0.01)
    assert hbm4.total_bw_tb_s == pytest.approx(8.192, abs=0.01)


def test_threed_ic_codesign_simulator_speedup():
    """
    Williams Roofline LLM Hızlanma Testi.
    """
    sim = ThreeDICCoDesignSimulator()
    res = sim.run_llm_roofline_benchmark()
    
    assert res["total_hbm4_bw_tb_s"] == pytest.approx(8.192, abs=0.01)
    assert res["llm_speedup"] > 50.0 # 64x hızlanma
    assert res["llm_decode_hbm4_tflops"] > res["llm_decode_ddr5_tflops"]


def test_chiplet_profiler_metrics():
    """
    3D-IC Profilleyici Metrik Testi.
    """
    mock_res = {
        "total_hbm4_bw_tb_s": 8.192,
        "llm_speedup": 64.0
    }
    metrics = ChipletProfilleyici.profille(mock_res)
    assert metrics["hbm4_bandwidth_score"] == 100.0
    assert metrics["chiplet_codesign_readiness"] > 98.0
