"""
Day 324: Neuromorphic Hardware Mapping (Intel Loihi 2 & SynSense)
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

from src.loihi_mapper import (
    LoihiNeuroCore,
    AERPacketRouter,
    NeuromorphicHardwareMapper,
)
from src.loihi_profilleyici import LoihiProfilleyici


def test_loihi_neuro_core_int8_quantization():
    """
    LoihiNeuroCore INT8 kuantizasyon testi.
    """
    core = LoihiNeuroCore(core_id=0, grid_x=0, grid_y=0, weight_bits=8)
    w_fp32 = np.array([[-1.0, 0.0, 0.5, 1.0]], dtype=np.float32)
    
    scale = core.load_quantized_weights(w_fp32)
    assert core.weights_int8 is not None
    assert core.weights_int8.dtype == np.int8
    assert core.weights_int8[0, 3] == 127  # Maksimum INT8 değer 127
    assert core.weights_int8[0, 0] == -127 or core.weights_int8[0, 0] == -128


def test_aer_packet_router_hop_distance():
    """
    Manhattan Mesafe (Hop) Hesabı Testi.
    Hop = |x1 - x2| + |y1 - y2|
    """
    core1 = LoihiNeuroCore(core_id=0, grid_x=0, grid_y=0)
    core2 = LoihiNeuroCore(core_id=1, grid_x=3, grid_y=2)
    
    hops = AERPacketRouter.calculate_manhattan_distance(core1, core2)
    assert hops == 5  # |0-3| + |0-2| = 5


def test_hardware_mapper_core_partitioning():
    """
    Nöron sayısı core kapasitesini aştığında otomatik bölümleme testi.
    """
    mapper = NeuromorphicHardwareMapper(mesh_rows=4, mesh_cols=4, max_neurons_per_core=50)
    w_large = np.random.randn(120, 32).astype(np.float32)  # 120 nöron > 50 -> 3 çekirdek gerek
    
    mapping_info = mapper.map_snn_weights(w_large)
    assert mapping_info["used_cores"] == 3
    assert mapping_info["core_utilization_pct"] == (3 / 16) * 100.0


def test_hardware_mapper_capacity_overflow():
    """
    Kapasite aşıldığında ValueError fırlatma testi.
    """
    mapper = NeuromorphicHardwareMapper(mesh_rows=2, mesh_cols=2, max_neurons_per_core=10)  # Toplam 4 core * 10 = 40 nöron max
    w_overflow = np.random.randn(100, 16).astype(np.float32)  # 100 nöron > 40
    
    with pytest.raises(ValueError):
        mapper.map_snn_weights(w_overflow)


def test_loihi_profiler_metrics():
    """
    LoihiProfilleyici donanım enerji ve skor testi.
    """
    mapping_info = {"used_cores": 4, "total_cores": 16, "core_utilization_pct": 25.0, "sqnr_db": 35.0}
    metrics = LoihiProfilleyici.profille(mapping_info, aer_packets=[])
    
    assert "loihi_energy_uj" in metrics
    assert "energy_saving_x" in metrics
    assert metrics["energy_saving_x"] > 1.0
