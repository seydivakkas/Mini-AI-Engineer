"""
Day 372: Custom RISC-V Vector Extension ISA Design for Transformer Kernels
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

from src.riscv_transformer_isa_motoru import (
    RISCVVectorRegisterFile,
    CustomTransformerISAProcessor,
    TransformerKernelBenchmark,
)
from src.riscv_isa_profilleyici import RISCVISAProfilleyici


def test_riscv_vrf_read_write():
    """
    RISC-V Vektör Kayıt Dosyası Okuma-Yazma Testi.
    """
    vrf = RISCVVectorRegisterFile(vlen_bits=256)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    vrf.write(0, data)
    read_data = vrf.read(0)
    assert np.allclose(data, read_data)


def test_custom_instructions_execution():
    """
    Özel RISC-V Komut Yürütme Testi (GeLU, Softmax, LayerNorm).
    """
    cpu = CustomTransformerISAProcessor()
    data = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.2], dtype=np.float32)
    cpu.vrf.write(0, data)
    
    # 1. GeLU
    cpu.exec_v_gelu_approx(vd=1, vs2=0)
    gelu_res = cpu.vrf.read(1)
    assert gelu_res[0] == 0.0 # GeLU(0) = 0
    assert gelu_res[1] > 0.8
    
    # 2. Softmax
    sum_e = cpu.exec_v_softmax_exp_sum(vd=2, vs2=0, max_val=2.0)
    assert sum_e > 0.0
    
    # 3. LayerNorm
    cpu.exec_v_layernorm_fused(vd=3, vs2=0)
    ln_res = cpu.vrf.read(3)
    assert np.abs(np.mean(ln_res)) < 1e-4 # Ortalama sıfır


def test_transformer_kernel_benchmark():
    """
    Transformer Çekirdeği Komut ve Saykıl Hızlanma Testi.
    """
    benchmark = TransformerKernelBenchmark()
    res = benchmark.run_benchmark(seq_len=4, hidden_dim=8)
    
    assert res["instruction_reduction"] > 10.0 # 10 kattan fazla komut tasarrufu
    assert res["cycle_speedup"] > 10.0
    assert res["mse_fidelity"] < 1e-4


def test_riscv_isa_profiler_metrics():
    """
    RISC-V Profilleyici Metrik Testi.
    """
    mock_res = {
        "scalar_instructions": 4000,
        "custom_instructions": 24,
        "instruction_reduction": 166.0,
        "scalar_cycles": 6400,
        "custom_cycles": 56,
        "cycle_speedup": 114.0,
        "mse_fidelity": 1e-6
    }
    metrics = RISCVISAProfilleyici.profille(mock_res)
    assert metrics["fidelity_score"] == 100.0
    assert metrics["isa_readiness_score"] > 95.0
