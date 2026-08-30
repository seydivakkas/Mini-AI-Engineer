"""
Day 380: Unit Tests for Integrated Photonic-Silicon Heterogeneous AI Supercomputer (Phase 19 Finale)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from photonic_silicon_supercomputer_motoru import (
    PhotonicTensorCore,
    QuantumCoprosessorInterface,
    RISCVVectorHost,
    CPOInterconnectFabric,
    HeterogeneousSupercomputerSoC,
    SupercomputerBenchmark
)


def test_fotonik_mzi_gemm_ve_enerji():
    """Silikon Fotonik 16x16 MZI tensör çekirdeğinin GEMM hesaplamasını ve enerjisini test eder."""
    core = PhotonicTensorCore(dim=16)
    W = np.eye(16, dtype=np.float32)
    x = np.ones(16, dtype=np.float32) * 2.0

    y_out, energy_pj, lat_ns = core.execute_optical_gemm(W, x)

    assert len(y_out) == 16
    np.testing.assert_allclose(y_out, x, rtol=1e-5)
    assert lat_ns < 1.0, "Fotonik GEMM gecikmesi 1 nanosaniyenin altında olmalıdır."
    assert energy_pj > 0.0


def test_kuantum_qaoa_moe_yonlendirme():
    """Kuantum QPU arayüzünün MoE uzman yönlendirmesini test eder."""
    qpu = QuantumCoprosessorInterface(num_qubits=8)
    cost = np.random.randn(8, 8)
    
    assignment, lat_us = qpu.optimize_combinatorial_routing(cost)
    assert len(assignment) == 8
    assert set(assignment).issubset({0, 1})
    assert lat_us > 0.0


def test_riscv_vektor_fused_aktivasyon():
    """RISC-V Vektör hostunun Fused GeLU ve Softmax fonksiyonlarını test eder."""
    host = RISCVVectorHost(vlen_bits=256)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    probs = host.execute_fused_gelu_softmax(x)
    assert len(probs) == 4
    assert abs(np.sum(probs) - 1.0) < 1e-5, "Softmax olasılık toplamı 1.0 olmalıdır."
    assert np.all(probs >= 0.0)


def test_tam_supercomputer_benchmark_ve_tops_watt():
    """Tam FAZ 19 BÜYÜK FİNALİ heterojen benchmark akışını ve TOPS/Watt verimliliğini test eder."""
    bench = SupercomputerBenchmark()
    res = bench.kos(num_runs=20)

    assert res["avg_tops_per_watt"] > 5.0, "Heterojen TOPS/Watt verimliliği > 5 TOPS/W olmalıdır."
    assert res["avg_energy_gain_x"] >= 10.0, "Klasik GPU'ya göre en az 10x enerji tasarrufu sağlanmalıdır."
    assert res["avg_latency_ns"] < 5000.0, "Ortalama çıkarım gecikmesi < 5 us olmalıdır."
