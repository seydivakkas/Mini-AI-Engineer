"""
Day 336: Triton Neuromorphic GPU Kernel: Sparse Spiking Matrix Multiplication (SpMM)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.triton_spmm_motoru import (
    SparseSpikeMatrix,
    PyTorchSparseSpMM,
    SpikingKernelBenchmark,
)
from src.triton_profilleyici import TritonProfilleyici


def test_sparse_spike_matrix_conversion():
    """
    Seyrek Spike Matris Dönüşüm ve Seyreklik Hesabı Testi.
    """
    dense_spikes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sparse = SparseSpikeMatrix(dense_spikes)
    
    assert sparse.nnz == 2
    assert abs(sparse.sparsity_pct - 66.666) < 1.0


def test_pytorch_sparse_spmm_numerical_equivalence():
    """
    Seyrek SpMM Çekirdeği ile Yoğun GEMM Sayısal Eşdeğerlik Testi.
    """
    dense_spikes = (torch.rand(32, 64) > 0.8).float()
    weight = torch.randn(64, 128)
    
    sparse_spikes = SparseSpikeMatrix(dense_spikes)
    
    y_dense = torch.matmul(dense_spikes, weight)
    y_sparse = PyTorchSparseSpMM.spmm_forward(sparse_spikes, weight)
    
    max_diff = torch.max(torch.abs(y_dense - y_sparse)).item()
    assert max_diff < 1e-5


def test_spiking_kernel_benchmark_levels():
    """
    SpMM Kernel Performans Benchmark Testi.
    """
    res = SpikingKernelBenchmark.benchmark_sparsity_levels(
        batch_size=16, in_dim=32, out_dim=32, sparsity_levels=[50.0, 90.0]
    )
    
    assert "speedup_factors" in res
    assert len(res["speedup_factors"]) == 2
    assert res["max_errors"][0] < 1e-4


def test_triton_profiler_metrics():
    """
    Triton GPU Çekirdeği Profilleyici Metrik Doğrulaması.
    """
    metrics = TritonProfilleyici.profille(sparsity_pct=90.0, speedup_factor=5.0, max_error=0.0)
    
    assert metrics["precision_score"] == 100.0
    assert metrics["triton_readiness_score"] > 80.0
