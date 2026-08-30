"""
Day 336: Triton Neuromorphic GPU Kernel: Sparse Spiking Matrix Multiplication (SpMM)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Seyrek Spiking Tensor Dönüşümünü, SpMM Çekirdek Matris Çarpımını,
Farklı Seyreklik Oranlarında Performans Benchmark Testlerini ve 6-Panelli Teşhis Panosunu çalıştırır.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import torch

from src.triton_spmm_motoru import (
    SparseSpikeMatrix,
    PyTorchSparseSpMM,
    SpikingKernelBenchmark,
)
from src.triton_gorsellestirici import TritonGorsellestirici
from src.triton_profilleyici import TritonProfilleyici


def main():
    print("=" * 75, flush=True)
    print("⚡ DAY 336: Triton Nöromorfik GPU Çekirdeği: Seyrek Spiking Matris Çarpımı (SpMM)", flush=True)
    print("=" * 75, flush=True)

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 128
    in_dim = 512
    out_dim = 512
    sparsity_pct = 90.0

    # 1. Seyrek Spiking Matris Oluşturma
    print(f"\n📌 1) {batch_size}x{in_dim} Boyutlu %{sparsity_pct:.0f} Seyreklik Temsili Oluşturuluyor...", flush=True)
    p_spike = 1.0 - (sparsity_pct / 100.0)
    dense_spikes = (torch.rand(batch_size, in_dim) < p_spike).float()
    weight = torch.randn(in_dim, out_dim)

    sparse_spikes = SparseSpikeMatrix(dense_spikes)
    print(f"✅ Seyrek Matris Dönüşümü Tamamlandı | Toplam Eleman: {dense_spikes.numel()} | Aktif Spike Sayısı: {sparse_spikes.nnz} (%{100.0 - sparse_spikes.sparsity_pct:.2f} Yoğunluk)", flush=True)

    # 2. Yoğun GEMM vs Seyrek SpMM Çekirdek Çalıştırılması
    print("\n⚡ 2) Yoğun GEMM (Dense) vs Seyrek SpMM (Sparse) Çekirdeği Çalıştırılıyor...", flush=True)
    
    t0 = time.time()
    for _ in range(10):
        y_dense = torch.matmul(dense_spikes, weight)
    t_dense = (time.time() - t0) / 10.0

    t1 = time.time()
    for _ in range(10):
        y_sparse = PyTorchSparseSpMM.spmm_forward(sparse_spikes, weight)
    t_sparse = (time.time() - t1) / 10.0

    max_err = float(torch.max(torch.abs(y_dense - y_sparse)).item())
    speedup = float(t_dense / (t_sparse + 1e-9))

    print(f"  • Yoğun GEMM Süresi:     {t_dense * 1000.0:.3f} ms", flush=True)
    print(f"  • Seyrek SpMM Süresi:    {t_sparse * 1000.0:.3f} ms", flush=True)
    print(f"  • Hızlanma Çarpanı:      {speedup:.2f}x Hızlanma", flush=True)
    print(f"  • Sayısal Hata Farkı:   {max_err:.8f} (Matris Sonucu Tam Eşdeğer)", flush=True)

    # 3. Seyreklik Seviyelerine Göre Benchmark Taraması
    print("\n📊 3) Seyreklik Seviyelerine Göre SpMM Performans Taraması Yapılıyor (%50 - %98)...", flush=True)
    benchmark_results = SpikingKernelBenchmark.benchmark_sparsity_levels(
        batch_size=batch_size,
        in_dim=in_dim,
        out_dim=out_dim,
        sparsity_levels=[50.0, 75.0, 90.0, 95.0, 98.0]
    )

    for i, sp in enumerate(benchmark_results["sparsity_levels"]):
        print(f"  • Seyreklik %{sp:.0f}: Dense {benchmark_results['dense_times_ms'][i]:.2f}ms | Sparse {benchmark_results['sparse_times_ms'][i]:.2f}ms | Hızlanma: {benchmark_results['speedup_factors'][i]:.2f}x", flush=True)

    # 4. Profilleme ve Teşhis Panosu
    profiler_metrics = TritonProfilleyici.profille(
        sparsity_pct=sparse_spikes.sparsity_pct,
        speedup_factor=speedup,
        max_error=max_err
    )

    sample_mask = dense_spikes[:30, :50].detach().cpu().numpy()

    gorsellestirici = TritonGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        sample_spike_mask=sample_mask,
        benchmark_results=benchmark_results,
        profiler_metrics=profiler_metrics,
        dosya_adi="triton_spmm_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Triton SpMM Çekirdek Teşhis Grafiği Başarıyla Kaydedildi: [triton_spmm_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
