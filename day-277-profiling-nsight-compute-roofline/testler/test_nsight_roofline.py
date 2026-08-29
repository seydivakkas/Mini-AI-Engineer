"""
PyTest Birim Testleri - Day 277 (FAZ 14): NVIDIA Nsight Compute & Roofline Modeli.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.roofline_profilleyici_motoru import NsightRooflineEngine
from src.roofline_raporlayici import RooflineRaporlayici
from src.gorsellestirici import RooflineGorsellestirici


def test_h100_hardware_constants():
    """1. H100 donanım sabitleri ve Ridge Point doğru hesaplanmalıdır."""
    assert NsightRooflineEngine.PEAK_TFLOPS_FP16 == 1979.0
    assert NsightRooflineEngine.HBM3_BANDWIDTH_TB_S == 3.35
    assert np.isclose(NsightRooflineEngine.RIDGE_POINT, 1979.0 / 3.35, atol=0.1)


def test_attainable_performance_memory_bound():
    """2. Düşük aritmetik yoğunlukta (Memory-bound) performans bant genişliğiyle sınırlanmalıdır."""
    i = 2.0  # FLOP / Byte
    attainable = NsightRooflineEngine.calculate_attainable_performance(arithmetic_intensity=i)
    assert np.isclose(attainable, 2.0 * 3.35, atol=0.01)
    assert attainable < NsightRooflineEngine.PEAK_TFLOPS_FP16


def test_attainable_performance_compute_bound():
    """3. Yüksek aritmetik yoğunlukta (Compute-bound) performans tepe TFLOPS'a doymalıdır."""
    i = 1000.0  # FLOP / Byte
    attainable = NsightRooflineEngine.calculate_attainable_performance(arithmetic_intensity=i)
    assert attainable == NsightRooflineEngine.PEAK_TFLOPS_FP16


def test_kernel_profile_analysis_classification():
    """4. analyze_kernel_profile memory-bound ve compute-bound durumlarını doğru sınıflandırmalıdır."""
    # Softmax (Memory bound)
    res_mem = NsightRooflineEngine.analyze_kernel_profile("Test Softmax", 1e8, 5e7, 0.015)
    assert res_mem["is_memory_bound"] is True
    assert "Memory-Bound" in res_mem["bottleneck_type"]
    assert "Long Scoreboard" in res_mem["dominant_stall"]

    # Fused GEMM (Compute bound)
    res_comp = NsightRooflineEngine.analyze_kernel_profile("Test GEMM", 4e12, 4e9, 2.0)
    assert res_comp["is_memory_bound"] is False
    assert "Compute-Bound" in res_comp["bottleneck_type"]


def test_benchmark_suite_generation():
    """5. get_standard_benchmark_suite 4 temel LLM çekirdeğini analiz etmelidir."""
    suite = NsightRooflineEngine.get_standard_benchmark_suite()
    assert len(suite) == 4
    names = [k["kernel_name"] for k in suite]
    assert "Standart Softmax" in names
    assert "FlashAttention-2" in names
    assert "Fused FP8 GEMM (70B)" in names


def test_roofline_raporlayici_output():
    """6. RooflineRaporlayici tüm eğrileri ve hızlanma metriklerini üretmelidir."""
    rapor = RooflineRaporlayici.basarim_profili_cikar()
    assert len(rapor["intensities"]) == 200
    assert len(rapor["roof_hbm3"]) == 200
    assert len(rapor["roof_l2"]) == 200
    assert len(rapor["roof_sram"]) == 200
    assert rapor["fuzed_speedup"] > 50.0  # FlashAttn vs Softmax hızlanması


def test_warp_stall_percentages_sum():
    """7. Nsight warp stall dağılımı toplamı %100 olmalıdır."""
    rapor = RooflineRaporlayici.basarim_profili_cikar()
    toplam = sum(rapor["warp_stalls"]["oranlar_yuzde"])
    assert np.isclose(toplam, 100.0)


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. RooflineGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_roofline_paneli.png")
    rapor = RooflineRaporlayici.basarim_profili_cikar()

    RooflineGorsellestirici.teshis_paneli_olustur(
        profil_raporu=rapor,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
