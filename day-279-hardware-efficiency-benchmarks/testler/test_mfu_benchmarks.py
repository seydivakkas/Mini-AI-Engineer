"""
PyTest Birim Testleri - Day 279 (FAZ 14): Donanım Verimliliği Başarım Paketi (MFU / HFUS).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mfu_benchmark_motoru import MFUBenchmarkEngine
from src.mfu_profilleyici import MFUProfilleyici
from src.gorsellestirici import MFUGorsellestirici


def test_theoretical_flops_calculation_forward():
    """1. İleri geçişte teorik FLOP hesabı doğru olmalıdır."""
    flops = MFUBenchmarkEngine.calculate_theoretical_flops_per_token(
        num_params=7e9,
        num_layers=32,
        num_heads=32,
        seq_len=2048,
        head_dim=128,
        is_training=False,
    )
    assert flops > 14e9  # En az 2 * 7e9


def test_theoretical_flops_calculation_training():
    """2. Eğitim modunda FLOP sayısı ileri geçişin 3 katı olmalıdır."""
    f_fwd = MFUBenchmarkEngine.calculate_theoretical_flops_per_token(7e9, 32, 32, 2048, 128, False)
    f_train = MFUBenchmarkEngine.calculate_theoretical_flops_per_token(7e9, 32, 32, 2048, 128, True)
    assert np.isclose(f_train, 3.0 * f_fwd)


def test_compute_efficiency_metrics_formula():
    """3. MFU ve HFUS metrik hesaplama formülleri doğru çalışmalıdır."""
    res = MFUBenchmarkEngine.compute_efficiency_metrics(
        tokens_per_second=10.0,
        flops_per_token=1e11,
        actual_hardware_flops_per_token=1.1e11,
        measured_bandwidth_gb_s=1675.0,
        hardware_peak_tflops=1000.0,
        hardware_peak_bandwidth_gb_s=3350.0,
    )
    # Model FLOPs / s = 10 * 1e11 = 1e12 FLOPs = 1 TFLOP
    # Peak = 1000 TFLOP -> MFU = (1 / 1000) * 100 = 0.1%
    assert np.isclose(res["mfu_yuzde"], 0.1)
    assert np.isclose(res["hfus_yuzde"], 0.11)
    assert np.isclose(res["mbu_yuzde"], 50.0)


def test_recomputation_overhead_metric():
    """4. Yeniden hesaplama (recomputation) farkı HFUS ile MFU arasındaki farka eşit olmalıdır."""
    res = MFUBenchmarkEngine.compute_efficiency_metrics(5.0, 1e10, 1.2e10, 1000.0)
    assert np.isclose(res["recomputation_overhead_yuzde"], res["hfus_yuzde"] - res["mfu_yuzde"])


def test_llama_70b_benchmark_comparison():
    """5. LLaMA-70B kıyaslama motoru 3 sistemi eksiksiz değerlendirmelidir."""
    res = MFUBenchmarkEngine.run_llama_70b_benchmark_comparison()
    assert "1. Naive PyTorch Baseline" in res["sistem_sonuclari"]
    assert "2. FlashAttention-2 + Compile" in res["sistem_sonuclari"]
    assert "3. FAZ-14 Fused Custom Suite" in res["sistem_sonuclari"]
    assert res["flops_per_token"] > 140e9


def test_profiler_reports_and_speedup():
    """6. MFUProfilleyici %67.8 MFU ve 2.8x hızlanma raporlamalıdır."""
    profil = MFUProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["mfu_yuzde"]["FAZ14_Fused_Custom_Suite"] == 67.8
    assert kars["mbu_yuzde"]["FAZ14_Fused_Custom_Suite"] == 92.5
    assert profil["hizlanma_orani"] > 2.5


def test_model_size_mfu_scaling():
    """7. Model boyutu büyüdükçe (7B -> 405B) MFU oranı artmalıdır."""
    profil = MFUProfilleyici.basarim_profili_cikar()
    mfu_list = profil["skala"]["faz14_custom_mfu"]
    assert mfu_list[-1] > mfu_list[0]
    assert len(mfu_list) == 4


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. MFUGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_mfu_paneli.png")
    profil = MFUProfilleyici.basarim_profili_cikar()

    MFUGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
