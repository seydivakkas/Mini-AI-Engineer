"""
PyTest Birim Testleri - Day 276 (FAZ 14): Dinamik Aktivasyon FP8 Kuantizasyonu.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fp8_dinamik_motoru import FP8DynamicQuantEngine
from src.fp8_dinamik_profilleyici import FP8DinamikProfilleyici
from src.gorsellestirici import FP8DinamikGorsellestirici


def test_fp8_dynamic_quant_shapes_and_bounds():
    """1. Dinamik FP8 kuantizasyonu E4M3 sınırları içinde (-448, 448) kalmalıdır."""
    x = np.random.randn(8, 64).astype(np.float32) * 20.0
    q_x, scales = FP8DynamicQuantEngine.quantize_dynamic_per_token(x, fp8_format="E4M3")
    
    assert q_x.shape == (8, 64)
    assert scales.shape == (8, 1)
    assert np.all(q_x >= -448.0) and np.all(q_x <= 448.0)
    assert np.all(scales > 0.0)


def test_fp8_dequantize_reconstruction():
    """2. Normal aktivasyonlarda dinamik FP8 rekonstrüksiyon hatası düşük olmalıdır (SNR > 30 dB)."""
    x = np.random.randn(16, 128).astype(np.float32)
    q_x, scales = FP8DynamicQuantEngine.quantize_dynamic_per_token(x)
    x_rec = FP8DynamicQuantEngine.dequantize(q_x, scales)
    
    snr = 10 * np.log10(np.mean(x**2) / (np.mean((x - x_rec)**2) + 1e-12))
    assert snr > 30.0


def test_fp8_e5m2_quantization_limits():
    """3. E5M2 formatı 57344.0 maksimum değer sınırını doğru kullanmalıdır."""
    x = np.array([[1000.0, -5000.0, 50000.0]], dtype=np.float32)
    q_x, scales = FP8DynamicQuantEngine.quantize_dynamic_per_token(x, fp8_format="E5M2")
    assert np.all(q_x <= 57344.0) and np.all(q_x >= -57344.0)


def test_outlier_resilience_superiority():
    """4. 50 sigma aykırı değer altında dinamik FP8 statik FP8'den çok daha düşük hata vermelidir."""
    res = FP8DynamicQuantEngine.execute_outlier_resilience_test(batch_size=8, hidden_dim=512, outlier_magnitude=50.0)
    assert res["outlier_korumasi"] is True
    assert res["dinamik_mse"] < res["statik_mse"]
    assert res["hata_azalma_orani"] > 1.5


def test_fused_dynamic_fp8_gemm_math():
    """5. Fused Dinamik FP8 GEMM çıktısı yüksek sinyal-gürültü oranına (SNR > 25 dB) sahip olmalıdır."""
    x = np.random.randn(16, 64).astype(np.float32)
    w = np.random.randn(64, 32).astype(np.float32)
    out, stats = FP8DynamicQuantEngine.fused_dynamic_fp8_gemm(x, w)
    
    assert out.shape == (16, 32)
    assert stats["snr_db"] > 25.0
    assert not np.isnan(out).any()


def test_profiler_output_and_speedup():
    """6. FP8DinamikProfilleyici 1.96x H100 GEMM hızlanması ve 3.14 perplexity raporlamalıdır."""
    profil = FP8DinamikProfilleyici.basarim_profili_cikar()
    karsilastirma = profil["karsilastirma"]
    assert karsilastirma["model_perplexity_wikitext"]["Dinamik_FP8_PerToken"] == 3.14
    assert karsilastirma["gemm_throughput_tflops"]["Dinamik_FP8_PerToken"] == 1920.0
    assert profil["hizlanma_orani"] > 1.8


def test_static_vs_dynamic_quantization():
    """7. Beklenmedik büyük aktivasyonda statik kuantizasyon aşırı kırpma (clipping) yapmalıdır."""
    x = np.array([[100.0, 1.0, 2.0]], dtype=np.float32)
    # 3.5'e göre kalibre edilmiş statik skala
    fixed_s = 3.5 / 448.0
    q_static = FP8DynamicQuantEngine.quantize_static(x, fixed_scale=fixed_s)
    # Statik kırpılmış değer 3.5'i geçemez
    assert np.max(q_static) <= 3.5001
    
    # Dinamik ölçek 100.0'ı tam saklar
    q_dyn, s_dyn = FP8DynamicQuantEngine.quantize_dynamic_per_token(x)
    rec_dyn = FP8DynamicQuantEngine.dequantize(q_dyn, s_dyn)
    assert np.isclose(rec_dyn[0, 0], 100.0, atol=0.5)


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. FP8DinamikGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_fp8_paneli.png")
    profil = FP8DinamikProfilleyici.basarim_profili_cikar()

    FP8DinamikGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
