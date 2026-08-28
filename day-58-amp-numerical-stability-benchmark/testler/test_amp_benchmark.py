"""
Day 58: Otomatik Karma Hassasiyet (AMP), FP16 vs BF16, GradScaler ve Sayısal Kararlılık Birim Testleri.
"""

import os
import pytest
import numpy as np
import torch
from src.sayisal_kararlilik import SayisalKararlilikAnalizoru
from src.amp_benchmark_motoru import AMPBenchmarkMotoru, KapsamliGorusModeli
from src.gorsellestirici import AMPGorsellestirici


def test_format_ozellikleri_dogruluk():
    """Kayan nokta formatlarının bit tanımlarının standart IEEE 754 ve BF16 ile uyumunu test eder."""
    formatlar = SayisalKararlilikAnalizoru.format_ozelliklerini_getir()
    assert formatlar["FP32 (Single)"]["toplam_bit"] == 32
    assert formatlar["FP16 (Half)"]["us_biti (exponent)"] == 5
    assert formatlar["BF16 (Brain Float)"]["us_biti (exponent)"] == 8
    assert formatlar["FP16 (Half)"]["grad_scaler_gerekir_mi"] is True
    assert formatlar["BF16 (Brain Float)"]["grad_scaler_gerekir_mi"] is False


def test_gradyan_underflow_simulasyonu():
    """Ham FP16'da underflow oluştuğunu ve GradScaler ile underflow'un sıfırlandığını test eder."""
    sim = SayisalKararlilikAnalizoru.gradyan_kaybi_simulasyonu(ornek_sayisi=10_000)
    assert sim["Ham FP16 (Scalersız)"]["underflow_orani"] > 0.0
    assert sim["AMP-FP16 (GradScaler)"]["underflow_orani"] == 0.0
    assert np.isfinite(sim["BF16 (Bfloat16)"]["ort_relatif_hata"])


def test_kapsamli_gorus_modeli_cikti_boyutu():
    """Modelin (N, 3, 32, 32) girişine karşılık (N, 10) boyutunda logit ürettiğini test eder."""
    model = KapsamliGorusModeli(in_channels=3, num_classes=10)
    dummy_x = torch.randn(4, 3, 32, 32)
    out = model(dummy_x)
    assert out.shape == (4, 10)
    assert torch.isfinite(out).all()


def test_amp_benchmark_fp32():
    """FP32 tam hassasiyet benchmark modunun sorunsuz çalıştığını test eder."""
    motor = AMPBenchmarkMotoru(device="cpu")
    loader = motor.sentetik_veri_olustur(num_samples=64, img_size=16, batch_size=16)
    sonuclar = motor.calistir_kiyaslama(loader, epochs=1, isinma_adimlari=1)

    assert "FP32 (Standart)" in sonuclar
    assert sonuclar["FP32 (Standart)"]["throughput_ornek_s"] > 0.0
    assert np.isfinite(sonuclar["FP32 (Standart)"]["nihai_loss"])


def test_amp_benchmark_fp16_ve_gradscaler():
    """AMP-FP16 modunun çalıştığını ve metrik ürettiğini test eder."""
    motor = AMPBenchmarkMotoru(device="cpu")
    loader = motor.sentetik_veri_olustur(num_samples=64, img_size=16, batch_size=16)
    sonuclar = motor.calistir_kiyaslama(loader, epochs=1, isinma_adimlari=1)

    assert "AMP-FP16 (GradScaler)" in sonuclar
    assert sonuclar["AMP-FP16 (GradScaler)"]["throughput_ornek_s"] > 0.0
    assert len(sonuclar["AMP-FP16 (GradScaler)"]["kayip_gecmisi"]) == 1


def test_amp_benchmark_bf16():
    """BF16 modunun çalıştığını ve sonlu kayıp ürettiğini test eder."""
    motor = AMPBenchmarkMotoru(device="cpu")
    loader = motor.sentetik_veri_olustur(num_samples=64, img_size=16, batch_size=16)
    sonuclar = motor.calistir_kiyaslama(loader, epochs=1, isinma_adimlari=1)

    assert "AMP-BF16" in sonuclar
    assert sonuclar["AMP-BF16"]["throughput_ornek_s"] > 0.0
    assert np.isfinite(sonuclar["AMP-BF16"]["nihai_loss"])


def test_gorsellestirici_panel_cizimi(tmp_path):
    """AMP teşhis panosunun başarıyla PNG dosyası ürettiğini test eder."""
    motor = AMPBenchmarkMotoru(device="cpu")
    loader = motor.sentetik_veri_olustur(num_samples=64, img_size=16, batch_size=16)
    sonuclar = motor.calistir_kiyaslama(loader, epochs=1, isinma_adimlari=1)
    sim = SayisalKararlilikAnalizoru.gradyan_kaybi_simulasyonu(ornek_sayisi=1000)

    hedef = str(tmp_path / "test_amp_paneli.png")
    cikis = AMPGorsellestirici.panel_ciz(sonuclar, sim, hedef_path=hedef)

    assert os.path.exists(cikis)
    assert os.path.getsize(cikis) > 1000
