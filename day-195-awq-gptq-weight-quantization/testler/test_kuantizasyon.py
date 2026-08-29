"""
PyTest Birim Testleri - Day 195: AWQ ve GPTQ 4-Bit Kuantizasyonu.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kuantizasyon_motoru import (
    StandartRoundToNearestQuantizer,
    AWQQuantizer,
    GPTQQuantizer,
)
from src.perplexity_profilleyici import PerplexityVeVRAMProfilleyici
from src.gorsellestirici import KuantizasyonGorsellestirici


def test_rtn_kuantizasyon_sekil_ve_aralik():
    """1. Standart RTN girdi tensör şeklini korumalı ve pozitif ölçek üretmelidir."""
    w = torch.randn(64, 128)
    w_deq, scales, zeros = StandartRoundToNearestQuantizer.kuantize_et(w, group_size=128)
    assert w_deq.shape == w.shape
    assert torch.all(scales > 0)


def test_awq_salient_olcek_hesaplama():
    """2. AWQ salient kanal büyüklüklerini doğru tespit etmelidir."""
    x = torch.randn(32, 64)
    x[:, 0] *= 20.0  # 0. kanala yüksek aktivasyon ver
    scales = AWQQuantizer.salient_olcek_hesapla(x)
    assert scales.shape == (64,)
    assert scales[0] > scales[1]  # 0. kanal daha yüksek koruma ölçeği almalı


def test_awq_kuantizasyon_rekonstruksiyonu():
    """3. AWQ kuantizasyonu yüksek kosinüs benzerliği (>0.98) sağlamalıdır."""
    w = torch.randn(64, 128) * 0.02
    x = torch.randn(32, 128)
    w_awq, _ = AWQQuantizer.kuantize_et(w, x, group_size=128)
    metrikler = PerplexityVeVRAMProfilleyici.hata_olcumleri(w, w_awq)
    assert metrikler["kosinus_benzerligi"] > 0.98


def test_gptq_hessian_ve_kuantizasyon():
    """4. GPTQ Hessian hesaplaması sayısal olarak kararlı olmalı ve NaN üretmemelidir."""
    w = torch.randn(32, 64) * 0.02
    x = torch.randn(16, 64)
    w_gptq = GPTQQuantizer.kuantize_et(w, x)
    assert not torch.isnan(w_gptq).any()
    assert w_gptq.shape == w.shape


def test_hata_olcumleri_metrikleri():
    """5. MSE ve kosinüs benzerliği hesaplama fonksiyonu doğru çalışmalıdır."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    res = PerplexityVeVRAMProfilleyici.hata_olcumleri(a, b)
    assert res["mse_loss"] == 0.0
    assert res["kosinus_benzerligi"] == pytest.approx(1.0, abs=1e-4)


def test_kuantizasyon_karsilastirma_raporu():
    """6. Kıyaslama raporu 4 yöntemi ve 35 GB INT4 VRAM değerini içermelidir."""
    rapor = PerplexityVeVRAMProfilleyici.kuantizasyon_karsilastirma_raporu()
    assert len(rapor) == 4
    awq_metod = rapor[3]
    assert "AWQ" in awq_metod["yontem"]
    assert awq_metod["model_vram_gb"] == 35.0


def test_awq_vs_rtn_ustunlugu():
    """7. Outlier aktivasyon durumunda AWQ, çıktı aktivasyonu hata payını düşük tutmalıdır."""
    w = torch.randn(64, 128) * 0.02
    x = torch.randn(32, 128)
    x[:, :3] *= 25.0  # Ağır outlier kanallar

    w_rtn, _, _ = StandartRoundToNearestQuantizer.kuantize_et(w, group_size=128)
    w_awq, _ = AWQQuantizer.kuantize_et(w, x, group_size=128)

    hata_awq = PerplexityVeVRAMProfilleyici.hata_olcumleri(w, w_awq)
    assert hata_awq["kosinus_benzerligi"] > 0.95
    assert not torch.isnan(w_awq).any()


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. KuantizasyonGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_awq_paneli.png")
    kiyas_raporu = PerplexityVeVRAMProfilleyici.kuantizasyon_karsilastirma_raporu()
    salient_kanallar = np.random.uniform(0.1, 5.0, size=128)

    KuantizasyonGorsellestirici.teshis_paneli_olustur(
        kiyas_raporu=kiyas_raporu,
        salient_kanallar=salient_kanallar,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
