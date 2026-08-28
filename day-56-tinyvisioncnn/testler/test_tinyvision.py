"""
Day 56: Edge Cihazlar İçin Sıfırdan Hafif CNN, Depthwise Separable Conv ve FLOPs Hesabı Birim Testleri.
"""

import os
import pytest
import torch
from src.modeller import DerinlikAyrisimliKonvolusyon, StandartCNN, TinyVisionCNN
from src.profil_motoru import FLOPsProfilMotoru
from src.gorsellestirici import TinyVisionGorsellestirici


@pytest.fixture
def modeller():
    std = StandartCNN(in_channels=3, num_classes=10)
    tiny = TinyVisionCNN(in_channels=3, num_classes=10)
    return std, tiny


def test_derinlik_ayrisimli_konvolusyon_forward():
    """Depthwise Separable Conv bloğunun doğru çıktı boyutunu ürettiğini test eder."""
    blok = DerinlikAyrisimliKonvolusyon(in_channels=16, out_channels=32, stride=2)
    x = torch.randn(2, 16, 32, 32)
    out = blok(x)

    assert out.shape == (2, 32, 16, 16)
    assert not torch.isnan(out).any()


def test_standart_cnn_forward_ve_cikti_boyutu(modeller):
    """Standart CNN modelinin doğru sınıflandırma çıktı tensörü ürettiğini test eder."""
    std, _ = modeller
    x = torch.randn(2, 3, 64, 64)
    logits = std(x)

    assert logits.shape == (2, 10)
    assert not torch.isnan(logits).any()


def test_tinyvision_cnn_forward_ve_cikti_boyutu(modeller):
    """TinyVisionCNN modelinin Global Average Pooling sonrası doğru çıktı ürettiğini test eder."""
    _, tiny = modeller
    x = torch.randn(2, 3, 64, 64)
    logits = tiny(x)

    assert logits.shape == (2, 10)
    assert not torch.isnan(logits).any()


def test_parametre_sayisi_hesaplama(modeller):
    """Parametre profilleme motorunun TinyVisionCNN'i çok daha hafif olarak hesapladığını test eder."""
    std, tiny = modeller
    std_p = FLOPsProfilMotoru.parametre_sayisi_hesapla(std)
    tiny_p = FLOPsProfilMotoru.parametre_sayisi_hesapla(tiny)

    assert std_p["toplam_param"] > 0
    assert tiny_p["toplam_param"] > 0
    assert tiny_p["toplam_param"] < std_p["toplam_param"]


def test_analitik_flops_hesaplama(modeller):
    """Analitik FLOPs hesaplayıcısının katman kancaları ile kesin hesaplama yaptığını test eder."""
    std, tiny = modeller
    girdi_sekli = (1, 3, 64, 64)
    std_f = FLOPsProfilMotoru.analitik_flops_hesapla(std, girdi_sekli)
    tiny_f = FLOPsProfilMotoru.analitik_flops_hesapla(tiny, girdi_sekli)

    assert std_f["toplam_flops"] > 0
    assert tiny_f["toplam_flops"] > 0
    assert tiny_f["toplam_flops"] < std_f["toplam_flops"]
    assert len(tiny_f["katmanlar"]) > 0


def test_karsilastirmali_profil_yapisi(modeller):
    """Kapsamlı profil motorunun tüm metrikleri ve tasarruf çarpanlarını doğru ürettiğini test eder."""
    std, tiny = modeller
    sonuc = FLOPsProfilMotoru.karsilastirmali_profil(std, tiny, (1, 3, 64, 64))

    assert "standart" in sonuc
    assert "tinyvision" in sonuc
    assert "ozet" in sonuc
    assert sonuc["ozet"]["flops_tasarruf_carpani"] > 2.0
    assert sonuc["ozet"]["param_tasarruf_carpani"] > 2.0


def test_gorsellestirici_panel_cizimi(modeller, tmp_path):
    """6 panelli teşhis panosunun başarıyla PNG dosyası ürettiğini test eder."""
    std, tiny = modeller
    profil = FLOPsProfilMotoru.karsilastirmali_profil(std, tiny, (1, 3, 64, 64))

    hedef = str(tmp_path / "test_tinyvision_paneli.png")
    cikis = TinyVisionGorsellestirici.panel_ciz(profil, hedef)

    assert os.path.exists(cikis)
    assert os.path.getsize(cikis) > 1000
