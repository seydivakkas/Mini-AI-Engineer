"""
Model Dayanıklılığı ve Bozulmalar Birim Testleri
------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.bozulma_motoru import GorselBozulmaMotoru
from src.dayaniklilik_olcucu import DayaniklilikOlcucu
from src.model import DayanikliVisionModeli


def test_bozulma_fonksiyonlari_boyut_korunumu():
    x = torch.randn(2, 3, 32, 32)
    bozulmalar = GorselBozulmaMotoru.tum_bozulma_fonksiyonlari()

    for ad, fn in bozulmalar.items():
        out = fn(x, siddet=2)
        assert out.shape == x.shape, f"{ad} boyut korunamadı!"
        assert not torch.isnan(out).any(), f"{ad} NaN üretti!"


def test_gaussian_noise_ve_siddet_olcekleme():
    x = torch.zeros(10, 3, 32, 32)
    out_s1 = GorselBozulmaMotoru.gaussian_noise(x, siddet=1)
    out_s5 = GorselBozulmaMotoru.gaussian_noise(x, siddet=5)

    var_s1 = torch.var(out_s1).item()
    var_s5 = torch.var(out_s5).item()

    assert var_s5 > var_s1


def test_gaussian_blur_ve_motion_blur():
    x = torch.randn(4, 3, 32, 32)
    g_blur = GorselBozulmaMotoru.gaussian_blur(x, siddet=3)
    m_blur = GorselBozulmaMotoru.motion_blur(x, siddet=3)

    assert g_blur.shape == x.shape
    assert m_blur.shape == x.shape
    assert not torch.isnan(g_blur).any()
    assert not torch.isnan(m_blur).any()


def test_parlaklik_ve_kontrast_degisimi():
    x = torch.randn(4, 3, 32, 32)
    x_bright = GorselBozulmaMotoru.parlaklik(x, siddet=4)
    x_contrast = GorselBozulmaMotoru.kontrast(x, siddet=4)

    assert x_bright.mean().item() > x.mean().item()
    assert torch.std(x_contrast).item() < torch.std(x).item()


def test_temiz_dogruluk_olcum():
    model = DayanikliVisionModeli(giris_kanali=3, sinif_sayisi=4, taban_kanal=8)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    acc = DayaniklilikOlcucu.temiz_dogruluk_olc(model, loader, cihaz="cpu")
    assert 0.0 <= acc <= 100.0


def test_kapsamli_stres_testi_metrikler():
    model = DayanikliVisionModeli(giris_kanali=3, sinif_sayisi=3, taban_kanal=8)
    x = torch.randn(6, 3, 32, 32)
    y = torch.randint(0, 3, (6,))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    rapor = DayaniklilikOlcucu.kapsamli_stres_testi(model, loader, cihaz="cpu")

    assert "mce" in rapor and 0.0 <= rapor["mce"] <= 100.0
    assert "rel_mce" in rapor
    assert "macc" in rapor and 0.0 <= rapor["macc"] <= 100.0
    assert len(rapor["siddet_egrisi"]) == 5
    assert len(rapor["bozulma_dogruluklari"]) == 8


def test_model_dayanikli_egitim_gecerliligi():
    model = DayanikliVisionModeli(giris_kanali=3, sinif_sayisi=2, taban_kanal=8)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    losses = DayanikliVisionModeli.egit(model, loader, epok_sayisi=2, lr=1e-3, dayanikli_egitim=True, cihaz="cpu")
    assert len(losses) == 2
    assert all(loss > 0.0 for loss in losses)


def test_gecersiz_siddet_seviyesi_hatasi():
    x = torch.randn(1, 3, 16, 16)
    with pytest.raises(AssertionError):
        _ = GorselBozulmaMotoru.gaussian_noise(x, siddet=0)

    with pytest.raises(AssertionError):
        _ = GorselBozulmaMotoru.gaussian_blur(x, siddet=6)
