"""
Day 73: SimCLR Temsil Öğrenimi ve NT-Xent Kaybı Birim Testleri
--------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.artirma_politikasi import TensorSimCLRArtirici
from src.simclr_model import TemelKodlayici, ProjeksiyonKafasi, SimCLRModeli
from src.nt_xent_loss import NTXentLoss
from src.egitim_dongusu import SimCLREgitimMotoru
from src.gorsellestirici import SimCLRGorsellestirici


def test_artirma_politikasi_cift_uretimi():
    artirici = TensorSimCLRArtirici(goruntu_boyutu=32)
    x = torch.rand(4, 3, 32, 32)
    v1, v2 = artirici.cift_uret(x)
    
    assert v1.shape == (4, 3, 32, 32)
    assert v2.shape == (4, 3, 32, 32)
    # İki görünüm stokastik olmalı (tamamen aynı olmamalı)
    assert not torch.allclose(v1, v2)


def test_temel_kodlayici_forward():
    kodlayici = TemelKodlayici(giris_kanali=3, temsil_boyutu=64)
    x = torch.randn(4, 3, 32, 32)
    h = kodlayici(x)
    
    assert h.shape == (4, 64)


def test_projeksiyon_kafasi_l2_norm():
    kafa = ProjeksiyonKafasi(temsil_boyutu=64, projeksiyon_boyutu=32)
    h = torch.randn(4, 64)
    z = kafa(h)
    
    assert z.shape == (4, 32)
    # L2 normalizasyonu kontrolü (norm ~ 1.0)
    normlar = torch.norm(z, p=2, dim=1)
    assert torch.allclose(normlar, torch.ones(4), atol=1e-4)


def test_simclr_model_forward():
    model = SimCLRModeli(giris_kanali=3, temsil_boyutu=64, projeksiyon_boyutu=32)
    x = torch.randn(4, 3, 32, 32)
    h, z = model(x)
    
    assert h.shape == (4, 64)
    assert z.shape == (4, 32)
    
    # Sadece temsil çıkarma testi
    h_tek = model.temsil_cikar(x)
    assert h_tek.shape == (4, 64)


def test_nt_xent_loss_hesaplama():
    loss_fn = NTXentLoss(sicaklik=0.5)
    
    # 4 örnekli pozitif çiftler
    z_i = F_norm = torch.nn.functional.normalize(torch.randn(4, 32), p=2, dim=1)
    z_j = F_norm = torch.nn.functional.normalize(torch.randn(4, 32), p=2, dim=1)
    
    kayip = loss_fn(z_i, z_j)
    assert isinstance(kayip, torch.Tensor)
    assert kayip.item() > 0.0
    
    # Geçersiz sıcaklık parametresi kontrolü
    with pytest.raises(ValueError):
        NTXentLoss(sicaklik=-0.1)


def test_hizalama_ve_duzenlilik_metrikleri():
    loss_fn = NTXentLoss(sicaklik=0.5)
    z_i = torch.nn.functional.normalize(torch.randn(8, 32), p=2, dim=1)
    z_j = torch.nn.functional.normalize(torch.randn(8, 32), p=2, dim=1)
    
    metrikler = loss_fn.hesapla_hizalama_ve_duzenlilik(z_i, z_j)
    assert "alignment_loss" in metrikler
    assert "uniformity_loss" in metrikler
    assert "pozitif_kosinus_ort" in metrikler
    assert "negatif_kosinus_ort" in metrikler
    assert "kosinus_marjini" in metrikler


def test_egitim_motoru_bir_epoch():
    model = SimCLRModeli(giris_kanali=3, temsil_boyutu=32, projeksiyon_boyutu=16)
    motor = SimCLREgitimMotoru(model=model, sicaklik=0.5, ogrenme_orani=1e-3, toplam_epoch=2)
    
    # Sentetik veri yükleyici
    v1 = torch.rand(16, 3, 32, 32)
    v2 = torch.rand(16, 3, 32, 32)
    dataset = TensorDataset(v1, v2)
    loader = DataLoader(dataset, batch_size=8)
    
    sonuclar = motor.bir_epoch_egit(loader)
    assert "loss" in sonuclar
    assert "alignment_loss" in sonuclar
    assert sonuclar["loss"] > 0.0


def test_gorsellestirici_pano_kayit(tmp_path):
    gorsellestirici = SimCLRGorsellestirici()
    
    ornek_ciftler = [
        (np.random.rand(32, 32, 3), np.random.rand(32, 32, 3)),
        (np.random.rand(32, 32, 3), np.random.rand(32, 32, 3))
    ]
    egitim_gecmisi = {
        "epoch": [1, 2, 3],
        "loss": [2.5, 2.1, 1.8],
        "lr": [1e-3, 5e-4, 1e-4],
        "alignment_loss": [0.8, 0.5, 0.3],
        "uniformity_loss": [-1.2, -1.5, -1.8],
        "pozitif_kosinus": [0.4, 0.6, 0.75],
        "negatif_kosinus": [0.1, 0.05, 0.01],
        "kosinus_marjini": [0.3, 0.55, 0.74]
    }
    temsiller_2d = np.random.randn(20, 2)
    etiketler = np.random.randint(0, 3, size=20)
    
    kayit_yolu = str(tmp_path / "test_simclr_pano.png")
    kaydedilen = gorsellestirici.olustur_teshis_paneli(
        ornek_ciftler=ornek_ciftler,
        egitim_gecmisi=egitim_gecmisi,
        temsiller_2d=temsiller_2d,
        etiketler=etiketler,
        kayit_yolu=kayit_yolu
    )
    assert os.path.exists(kaydedilen)
    assert os.path.getsize(kaydedilen) > 10000
