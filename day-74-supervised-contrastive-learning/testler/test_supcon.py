"""
Supervised Contrastive Learning (SupCon) Birim Test Paketi
----------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.artirma_politikasi import TensorSupConArtirici
from src.supcon_model import TemelKodlayici, ProjeksiyonKafasi, DogrusalSiniflandirici, SupConModeli
from src.supcon_loss import SupConLoss
from src.egitim_motoru import SupConEgitimMotoru
from src.gorsellestirici import SupConGorsellestirici


def test_artirici_cift_cikti_boyutu():
    artirici = TensorSupConArtirici(goruntu_boyutu=32)
    x = torch.rand(4, 3, 32, 32)
    v1, v2 = artirici.cift_uret(x)
    
    assert v1.shape == (4, 3, 32, 32)
    assert v2.shape == (4, 3, 32, 32)
    assert not torch.equal(v1, v2)
    assert (v1 >= 0.0).all() and (v1 <= 1.0).all()


def test_temel_kodlayici_boyutlar():
    kodlayici = TemelKodlayici(giris_kanali=3, temsil_boyutu=128)
    x = torch.randn(4, 3, 32, 32)
    h = kodlayici(x)
    
    assert h.shape == (4, 128)


def test_projeksiyon_kafasi_l2_norm():
    projeksiyon = ProjeksiyonKafasi(temsil_boyutu=128, projeksiyon_boyutu=64)
    h = torch.randn(4, 128)
    z = projeksiyon(h)
    
    assert z.shape == (4, 64)
    normlar = torch.norm(z, p=2, dim=1)
    assert torch.allclose(normlar, torch.ones_like(normlar), atol=1e-5)


def test_supcon_model_entegrasyonu():
    model = SupConModeli(giris_kanali=3, temsil_boyutu=128, projeksiyon_boyutu=64, sinif_sayisi=5)
    x = torch.randn(4, 3, 32, 32)
    
    h, z = model(x)
    assert h.shape == (4, 128)
    assert z.shape == (4, 64)
    
    logits = model.siniflandir(x)
    assert logits.shape == (4, 5)


def test_supcon_loss_matematiksel_dogruluk():
    loss_fn = SupConLoss(sicaklik=0.1)
    
    # 4 örnek, her birinin 2 görünümü: (4, 2, 64)
    # Sınıf 0: Örnek 0 ve 1, Sınıf 1: Örnek 2 ve 3
    etiketler = torch.tensor([0, 0, 1, 1])
    
    # Mükemmel ayrışmış temsiller
    z1 = torch.zeros(4, 64)
    z2 = torch.zeros(4, 64)
    z1[0:2, 0] = 1.0
    z1[2:4, 1] = 1.0
    z2[0:2, 0] = 1.0
    z2[2:4, 1] = 1.0
    
    z_cift = torch.stack([z1, z2], dim=1)
    kayip = loss_fn(z_cift, etiketler)
    
    assert kayip.item() >= 0.0
    assert not torch.isnan(kayip)


def test_supcon_loss_geometrik_marjin():
    loss_fn = SupConLoss(sicaklik=0.1)
    etiketler = torch.tensor([0, 0, 1, 1])
    
    z1 = torch.zeros(4, 64)
    z2 = torch.zeros(4, 64)
    z1[0:2, 0] = 1.0
    z1[2:4, 1] = 1.0
    z2[0:2, 0] = 1.0
    z2[2:4, 1] = 1.0
    
    z_cift = torch.stack([z1, z2], dim=1)
    metrikler = loss_fn.hesapla_geometrik_ayrisma(z_cift, etiketler)
    
    assert metrikler["sinif_ici_kosinus"] > 0.99
    assert metrikler["siniflar_arasi_kosinus"] < 0.01
    assert metrikler["ayrisma_marjini"] > 0.98


def test_supcon_egitim_motoru_stage1_ve_stage2():
    model = SupConModeli(giris_kanali=3, temsil_boyutu=32, projeksiyon_boyutu=16, sinif_sayisi=2)
    motor = SupConEgitimMotoru(model=model, sicaklik=0.1, ogrenme_orani=1e-3, cihaz="cpu")
    
    v1 = torch.rand(8, 3, 32, 32)
    v2 = torch.rand(8, 3, 32, 32)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    ds = TensorDataset(v1, v2, y)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    # Stage 1
    s1 = motor.egit_stage1_kontrastif(loader, toplam_epoch=1)
    assert len(s1["loss"]) == 1
    
    # Stage 2
    s2 = motor.egit_stage2_dogrusal_siniflandirici(loader, loader, toplam_epoch=1)
    assert len(s2["loss"]) == 1
    assert s2["dogruluk"][0] >= 0.0


def test_supcon_gorsellestirici_pano_uretimi(tmp_path):
    gorsellestirici = SupConGorsellestirici()
    
    ornek_ciftler = [(np.random.rand(32, 32, 3), np.random.rand(32, 32, 3), 0) for _ in range(4)]
    stage1 = {
        "epoch": [1, 2],
        "loss": [2.5, 2.1],
        "sinif_ici_kosinus": [0.7, 0.9],
        "siniflar_arasi_kosinus": [0.2, 0.1],
        "ayrisma_marjini": [0.5, 0.8]
    }
    stage2 = {
        "epoch": [1, 2],
        "loss": [1.5, 0.8],
        "dogruluk": [75.0, 95.0]
    }
    temsiller_2d = np.random.randn(8, 2)
    etiketler = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    
    kayit = os.path.join(str(tmp_path), "test_supcon.png")
    c = gorsellestirici.olustur_teshis_paneli(ornek_ciftler, stage1, stage2, temsiller_2d, etiketler, kayit)
    
    assert os.path.exists(c)
    assert os.path.getsize(c) > 1000
