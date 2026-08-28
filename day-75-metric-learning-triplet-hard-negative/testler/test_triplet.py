"""
Triplet Metric Learning ve Madencilik Stratejileri Birim Test Paketi
---------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.triplet_ag import MetrikOznitelikAgi
from src.mining_motoru import TripletMadencisi
from src.triplet_loss import ModulerTripletMarginLoss
from src.egitim_dongusu import TripletEgitimMotoru
from src.gorsellestirici import TripletGorsellestirici


def test_metrik_ag_l2_normalize_cikti():
    model = MetrikOznitelikAgi(giris_kanali=3, gomulme_boyutu=64)
    x = torch.randn(4, 3, 32, 32)
    e = model(x)
    
    assert e.shape == (4, 64)
    normlar = torch.norm(e, p=2, dim=1)
    assert torch.allclose(normlar, torch.ones_like(normlar), atol=1e-5)


def test_ikili_mesafe_matrisi_simetri_ve_kosegen():
    e = torch.randn(6, 64)
    e = torch.nn.functional.normalize(e, p=2, dim=1)
    
    D = TripletMadencisi.ikili_mesafe_matrisi(e, kareli=False)
    assert D.shape == (6, 6)
    # Köşegen 0 olmalı
    assert torch.allclose(torch.diagonal(D), torch.zeros(6), atol=1e-5)
    # Simetrik olmalı
    assert torch.allclose(D, D.T, atol=1e-5)


def test_batch_hard_mining_en_zor_secim():
    madenci = TripletMadencisi(marjin=0.3)
    
    # 4 örnek: 0,1 -> Sınıf 0; 2,3 -> Sınıf 1
    e = torch.zeros(4, 4)
    e[0, 0] = 1.0 # a
    e[1, 0] = 0.8; e[1, 1] = 0.6 # p (mesafe var)
    e[2, 0] = 0.9; e[2, 2] = 0.435 # n (yakın negatif)
    e[3, 3] = 1.0 # n (uzak negatif)
    e = torch.nn.functional.normalize(e, p=2, dim=1)
    
    y = torch.tensor([0, 0, 1, 1])
    kayip, istatistik = madenci.madencilik_yap(e, y, strateji="batch_hard")
    
    assert kayip.item() >= 0.0
    assert istatistik["toplam_triplet"] > 0
    assert not torch.isnan(kayip)


def test_batch_semi_hard_madencilik():
    madenci = TripletMadencisi(marjin=0.5)
    e = torch.randn(8, 32)
    e = torch.nn.functional.normalize(e, p=2, dim=1)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    kayip, istatistik = madenci.madencilik_yap(e, y, strateji="batch_semi_hard")
    assert kayip.item() >= 0.0
    assert "aktif_triplet_orani" in istatistik
    assert istatistik["toplam_triplet"] > 0


def test_batch_all_madencilik():
    madenci = TripletMadencisi(marjin=0.3)
    e = torch.randn(8, 32)
    e = torch.nn.functional.normalize(e, p=2, dim=1)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    kayip, istatistik = madenci.madencilik_yap(e, y, strateji="batch_all")
    assert kayip.item() >= 0.0
    assert istatistik["toplam_triplet"] > 0


def test_triplet_loss_marjin_dogrulamasi():
    loss_fn = ModulerTripletMarginLoss(marjin=0.3, strateji="batch_all")
    
    # Mükemmel ayrışmış temsiller
    e = torch.zeros(4, 16)
    e[0:2, 0] = 1.0 # Sınıf 0: mesafe 0
    e[2:4, 1] = 1.0 # Sınıf 1: mesafe 0 (aradaki mesafe sqrt(2) ~ 1.41 > 0.3)
    y = torch.tensor([0, 0, 1, 1])
    
    kayip, istatistik = loss_fn(e, y)
    # d(a,p)=0, d(a,n)=1.41 => 0 - 1.41 + 0.3 = -1.11 < 0 => Loss = 0
    assert kayip.item() == 0.0
    assert istatistik["aktif_triplet_orani"] == 0.0
    assert istatistik["kolay_orani"] == 100.0


def test_triplet_egitim_motoru_birim_dongu():
    model = MetrikOznitelikAgi(giris_kanali=3, gomulme_boyutu=16)
    motor = TripletEgitimMotoru(model=model, marjin=0.3, strateji="batch_semi_hard", cihaz="cpu")
    
    x = torch.rand(8, 3, 32, 32)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    gecmis = motor.egit(loader, toplam_epoch=1)
    assert len(gecmis["loss"]) == 1
    assert gecmis["loss"][0] >= 0.0


def test_triplet_gorsellestirici_pano_uretimi(tmp_path):
    gorsellestirici = TripletGorsellestirici()
    gecmis = {
        "epoch": [1, 2],
        "loss": [0.25, 0.12],
        "d_ap": [0.4, 0.2],
        "d_an": [0.8, 1.2],
        "marjin": [0.4, 1.0],
        "aktif_oran": [80.0, 30.0],
        "zor_oran": [20.0, 5.0],
        "yari_zor_oran": [60.0, 25.0],
        "kolay_oran": [20.0, 70.0]
    }
    gomulmeler_2d = np.random.randn(10, 2)
    etiketler = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
    kayit = os.path.join(str(tmp_path), "test_triplet.png")
    
    c = gorsellestirici.olustur_teshis_paneli(gecmis, gomulmeler_2d, etiketler, marjin=0.3, kayit_yolu=kayit)
    assert os.path.exists(c)
    assert os.path.getsize(c) > 1000
