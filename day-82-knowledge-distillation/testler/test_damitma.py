"""
Knowledge Distillation Birim Testleri
-------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.kayip_damitma import BilgiDamitmaKaybi
from src.modeller import DerinKonvolusyonelOgretmen, KompaktOgrenciModeli
from src.damitici_motor import BilgiDamiticiMotor


def test_bilgi_damitma_kaybi_alfa_ve_sicaklik():
    # alfa = 0.0 iken sadece CE kaybı olmalı
    kayip_fn_ce = BilgiDamitmaKaybi(sicaklik=4.0, alfa=0.0)
    z_s = torch.randn(4, 5)
    z_t = torch.randn(4, 5)
    y = torch.tensor([0, 1, 2, 3])

    loss, metrikler = kayip_fn_ce(z_s, z_t, y)
    beklenen_ce = F.cross_entropy(z_s, y)
    assert pytest.approx(loss.item(), rel=1e-4) == beklenen_ce.item()


def test_sicaklik_olcekleme_ve_tau_kare_katsayisi():
    # τ değişse de kayıp pozitif ve sonlu olmalıdır
    z_s = torch.randn(4, 10)
    z_t = torch.randn(4, 10)
    y = torch.tensor([1, 2, 3, 4])

    for tau in [1.0, 4.0, 10.0]:
        kayip_fn = BilgiDamitmaKaybi(sicaklik=tau, alfa=0.5)
        loss, _ = kayip_fn(z_s, z_t, y)
        assert loss.item() > 0.0
        assert torch.isfinite(loss)


def test_modeller_ileri_gecis_boyutlari():
    ogretmen = DerinKonvolusyonelOgretmen(giris_kanali=3, sinif_sayisi=10, taban_kanal=16)
    ogrenci = KompaktOgrenciModeli(giris_kanali=3, sinif_sayisi=10, taban_kanal=8)

    x = torch.randn(2, 3, 32, 32)
    out_t = ogretmen(x)
    out_s = ogrenci(x)

    assert out_t.shape == (2, 10)
    assert out_s.shape == (2, 10)


def test_ogretmen_parametre_dondurma():
    ogretmen = DerinKonvolusyonelOgretmen(sinif_sayisi=5, taban_kanal=8)
    ogrenci = KompaktOgrenciModeli(sinif_sayisi=5, taban_kanal=8)

    _ = BilgiDamiticiMotor(ogrenci_modeli=ogrenci, ogretmen_modeli=ogretmen)

    # Öğretmenin tüm parametreleri requires_grad = False olmalı
    for p in ogretmen.parameters():
        assert p.requires_grad is False


def test_ogrenci_gradyan_akisi():
    ogretmen = DerinKonvolusyonelOgretmen(sinif_sayisi=5, taban_kanal=8)
    ogrenci = KompaktOgrenciModeli(sinif_sayisi=5, taban_kanal=8)

    kayip_fn = BilgiDamitmaKaybi(sicaklik=3.0, alfa=0.7)

    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 2])

    with torch.no_grad():
        z_t = ogretmen(x)
    z_s = ogrenci(x)

    loss, _ = kayip_fn(z_s, z_t, y)
    loss.backward()

    # Öğrenci ağırlıkları gradyan almalı
    assert ogrenci.kafa.weight.grad is not None
    assert ogretmen.siniflandirici[0].weight.grad is None


def test_gecersiz_alfa_ve_sicaklik_hatalari():
    with pytest.raises(AssertionError):
        _ = BilgiDamitmaKaybi(sicaklik=-1.0)

    with pytest.raises(AssertionError):
        _ = BilgiDamitmaKaybi(alfa=1.5)


def test_damitici_motor_tek_epok_egitim():
    ogrenci = KompaktOgrenciModeli(sinif_sayisi=3, taban_kanal=8)
    ogretmen = DerinKonvolusyonelOgretmen(sinif_sayisi=3, taban_kanal=8)

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 3, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)

    motor = BilgiDamiticiMotor(ogrenci, ogretmen, ogrenme_orani=1e-3)
    loss, ce_loss, kl_loss, tr_acc = motor.egitim_adimi(loader)

    assert isinstance(loss, float) and loss > 0
    assert 0.0 <= tr_acc <= 100.0

    val_acc = motor.dogrulama_adimi(loader)
    assert 0.0 <= val_acc <= 100.0


def test_dark_knowledge_olasılık_dagilimi():
    # Yüksek sıcaklıkta entropi artmalı (olasılıklar yumuşamalı)
    logits = torch.tensor([10.0, 2.0, 1.0, 0.0])

    p_low = F.softmax(logits / 1.0, dim=-1)
    p_high = F.softmax(logits / 10.0, dim=-1)

    # Düşük sıcaklıkta baskın sınıf %99+ iken, yüksek sıcaklıkta diğer sınıflar da anlamlı olasılık alır
    assert p_low[0].item() > p_high[0].item()
    assert p_high[1].item() > p_low[1].item()
