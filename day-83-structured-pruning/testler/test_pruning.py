"""
L1/L2 Norm Tabanlı Yapısal Budama Birim Testleri
------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.model import BudanabilirVisionCNN
from src.budayici import YapisalFiltreBudayici
from src.olcumleyici import PerformansOlcumleyici


def test_filtre_l1_norm_hesaplama():
    conv = nn.Conv2d(2, 3, kernel_size=2, bias=False)
    # 3 filtre, her biri 2x2x2 = 8 eleman
    conv.weight.data.fill_(1.0)
    # 2. filtrenin değerlerini 2.0 yapalım
    conv.weight.data[1].fill_(2.0)

    skorlar = YapisalFiltreBudayici.filtre_norm_hesapla(conv, "L1")
    assert skorlar.shape == (3,)
    assert skorlar[0].item() == 8.0
    assert skorlar[1].item() == 16.0
    assert skorlar[2].item() == 8.0


def test_filtre_l2_norm_hesaplama():
    conv = nn.Conv2d(1, 2, kernel_size=2, bias=False)
    # Filtre 0: 4 tane 3.0 -> sqrt(4 * 9) = sqrt(36) = 6.0
    conv.weight.data[0].fill_(3.0)
    # Filtre 1: 4 tane 4.0 -> sqrt(4 * 16) = sqrt(64) = 8.0
    conv.weight.data[1].fill_(4.0)

    skorlar = YapisalFiltreBudayici.filtre_norm_hesapla(conv, "L2")
    assert pytest.approx(skorlar[0].item(), rel=1e-4) == 6.0
    assert pytest.approx(skorlar[1].item(), rel=1e-4) == 8.0


def test_korunacak_indeksler_secimi():
    skorlar = torch.tensor([1.2, 5.8, 0.4, 3.1, 9.5])
    # %40 budama -> 5 * 0.6 = 3 filtre korunmalı (en büyük 3'ü: 9.5, 5.8, 3.1 -> indeksler: 4, 1, 3 -> sıralı: 1, 3, 4)
    korunan = YapisalFiltreBudayici.korunacak_indeksleri_sec(skorlar, budama_orani=0.4)

    assert len(korunan) == 3
    assert korunan.tolist() == [1, 3, 4]


def test_yapisal_budama_fiziksel_kanal_boyutlari():
    model = BudanabilirVisionCNN(giris_kanali=3, sinif_sayisi=10, kanallar=[32, 64, 128])
    budanmis_model, rapor = YapisalFiltreBudayici.modeli_yapisal_buda(model, budama_orani=0.25, norm_tipi="L1")

    # 32 * 0.75 = 24, 64 * 0.75 = 48, 128 * 0.75 = 96
    assert budanmis_model.kanallar == [24, 48, 96]
    assert budanmis_model.conv1.out_channels == 24
    assert budanmis_model.conv2.in_channels == 24
    assert budanmis_model.conv2.out_channels == 48
    assert budanmis_model.conv3.in_channels == 48
    assert budanmis_model.conv3.out_channels == 96
    assert budanmis_model.fc.in_features == 96


def test_yapisal_budama_ileri_gecis_gecerliligi():
    model = BudanabilirVisionCNN(giris_kanali=3, sinif_sayisi=5, kanallar=[16, 32, 64])
    budanmis_model, _ = YapisalFiltreBudayici.modeli_yapisal_buda(model, budama_orani=0.5, norm_tipi="L1")

    x = torch.randn(4, 3, 32, 32)
    cikis = budanmis_model(x)

    assert cikis.shape == (4, 5)
    assert not torch.isnan(cikis).any()


def test_katman_dikisleri_agirlik_kopyalama():
    model = BudanabilirVisionCNN(giris_kanali=1, sinif_sayisi=2, kanallar=[4, 4, 4])
    budanmis_model, rapor = YapisalFiltreBudayici.modeli_yapisal_buda(model, budama_orani=0.5, norm_tipi="L1")

    k2 = rapor["korunan_indeksler"]["conv2"]
    k1 = rapor["korunan_indeksler"]["conv1"]

    # conv2 ağırlıklarının doğru dilimlendiğini doğrula
    beklenen_w = model.conv2.weight.data[k2, :, :, :][:, k1, :, :]
    fark = torch.abs(budanmis_model.conv2.weight.data - beklenen_w).sum().item()
    assert fark == 0.0


def test_performans_olcumleyici_metrikler():
    model = BudanabilirVisionCNN(giris_kanali=3, sinif_sayisi=10, kanallar=[16, 32, 64])
    params = PerformansOlcumleyici.parametre_sayisi(model)
    boyut_mb = PerformansOlcumleyici.model_boyutu_mb(model)
    gecikme_raporu = PerformansOlcumleyici.cikarim_gecikmesi_ve_fps(model, (1, 3, 32, 32), cihaz="cpu", tekrar=10)

    assert params > 0
    assert boyut_mb > 0.0
    assert "gecikme_ms" in gecikme_raporu and gecikme_raporu["gecikme_ms"] > 0.0
    assert "fps" in gecikme_raporu and gecikme_raporu["fps"] > 0.0


def test_gecersiz_budama_orani_hatasi():
    model = BudanabilirVisionCNN()
    with pytest.raises(AssertionError):
        _ = YapisalFiltreBudayici.modeli_yapisal_buda(model, budama_orani=-0.1)

    with pytest.raises(AssertionError):
        _ = YapisalFiltreBudayici.modeli_yapisal_buda(model, budama_orani=1.0)
