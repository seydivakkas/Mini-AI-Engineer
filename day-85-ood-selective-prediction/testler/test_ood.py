"""
Enerji Tabanlı OOD ve Seçici Tahmin Birim Testleri
-------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from src.enerji_ood import EnerjiTabanliOODDedektoru
from src.secmeli_tahminci import SecmeliTahminci
from src.metrikler import OODMetrikleri
from src.model import VisionOODModeli


def test_enerji_skoru_hesaplama():
    z = torch.tensor([[1.0, 2.0, 3.0]])
    # T=1.0 için S = log(exp(1) + exp(2) + exp(3))
    beklenen = torch.logsumexp(z, dim=-1)
    hesaplanan = EnerjiTabanliOODDedektoru.enerji_skoru_hesapla(z, sicaklik=1.0)
    assert torch.allclose(hesaplanan, beklenen)


def test_id_vs_ood_enerji_ayrimi():
    # ID: Belirgin bir sınıf logiti yüksek (ör. [10.0, 0.0, 0.0])
    z_id = torch.tensor([[10.0, 0.0, 0.0]])
    # OOD: Dağılmış, düşük sinyal (ör. [0.5, 0.5, 0.5])
    z_ood = torch.tensor([[0.5, 0.5, 0.5]])

    skor_id = EnerjiTabanliOODDedektoru.enerji_skoru_hesapla(z_id, 1.0).item()
    skor_ood = EnerjiTabanliOODDedektoru.enerji_skoru_hesapla(z_ood, 1.0).item()

    assert skor_id > skor_ood


def test_msp_skoru_hesaplama():
    z = torch.tensor([[2.0, 1.0, 0.0]])
    msp = EnerjiTabanliOODDedektoru.msp_skoru_hesapla(z)
    beklenen = F.softmax(z, dim=-1).max(dim=-1)[0]
    assert torch.allclose(msp, beklenen)
    assert (1.0 / 3.0) <= msp.item() <= 1.0


def test_esik_belirleme_tpr():
    dedektor = EnerjiTabanliOODDedektoru()
    # 100 ID örneği
    id_logits = torch.randn(100, 5) + 3.0
    esik = dedektor.esik_belirle(id_logits, hedef_tpr=0.95)

    skorlar = dedektor.enerji_skoru_hesapla(id_logits).numpy()
    gecen_orani = (skorlar >= esik).mean()
    assert pytest.approx(gecen_orani, rel=0.1) == 0.95


def test_secmeli_tahmin_filtreleme():
    secmeli = SecmeliTahminci(esik_degeri=5.0, skor_tipi="enerji", sicaklik=1.0)
    # 2 örnek: biri yüksek enerjili (ID), biri düşük enerjili (OOD)
    z = torch.tensor([[10.0, 1.0], [1.0, 1.0]])

    rapor = secmeli.secmeli_tahmin_yap(z)
    assert rapor["kabul_maskesi"][0].item() is True
    assert rapor["kabul_maskesi"][1].item() is False
    assert len(rapor["kabul_indeksleri"]) == 1
    assert len(rapor["red_indeksleri"]) == 1


def test_auroc_ve_fpr95_hesaplama():
    id_skorlar = np.random.normal(loc=10.0, scale=1.0, size=100)
    ood_skorlar = np.random.normal(loc=2.0, scale=1.0, size=100)

    metrikler = OODMetrikleri.hesapla_ood_metrikleri(id_skorlar, ood_skorlar)
    assert metrikler["auroc"] > 95.0
    assert metrikler["fpr95"] < 10.0
    assert 0.0 <= metrikler["aupr"] <= 100.0


def test_kapsam_risk_monotonluk():
    logits = torch.randn(50, 4)
    labels = torch.randint(0, 4, (50,))

    rapor = SecmeliTahminci.kapsam_risk_egrisi(logits, labels, "enerji", adim_sayisi=20)
    kapsam = rapor["kapsam"]
    # Eşik arttıkça kapsam monoton azalmalıdır
    for i in range(len(kapsam) - 1):
        assert kapsam[i] >= kapsam[i+1]


def test_vision_ood_model_ileri_gecis():
    model = VisionOODModeli(giris_kanali=3, sinif_sayisi=10, taban_kanal=16)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)
    assert not torch.isnan(out).any()
