"""
Olasılık Kalibrasyonu ve ECE Birim Testleri
-------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.metrikler import KalibrasyonMetrikleri
from src.kalibrator import SicaklikKalibratoru
from src.model import GuvenilmezVisionModeli


def test_ece_mukemmel_kalibre_model():
    # 2 sınıflı, güveni %80 olan ve tam %80 doğru çıkan 100 örnek
    # logitler: log(0.8 / 0.2) = 1.3863
    logitler = torch.tensor([[1.3863, 0.0]] * 100)
    # 80 tanesi 0, 20 tanesi 1
    etiketler = torch.tensor([0] * 80 + [1] * 20)

    metrikler = KalibrasyonMetrikleri.hesapla_tum_metrikler(logitler, etiketler, n_bins=10)
    # ECE neredeyse 0 olmalıdır
    assert metrikler["ece"] < 1.0


def test_ece_asiri_guvenli_model():
    # %100 güvenle 0 diyen ama sadece %50'si doğru olan model
    logitler = torch.tensor([[100.0, 0.0]] * 100)
    etiketler = torch.tensor([0] * 50 + [1] * 50)

    metrikler = KalibrasyonMetrikleri.hesapla_tum_metrikler(logitler, etiketler, n_bins=10)
    # Güven = 1.0, Doğruluk = 0.5 -> Fark = 0.5 -> ECE = %50
    assert pytest.approx(metrikler["ece"], rel=1e-2) == 50.0
    assert pytest.approx(metrikler["mce"], rel=1e-2) == 50.0


def test_temperature_scaling_ileri_gecis():
    kalibrator = SicaklikKalibratoru(baslangic_sicaklik=2.0)
    z = torch.tensor([[4.0, 2.0], [6.0, -2.0]])

    olcekli_z = kalibrator(z)
    beklenen = z / 2.0
    assert torch.allclose(olcekli_z, beklenen)


def test_temperature_scaling_tahmin_sirasi_korunumu():
    # Herhangi bir T > 0 için argmax ve top-1 doğruluk asla değişmemelidir!
    z = torch.randn(50, 10)
    etiketler = torch.randint(0, 10, (50,))

    for t in [0.2, 0.8, 1.5, 3.5, 10.0]:
        kalibrator = SicaklikKalibratoru(baslangic_sicaklik=t)
        olcekli_z = kalibrator(z)

        assert torch.equal(z.argmax(dim=-1), olcekli_z.argmax(dim=-1))


def test_temperature_scaling_lbfgs_optimizasyon():
    # Aşırı güvenli logitler üret
    z = torch.randn(100, 5) * 10.0
    y = torch.randint(0, 5, (100,))

    kalibrator = SicaklikKalibratoru(baslangic_sicaklik=1.0)
    onceki_nll = F.cross_entropy(z, y).item()

    rapor = kalibrator.kalibre_et(z, y, max_iter=30)
    sonraki_nll = F.cross_entropy(kalibrator(z), y).item()

    # NLL düşmüş olmalı ve T > 1.0 olmalı (aşırı güveni yumuşatmak için)
    assert sonraki_nll < onceki_nll
    assert rapor["optimal_sicaklik"] > 1.0


def test_nll_ve_brier_skor_gecerliligi():
    logitler = torch.randn(20, 4)
    etiketler = torch.randint(0, 4, (20,))

    metrikler = KalibrasyonMetrikleri.hesapla_tum_metrikler(logitler, etiketler)
    assert metrikler["nll"] > 0.0
    assert metrikler["brier_score"] >= 0.0
    assert 0.0 <= metrikler["dogruluk"] <= 100.0


def test_logit_toplayici_boyutlar():
    model = GuvenilmezVisionModeli(giris_kanali=3, sinif_sayisi=5, taban_kanal=8)
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 5, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    logits, labels = GuvenilmezVisionModeli.logit_topla(model, loader, cihaz="cpu")
    assert logits.shape == (16, 5)
    assert labels.shape == (16,)


def test_sicaklik_sinirlama_guvenligi():
    # Sıcaklık 0 veya negatif girilse dahi güvenli aralığa clamp edilmeli
    kalibrator = SicaklikKalibratoru(baslangic_sicaklik=-5.0)
    z = torch.tensor([[2.0, 1.0]])
    cikis = kalibrator(z)

    assert not torch.isnan(cikis).any()
    assert not torch.isinf(cikis).any()
