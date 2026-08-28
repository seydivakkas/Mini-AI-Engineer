"""
Day 93: Kapsamlı Değerlendirme, Yanlılık ve Model Card Birim Testleri
---------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.model import FinalVisionClassifier
from src.metrik_hesaplayici import MetrikHesaplayici, ModelMetrikleri
from src.yanlilik_denetleyicisi import YanlilikDenetleyicisi, AdillikRaporu
from src.model_card_uretici import ModelCardUretici, ModelMetadata
from src.gorsellestirici import DegerlendirmeGorsellestirici


def test_metrik_hesaplayici_temel():
    """Accuracy, Precision, Recall ve F1 skorlarının matematiksel doğruluğunu test eder."""
    hesaplayici = MetrikHesaplayici(sinif_sayisi=3)
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])  # 5/6 doğru

    metrikler = hesaplayici.hesapla(y_true, y_pred)

    assert pytest.approx(metrikler.dogruluk, abs=1e-3) == 5 / 6
    assert metrikler.toplam_ornek == 6
    assert metrikler.macro_f1 > 0.70
    assert metrikler.weighted_f1 > 0.70


def test_karisiklik_matrisi_ve_sinif_f1():
    """Karışıklık matrisinin boyutunu ve sınıf bazlı F1 skorlarını test eder."""
    hesaplayici = MetrikHesaplayici(sinif_sayisi=4)
    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 0, 1])

    metrikler = hesaplayici.hesapla(y_true, y_pred)
    assert metrikler.karisiklik_matrisi.shape == (4, 4)
    assert np.trace(metrikler.karisiklik_matrisi) == 6
    assert metrikler.dogruluk == 1.0
    assert len(metrikler.sinif_f1_skorlari) == 4


def test_kalibrasyon_ve_ece_hesabi():
    """ECE (Expected Calibration Error) ve Brier skorunun doğruluğunu test eder."""
    hesaplayici = MetrikHesaplayici(sinif_sayisi=2, ece_kutu_sayisi=5)
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    probs = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
    ])

    metrikler = hesaplayici.hesapla(y_true, y_pred, olasiliklar=probs)
    assert metrikler.kalibrasyon.ece_skoru >= 0.0
    assert metrikler.kalibrasyon.brier_skoru >= 0.0
    assert len(metrikler.kalibrasyon.kutu_guvenleri) == 5


def test_yanlilik_denetleyicisi_dilim_analizi():
    """Alt grup dilimleme (data slicing) ve dilim doğruluklarının doğruluğunu test eder."""
    denetleyici = YanlilikDenetleyicisi()
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0])

    dilimler = {
        "Aydınlık": np.array([True, True, True, False, False, False]),
        "Karanlık": np.array([False, False, False, True, True, True]),
    }

    rapor: AdillikRaporu = denetleyici.dilimleri_degerlendir(y_true, y_pred, dilimler)
    assert "Aydınlık" in rapor.dilim_sonuclari
    assert "Karanlık" in rapor.dilim_sonuclari
    assert rapor.dilim_sonuclari["Aydınlık"].dogruluk == 1.0
    assert rapor.dilim_sonuclari["Karanlık"].dogruluk < 1.0


def test_demographic_parity_ve_disparate_impact():
    """Demographic Parity ve Disparate Impact %80 kuralının denetimini test eder."""
    denetleyici = YanlilikDenetleyicisi(adillik_esigi=0.80)
    y_true = np.array([1, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1, 1])

    dilimler = {
        "Grup_A": np.array([True, True, True, False, False, False]),
        "Grup_B": np.array([False, False, False, True, True, True]),
    }

    rapor = denetleyici.dilimleri_degerlendir(y_true, y_pred, dilimler)
    assert pytest.approx(rapor.disparate_impact_orani, abs=1e-3) == 1.0
    assert rapor.adillik_esigi_gecti_mi is True


def test_model_cikarim_ve_olasiliklar():
    """FinalVisionClassifier modelinin tensör çıkarımını ve tahmin_et metodunu test eder."""
    model = FinalVisionClassifier(giris_kanali=3, sinif_sayisi=5, taban_filtre=16)
    x = torch.randn(4, 3, 32, 32)

    siniflar, olasiliklar = model.tahmin_et(x)
    assert siniflar.shape == (4,)
    assert olasiliklar.shape == (4, 5)
    assert torch.allclose(olasiliklar.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_model_card_dosya_uretimi():
    """ModelCardUretici'nin geçerli bir MODEL_CARD.md belgesi oluşturduğunu test eder."""
    with tempfile.TemporaryDirectory() as gecici_dizin:
        cikti_yolu = os.path.join(gecici_dizin, "MODEL_CARD.md")

        hesaplayici = MetrikHesaplayici(sinif_sayisi=3)
        metrikler = hesaplayici.hesapla(np.array([0, 1, 2]), np.array([0, 1, 2]))

        denetleyici = YanlilikDenetleyicisi()
        adillik = denetleyici.dilimleri_degerlendir(
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            {"TestDilimi": np.array([True, True, True])},
        )

        uretici = ModelCardUretici()
        icerik = uretici.model_card_olustur(metrikler, adillik, cikti_yolu)

        assert os.path.exists(cikti_yolu)
        assert "# 📄 Model Card" in icerik
        assert "## 📊 Nicel Değerlendirme Sonuçları" in icerik
        assert "## ⚖️ Adillik ve Alt Grup Dilim" in icerik


def test_gorsellestirici_pano_uretme():
    """6 panelli teşhis panosunun hatasız oluşturulup diske kaydedildiğini test eder."""
    gorsellestirici = DegerlendirmeGorsellestirici(cizim_boyutu=(12, 8), dpi=100)

    with tempfile.TemporaryDirectory() as gecici_dizin:
        cikti_dosyasi = os.path.join(gecici_dizin, "test_degerlendirme_paneli.png")

        hesaplayici = MetrikHesaplayici(sinif_sayisi=3)
        metrikler = hesaplayici.hesapla(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 1, 2, 0, 1, 2]),
            olasiliklar=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 2),
        )

        denetleyici = YanlilikDenetleyicisi()
        adillik = denetleyici.dilimleri_degerlendir(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 1, 2, 0, 1, 2]),
            {"A": np.array([True, True, True, False, False, False])},
        )

        gorsellestirici.olustur_degerlendirme_paneli(
            metrikler=metrikler,
            adillik_raporu=adillik,
            metadata=ModelMetadata(),
            kayit_yolu=cikti_dosyasi,
        )

        assert os.path.exists(cikti_dosyasi)
        assert os.path.getsize(cikti_dosyasi) > 1000
