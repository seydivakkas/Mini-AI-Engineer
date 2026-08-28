"""
Day 50: Model Değerlendirme & Eşik Değeri Mühendisliği Birim Testleri.
"""

import os
import pytest
import numpy as np
from src.kalibrasyon_motoru import OlasilikKalibratoru
from src.esik_muhendisi import EsikDegeriMuhendisi
from src.gorsellestirici import EsikMuhendisligiGorsellestirici


@pytest.fixture
def ornek_tahminler():
    np.random.seed(42)
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0] * 10)
    y_prob = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.15, 0.25, 0.85, 0.05] * 10)
    return y_true, y_prob


def test_brier_skoru_hesaplama(ornek_tahminler):
    """Brier skorunun hesaplandığını ve 0 ile 1 aralığında olduğunu test eder."""
    y_true, y_prob = ornek_tahminler
    res = OlasilikKalibratoru.kalibrasyon_analizi_yap(y_true, y_prob)
    assert "brier_skoru" in res
    assert 0.0 <= res["brier_skoru"] <= 1.0


def test_ece_hesaplama(ornek_tahminler):
    """Expected Calibration Error (ECE) skorunun doğruluğunu test eder."""
    y_true, y_prob = ornek_tahminler
    res = OlasilikKalibratoru.kalibrasyon_analizi_yap(y_true, y_prob)
    assert "ece_skoru" in res
    assert 0.0 <= res["ece_skoru"] <= 1.0


def test_izotonik_kalibrasyon_etkisi(ornek_tahminler):
    """İzotonik kalibrasyonun olasılıkları geçerli [0, 1] aralığında tuttuğunu test eder."""
    y_true, y_prob = ornek_tahminler
    kalibre = OlasilikKalibratoru.izotonik_kalibre_et(y_true, y_prob, y_prob)
    assert len(kalibre) == len(y_prob)
    assert np.all((kalibre >= 0.0) & (kalibre <= 1.0))


def test_fbeta_skor_siralamasi(ornek_tahminler):
    """F0.5 (Precision) eşiğinin F2 (Recall) eşiğinden daha yüksek olduğunu test eder."""
    y_true, y_prob = ornek_tahminler
    res = EsikDegeriMuhendisi.esik_tarama_analizi(y_true, y_prob)

    assert "optimal_f05_esigi" in res
    assert "optimal_f1_esigi" in res
    assert "optimal_f2_esigi" in res
    assert res["optimal_f05_esigi"] >= res["optimal_f2_esigi"]


def test_maliyet_matrisi_net_kazanc(ornek_tahminler):
    """Maliyet matrisinin net finansal kazanç ve optimal eşik ürettiğini test eder."""
    y_true, y_prob = ornek_tahminler
    maliyet = {"b_tp": 2000.0, "b_tn": 10.0, "c_fp": 50.0, "c_fn": 3000.0}
    res = EsikDegeriMuhendisi.esik_tarama_analizi(y_true, y_prob, maliyet_matrisi=maliyet)

    assert "optimal_finansal_esik" in res
    assert "maksimum_net_kazanc" in res
    assert res["maksimum_net_kazanc"] > 0


def test_dca_net_benefit(ornek_tahminler):
    """Decision Curve Analysis dizisinin doğru uzunlukta olduğunu test eder."""
    y_true, y_prob = ornek_tahminler
    res = EsikDegeriMuhendisi.esik_tarama_analizi(y_true, y_prob)
    assert len(res["dca_net_benefit"]) == len(res["esikler"])


def test_gorsellestirici_panel_cizimi(ornek_tahminler, tmp_path):
    """6 panelli teşhis panosunun PNG çıktısını ürettiğini test eder."""
    y_true, y_prob = ornek_tahminler
    kalibrasyon = OlasilikKalibratoru.kalibrasyon_analizi_yap(y_true, y_prob)
    esik_sonuc = EsikDegeriMuhendisi.esik_tarama_analizi(y_true, y_prob)

    cikis_yolu = str(tmp_path / "test_esik_paneli.png")
    yol = EsikMuhendisligiGorsellestirici.panel_ciz(
        kalibrasyon_sonuc=kalibrasyon,
        esik_sonuc=esik_sonuc,
        hedef_path=cikis_yolu
    )

    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
