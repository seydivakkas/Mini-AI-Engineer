"""
Day 43: Veri Kayması (Data Drift) Tespiti ve KS-Testi Birim Testleri.
"""

import os
import pytest
import numpy as np
from src.dagilim_olcer import KSVeWassersteinHesaplayici
from src.kayma_tespitci import VeriKaymasiDedektoru
from src.gorsellestirici import VeriKaymasiGorsellestirici


def test_ayni_dagilim_drift_yok():
    """Aynı dağılımdan gelen iki örneklemde drift olmadığını test eder."""
    np.random.seed(42)
    ref = np.random.normal(10.0, 2.0, size=1000)
    prod = np.random.normal(10.0, 2.0, size=500)

    sonuc = KSVeWassersteinHesaplayici.olc(ref, prod, alpha=0.05)
    assert sonuc["p_degeri"] >= 0.05
    assert sonuc["drift_tespit_edildi"] is False
    assert sonuc["kayma_derecesi"] == "KAYMA_YOK_STABIL"
    assert sonuc["psi_skoru"] < 0.10


def test_belirgin_kayma_tespiti():
    """Ortalaması kayan dağılımın KS testiyle yakalandığını test eder."""
    np.random.seed(42)
    ref = np.random.normal(10.0, 1.0, size=1000)
    prod = np.random.normal(14.0, 1.0, size=500)  # +4.0 belirgin kayma

    sonuc = KSVeWassersteinHesaplayici.olc(ref, prod, alpha=0.05)
    assert sonuc["drift_tespit_edildi"] is True
    assert sonuc["p_degeri"] < 0.001
    assert sonuc["kayma_derecesi"] == "KRITIK_KAYMA_ALARM"
    assert sonuc["ks_istatistigi"] > 0.5


def test_wasserstein_mesafesi_pozitiflik():
    """Wasserstein mesafesinin iki farklı dağılım için kesin pozitif olduğunu test eder."""
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prod = np.array([3.0, 4.0, 5.0, 6.0, 7.0])

    w1 = KSVeWassersteinHesaplayici.olc(ref, prod)["wasserstein_mesafesi"]
    assert w1 == pytest.approx(2.0, abs=0.1)


def test_psi_hesaplama_mantigi():
    """PSI skorunun stabil veride <0.1, kaymış veride >0.25 olduğunu test eder."""
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    prod_ayni = np.random.normal(0, 1, 500)
    prod_kaymis = np.random.normal(2, 1, 500)

    psi_ayni = KSVeWassersteinHesaplayici.psi_hesapla(ref, prod_ayni)
    psi_kaymis = KSVeWassersteinHesaplayici.psi_hesapla(ref, prod_kaymis)

    assert psi_ayni < 0.10
    assert psi_kaymis >= 0.25


def test_coklu_oznitelik_dedektoru_alarm():
    """Çok öznitelikli dedektörün kritik durumda yeniden eğitim alarmı verdiğini test eder."""
    np.random.seed(42)
    ref = {
        "oz_1": np.random.normal(0, 1, 500),
        "oz_2": np.random.normal(5, 2, 500)
    }
    prod = {
        "oz_1": np.random.normal(3, 1, 200),  # Kaymış
        "oz_2": np.random.normal(12, 2, 200)  # Kaymış
    }

    dedektor = VeriKaymasiDedektoru(ref)
    rapor = dedektor.teftis_et(prod)

    assert rapor["alarm_verildi"] is True
    assert rapor["genel_durum"] == "KRITIK_VERI_KAYMASI_ALARM"
    assert "YENIDEN_EGITIM" in rapor["mlops_aksiyonu"]


def test_ampirik_cdf_monotonik():
    """Hesaplanan ampirik CDF'in monoton artan ve [0, 1] aralığında olduğunu test eder."""
    dizi = np.array([5.0, 1.0, 3.0, 8.0, 2.0])
    izgara = np.linspace(0, 10, 50)
    cdf = KSVeWassersteinHesaplayici.ampirik_cdf_hesapla(dizi, izgara)

    assert np.all(np.diff(cdf) >= 0.0)
    assert 0.0 <= np.min(cdf) <= 1.0
    assert np.max(cdf) == 1.0


def test_gorsellestirici_panosu_kayit(tmp_path):
    """6 panelli teşhis panosunun başarıyla PNG ürettiğini test eder."""
    ref = np.random.normal(10, 2, 200)
    prod = np.random.normal(12, 2, 100)
    rapor = VeriKaymasiDedektoru({"f1": ref}).teftis_et({"f1": prod})

    cikis_yolu = str(tmp_path / "test_drift_panel.png")
    yol = VeriKaymasiGorsellestirici.panel_ciz("f1", ref, prod, rapor, hedef_path=cikis_yolu)
    assert os.path.exists(yol)
