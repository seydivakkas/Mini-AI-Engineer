"""
Day 38: Halı Görsel Arama & Çoklu Özellik Füzyonu Birim Testleri.
"""

import os
import pytest
import numpy as np
from PIL import Image
from src.renk_cikarici import RenkOzellikCikarici
from src.doku_cikarici import DokuOzellikCikarici
from src.fuzyon_arama_motoru import CokluOzellikFuzyonAramaMotoru
from src.hali_katalog_verisi import sentetik_katalog_uret, sentetik_hali_deseni_olustur
from src.gorsellestirici import HaliGorselAramaGorsellestirici


def test_renk_cikarici_boyut_ve_norm():
    """Renk vektörünün L2 normunun 1.0 olduğunu doğrular."""
    img = Image.new("RGB", (60, 60), color=(180, 50, 40))
    cikarici = RenkOzellikCikarici(h_bins=8, s_bins=4, v_bins=4)
    sonuc = cikarici.cikar(img)

    vec = sonuc["renk_vektoru"]
    assert len(vec) == (8 * 4 * 4) + 9
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)


def test_doku_cikarici_haralick():
    """GLCM Haralick metriklerinin geçerli aralıklarda olduğunu doğrular."""
    img = Image.new("RGB", (60, 60), color=(100, 100, 100))
    cikarici = DokuOzellikCikarici(gri_seviye_sayisi=16)
    sonuc = cikarici.cikar(img)

    h_ist = sonuc["haralick_ortalama"]
    assert h_ist["kontrast"] >= 0.0
    assert 0.0 <= h_ist["homojenlik"] <= 1.0
    assert 0.0 <= h_ist["enerji"] <= 1.0


def test_doku_cikarici_lbp_toplam():
    """LBP mikro-doku histogramının toplamının 1.0 olduğunu doğrular."""
    arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    cikarici = DokuOzellikCikarici(lbp_bins=16)
    sonuc = cikarici.cikar(img)
    lbp_hist = sonuc["lbp_histogram"]
    assert np.sum(lbp_hist) == pytest.approx(1.0, abs=1e-4)


def test_katalog_indeksleme():
    """Katalogdaki tüm halıların başarıyla indekslendiğini test eder."""
    katalog = sentetik_katalog_uret()
    motor = CokluOzellikFuzyonAramaMotoru()
    motor.katalog_indeksle(katalog)

    assert len(motor.indeks) == len(katalog)
    assert "renk_vektoru" in motor.indeks[0]
    assert "doku_vektoru" in motor.indeks[0]


def test_gorsel_arama_ayni_gorsel():
    """Birebir aynı görselle arama yapıldığında Top-1 eşleşmenin %100'e yakın çıktığını test eder."""
    katalog = sentetik_katalog_uret()
    motor = CokluOzellikFuzyonAramaMotoru(renk_agirligi=0.5, doku_agirligi=0.5)
    motor.katalog_indeksle(katalog)

    hedef_item = katalog[0]
    sonuc = motor.gorsel_ara(hedef_item["gorsel"], top_k=1)

    top_1 = sonuc["sonuclar"][0]
    assert top_1["id"] == hedef_item["id"]
    assert top_1["hibrit_skor"] >= 99.0


def test_agirlik_modifikasyonu():
    """Renk ağırlığı %100 ve Doku ağırlığı %100 olduğunda skorların değiştiğini test eder."""
    katalog = sentetik_katalog_uret()
    motor = CokluOzellikFuzyonAramaMotoru()
    motor.katalog_indeksle(katalog)

    sorgu = sentetik_hali_deseni_olustur("CARPET-MODERN-02")
    sonuc_renk = motor.gorsel_ara(sorgu, top_k=2, ozel_renk_agirligi=1.0)
    sonuc_doku = motor.gorsel_ara(sorgu, top_k=2, ozel_renk_agirligi=0.0)

    # Renk skoru ile hibrit skor birebir eşit olmalıdır
    assert sonuc_renk["sonuclar"][0]["hibrit_skor"] == pytest.approx(sonuc_renk["sonuclar"][0]["renk_skor"], abs=0.01)
    # Doku skoru ile hibrit skor birebir eşit olmalıdır
    assert sonuc_doku["sonuclar"][0]["hibrit_skor"] == pytest.approx(sonuc_doku["sonuclar"][0]["doku_skor"], abs=0.01)


def test_gorsel_arama_panosu_cizimi(tmp_path):
    """6 panelli görselleştiricinin çizim oluşturmasını test eder."""
    katalog = sentetik_katalog_uret()
    motor = CokluOzellikFuzyonAramaMotoru()
    motor.katalog_indeksle(katalog)

    sorgu = sentetik_hali_deseni_olustur("CARPET-CLASSIC-01")
    arama_sonucu = motor.gorsel_ara(sorgu, top_k=3)

    cikis_path = str(tmp_path / "test_gorsel_arama_panel.png")
    yol = HaliGorselAramaGorsellestirici.arama_paneli_ciz(sorgu, arama_sonucu, hedef_path=cikis_path)

    assert os.path.exists(yol)
