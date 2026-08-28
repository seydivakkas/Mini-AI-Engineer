"""
Day 37: Halı/Tekstil Renk Ayrıştırma & İplik Analizi Birim Testleri.
"""

import os
import pytest
import numpy as np
from PIL import Image
from src.renk_donusturucu import rgb_to_lab, lab_to_rgb
from src.delta_e_hesaplayici import delta_e_2000
from src.iplik_kumeleyici import IplikRenkKumeleyici
from src.katalog_esleyici import IplikKatalogEsleyici
from src.gorsellestirici import HaliRenkGorsellestirici


def test_rgb_lab_roundtrip():
    """RGB -> LAB -> RGB tersinirlik (round-trip) testi."""
    orijinal_rgb = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[128, 128, 128], [255, 255, 255], [0, 0, 0]]
    ], dtype=np.uint8)

    lab = rgb_to_lab(orijinal_rgb)
    reconstructed_rgb = lab_to_rgb(lab)

    # Yuvarlama payı: maksimum 1 uint8 fark
    np.testing.assert_allclose(orijinal_rgb, reconstructed_rgb, atol=1.5)


def test_delta_e_2000_zero():
    """Aynı rengin Delta-E 2000 farkının 0.0 olduğunu doğrular."""
    lab = [50.0, 25.0, -15.0]
    fark = delta_e_2000(lab, lab)
    assert fark == pytest.approx(0.0, abs=1e-5)


def test_delta_e_2000_standard_pair():
    """Bilinen iki CIELAB rengi arasındaki Delta-E 2000 değerini test eder."""
    lab1 = [50.0, 2.6772, -79.7751]
    lab2 = [50.0, 0.0000, -82.7485]
    dE = delta_e_2000(lab1, lab2)
    # CIE standart benchmark değeri ~2.04
    assert 1.8 <= dE <= 2.3


def test_iplik_kumeleyici_toplam_yuzde():
    """Çıkarılan iplik yüzdelerinin toplamının %100 olduğunu doğrular."""
    img_data = np.zeros((100, 100, 3), dtype=np.uint8)
    img_data[:50, :50] = [200, 30, 40]    # 25%
    img_data[:50, 50:] = [30, 200, 40]    # 25%
    img_data[50:, :50] = [30, 40, 200]    # 25%
    img_data[50:, 50:] = [220, 220, 220]  # 25%
    img = Image.fromarray(img_data)

    kumeleyici = IplikRenkKumeleyici(k_iplik=4, max_iter=20)
    sonuc = kumeleyici.iplik_renklerini_ayristir(img)

    assert len(sonuc["iplikler"]) == 4
    toplam_yuzde = sum([i["yuzde"] for i in sonuc["iplikler"]])
    assert toplam_yuzde == pytest.approx(100.0, abs=0.1)


def test_katalog_esleyici_mukemmel_eslesme():
    """Katalogdaki renkle birebir eşleşme testi."""
    katalog = [{"kod": "TEST-1", "ad": "Test Kırmızı", "rgb": [200, 30, 40]}]
    esleyici = IplikKatalogEsleyici(ozel_katalog=katalog)

    cikarilan_lab = rgb_to_lab(np.array([[200, 30, 40]]))[0].tolist()
    cikarilan_iplikler = [{
        "iplik_id": "IPLIK-01",
        "yuzde": 100.0,
        "lab": cikarilan_lab,
        "rgb": [200, 30, 40],
        "hex": "#c81e28"
    }]

    rapor = esleyici.esle_ve_raporla(cikarilan_iplikler)
    assert rapor["genel_parti_onayi"] is True
    assert rapor["eslesmeler"][0]["kalite_durumu"] == "MUKEMMEL_UYUM"
    assert rapor["eslesmeler"][0]["delta_e_2000"] < 1.0


def test_katalog_esleyici_parti_farki_red():
    """Tolerans dışı sapan rengin reddedilmesi testi."""
    katalog = [{"kod": "TEST-1", "ad": "Test Beyaz", "rgb": [250, 250, 250]}]
    esleyici = IplikKatalogEsleyici(ozel_katalog=katalog, tolerans_esigi=3.0)

    cikarilan_lab = rgb_to_lab(np.array([[10, 10, 10]]))[0].tolist()  # Simsiyah
    cikarilan_iplikler = [{
        "iplik_id": "IPLIK-01",
        "yuzde": 100.0,
        "lab": cikarilan_lab,
        "rgb": [10, 10, 10],
        "hex": "#0a0a0a"
    }]

    rapor = esleyici.esle_ve_raporla(cikarilan_iplikler)
    assert rapor["genel_parti_onayi"] is False
    assert rapor["eslesmeler"][0]["kalite_durumu"] == "PARTI_FARKI_RED"
    assert rapor["eslesmeler"][0]["delta_e_2000"] > 30.0


def test_hali_renk_gorsellestirici(tmp_path):
    """6 panelli teşhis panosu görselleştirici testi."""
    hali_rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    kuantize_rgb = np.zeros((60, 60, 3), dtype=np.uint8)

    kumeleme_mock = {
        "iplikler": [
            {"iplik_id": "IPLIK-01", "yuzde": 60.0, "rgb": [150, 30, 40], "lab": [30, 40, 20]},
            {"iplik_id": "IPLIK-02", "yuzde": 40.0, "rgb": [30, 150, 40], "lab": [40, -30, 20]}
        ]
    }
    esleme_mock = {
        "eslesmeler": [
            {
                "iplik_id": "IPLIK-01", "iplik_yuzdesi": 60.0, "cikarilan_rgb": [150, 30, 40],
                "cikarilan_lab": [30, 40, 20], "katalog_ad": "Bordo", "katalog_kod": "Y-1",
                "katalog_rgb": [148, 28, 38], "katalog_lab": [29, 39, 19], "delta_e_2000": 1.2
            }
        ]
    }

    cikis_path = str(tmp_path / "test_hali_panel.png")
    yol = HaliRenkGorsellestirici.hali_renk_paneli_ciz(
        hali_rgb, kuantize_rgb, kumeleme_mock, esleme_mock, hedef_path=cikis_path
    )
    assert os.path.exists(yol)
