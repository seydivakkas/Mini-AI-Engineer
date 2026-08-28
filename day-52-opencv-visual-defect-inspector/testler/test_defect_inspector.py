"""
Day 52: OpenCV ile Kural Tabanlı Görsel Kusur & Bulanıklık Tespiti Birim Testleri.
"""

import os
import pytest
import cv2
import numpy as np
from src.bulaniklik_analizoru import BulaniklikAnalizoru
from src.kusur_tespit_motoru import MorfolojikKusurDedektoru
from src.gorsellestirici import KusurTeftisGorsellestirici


@pytest.fixture
def ornek_gorseller():
    # Net dokulu görsel
    x = np.linspace(0, 20 * np.pi, 200)
    xx, yy = np.meshgrid(x, x)
    net_gri = (np.sin(xx) * np.cos(yy) * 50.0 + 128.0).astype(np.uint8)
    net_rgb = cv2.cvtColor(net_gri, cv2.COLOR_GRAY2RGB)

    # Bulanık görsel
    bulanık_rgb = cv2.GaussianBlur(net_rgb, (21, 21), 7.0)

    # Kusurlu görsel
    kusurlu_rgb = net_rgb.copy()
    cv2.circle(kusurlu_rgb, (80, 80), 12, (20, 20, 20), -1)  # Koyu leke
    cv2.line(kusurlu_rgb, (140, 30), (180, 170), (255, 255, 255), 3)  # Çizik

    return net_rgb, bulanık_rgb, kusurlu_rgb


def test_laplacian_varyansi_netlik(ornek_gorseller):
    """Net görselin Laplacian varyansının bulanık görselden belirgin yüksek olduğunu test eder."""
    net, bulanik, _ = ornek_gorseller
    res_net = BulaniklikAnalizoru.analiz_et(net)
    res_bulanik = BulaniklikAnalizoru.analiz_et(bulanik)

    assert res_net["laplacian_varyansi"] > res_bulanik["laplacian_varyansi"] * 2.0
    assert res_net["net_mi"] is True
    assert res_bulanik["net_mi"] is False


def test_fft_frekans_spektrumu(ornek_gorseller):
    """Net görselin FFT yüksek frekans enerji oranının (HFR) yüksek olduğunu test eder."""
    net, bulanik, _ = ornek_gorseller
    hfr_net, _ = BulaniklikAnalizoru.fft_frekans_spektrumu_hesapla(cv2.cvtColor(net, cv2.COLOR_RGB2GRAY))
    hfr_bulanik, _ = BulaniklikAnalizoru.fft_frekans_spektrumu_hesapla(cv2.cvtColor(bulanik, cv2.COLOR_RGB2GRAY))

    assert hfr_net > hfr_bulanik


def test_tenengrad_skoru(ornek_gorseller):
    """Tenengrad gradyan odak skorunun bulanıklaşmayla azaldığını test eder."""
    net, bulanik, _ = ornek_gorseller
    t_net = BulaniklikAnalizoru.tenengrad_netlik_skoru(cv2.cvtColor(net, cv2.COLOR_RGB2GRAY))
    t_bulanik = BulaniklikAnalizoru.tenengrad_netlik_skoru(cv2.cvtColor(bulanik, cv2.COLOR_RGB2GRAY))

    assert t_net > t_bulanik


def test_kusur_tespiti_ve_sayisi(ornek_gorseller):
    """Kusurlu görselde eklenen leke ve çizik anomalilerinin tespit edildiğini test eder."""
    _, _, kusurlu = ornek_gorseller
    dedektor = MorfolojikKusurDedektoru(min_kusur_alani=15)
    res = dedektor.kusurlari_tespit_et(kusurlu, kernel_boyutu=9, esik_degeri=35)

    assert res["kusurlu_mu"] is True
    assert res["kusur_sayisi"] >= 2
    assert len(res["kusurlar"]) >= 2


def test_kusursuz_gorsel_kalite_puani(ornek_gorseller):
    """Kusursuz görselin kalite puanının 100 olduğunu ve kusur sayısının 0 olduğunu test eder."""
    net, _, _ = ornek_gorseller
    dedektor = MorfolojikKusurDedektoru(min_kusur_alani=30)
    res = dedektor.kusurlari_tespit_et(net, kernel_boyutu=9, esik_degeri=50)

    assert res["kusurlu_mu"] is False
    assert res["kusur_sayisi"] == 0
    assert res["kalite_puani"] == 100.0


def test_morfolojik_maske_boyutlari(ornek_gorseller):
    """Morfolojik ikili maskenin girdi boyutlarıyla birebir eşleştiğini test eder."""
    _, _, kusurlu = ornek_gorseller
    dedektor = MorfolojikKusurDedektoru()
    res = dedektor.kusurlari_tespit_et(kusurlu)

    assert res["binary_mask"].shape == kusurlu.shape[:2]


def test_gorsellestirici_panel_cizimi(ornek_gorseller, tmp_path):
    """6 panelli teşhis panosunun PNG çıktısını başarıyla ürettiğini test eder."""
    _, _, kusurlu = ornek_gorseller
    bulaniklik = BulaniklikAnalizoru.analiz_et(kusurlu)
    dedektor = MorfolojikKusurDedektoru()
    kusur = dedektor.kusurlari_tespit_et(kusurlu)

    cikis_yolu = str(tmp_path / "test_kusur_paneli.png")
    yol = KusurTeftisGorsellestirici.panel_ciz(
        bulaniklik_sonuc=bulaniklik,
        kusur_sonuc=kusur,
        hedef_path=cikis_yolu
    )

    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
