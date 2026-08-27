"""Renk Uzayları ve Segmentasyon Birim Testleri.

Bu dosya; RGB, HSV, LAB dönüşümlerini, OpenCV Ton aralıklarını,
kırmızı renk maskesi birleşimini ve CIELAB Delta-E renk mesafesini test eder.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.renk_donusturucu import RenkUzayiDonusturucu, RenkSegmentasyoncu
from src.gorsellestirici import RenkUzayiGorsellestirici


def test_bgr_to_rgb_ve_kanallar():
    """BGR'dan RGB'ye dönüşümde mavi ve kırmızı kanallarının yer değiştirdiğini test eder."""
    resim_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    resim_bgr[:, :, 0] = 255  # Saf mavi

    rgb = RenkUzayiDonusturucu.bgr_to_rgb(resim_bgr)
    assert rgb[0, 0, 2] == 255  # RGB'de son kanal mavi olmalı
    assert rgb[0, 0, 0] == 0


def test_hsv_donusum_ve_deger_araliklari():
    """HSV dönüşümünde Ton (Hue) değerinin 0-179 arasında sınırlandığını doğrular."""
    resim_bgr = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
    hsv = RenkUzayiDonusturucu.bgr_to_hsv(resim_bgr)

    assert hsv.shape == resim_bgr.shape
    assert np.max(hsv[:, :, 0]) <= 179
    assert np.min(hsv[:, :, 0]) >= 0


def test_lab_ve_ycrcb_donusumu():
    """CIELAB ve YCrCb dönüşümlerinin başarıyla gerçekleştiğini test eder."""
    resim_bgr = np.random.randint(0, 256, (15, 15, 3), dtype=np.uint8)
    lab = RenkUzayiDonusturucu.bgr_to_lab(resim_bgr)
    ycrcb = RenkUzayiDonusturucu.bgr_to_ycrcb(resim_bgr)

    assert lab.shape == resim_bgr.shape
    assert ycrcb.shape == resim_bgr.shape


def test_kirmizi_renk_maskesi_cift_aralik():
    """Saf kırmızı piksellerin (BGR: 0, 0, 255) çift aralık maskesiyle yakalandığını test eder."""
    kirmizi_gorsel = np.zeros((30, 30, 3), dtype=np.uint8)
    kirmizi_gorsel[:, :, 2] = 255  # Saf Kırmızı

    maske = RenkSegmentasyoncu.kirmizi_renk_maskesi(kirmizi_gorsel)
    # Tüm pikseller 255 (beyaz/seçili) olmalıdır
    assert np.all(maske == 255)


def test_cielab_delta_e_tam_eslesme():
    """Aynı rengin CIELAB Delta-E mesafesinin 0 olduğunu ve maskede çıktığını test eder."""
    hedef_renk = (50, 180, 50)
    gorsel = np.full((20, 20, 3), hedef_renk, dtype=np.uint8)

    maske = RenkSegmentasyoncu.cielab_delta_e_maskesi(gorsel, hedef_renk_bgr=hedef_renk, delta_e_esik=5.0)
    assert np.all(maske == 255)


def test_maske_uygulama_karartma():
    """Maskenin 0 olduğu bölgelerde piksellerin siyaha (0,0,0) dönüştüğünü doğrular."""
    gorsel = np.full((10, 10, 3), 200, dtype=np.uint8)
    maske = np.zeros((10, 10), dtype=np.uint8)
    maske[0:5, 0:5] = 255

    sonuc = RenkSegmentasyoncu.maskeyi_uygula(gorsel, maske)
    assert np.all(sonuc[0:5, 0:5] == 200)
    assert np.all(sonuc[5:, 5:] == 0)


def test_gorsellestirici_png_uretimi(tmp_path):
    """12 panelli görsel analiz çizelgesinin diske geçerli PNG olarak yazıldığını doğrular."""
    bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    gri = np.zeros((20, 20), dtype=np.uint8)
    hedef = tmp_path / "panel_test.png"

    cikti = RenkUzayiGorsellestirici.analiz_paneli_ciz(
        orijinal_bgr=bgr,
        rgb_kanallari=(gri, gri, gri),
        hsv_kanallari=(gri, gri, gri),
        lab_kanallari=(gri, gri, gri),
        maske=gri,
        segmente_bgr=bgr,
        dosya_yolu=hedef
    )
    assert cikti.exists()
    assert cikti.stat().st_size > 0
