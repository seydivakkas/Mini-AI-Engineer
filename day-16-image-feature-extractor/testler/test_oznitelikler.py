"""Geleneksel Görsel Öznitelik Çıkarıcı Birim Testleri.

Bu dosya; SIFT, ORB, HOG ve LBP fonksiyonlarının çıktı boyutlarını, veri tiplerini,
hata yakalama mekanizmalarını ve çizelge oluşturma işlevlerini test eder.
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

from src.oznitelik_cikarici import GorselOznitelikCikarici
from src.gorsellestirici import OznitelikGorsellestirici


def test_sift_cikar_boyut_ve_tip():
    """SIFT tanımlayıcılarının float32 ve 128 boyutlu olduğunu doğrular."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, 255, 5)
    cv2.rectangle(img, (70, 70), (130, 130), 180, -1)

    kp, des, sure = GorselOznitelikCikarici.sift_cikar(img, maks_nokta=100)
    assert isinstance(kp, list)
    assert len(kp) > 0
    assert des.dtype == np.float32
    assert des.shape[1] == 128
    assert sure >= 0.0


def test_orb_cikar_boyut_ve_tip():
    """ORB tanımlayıcılarının uint8 ve 32 bayt (256 bit) olduğunu doğrular."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, 255, 5)
    cv2.rectangle(img, (70, 70), (130, 130), 180, -1)

    kp, des, sure = GorselOznitelikCikarici.orb_cikar(img, maks_nokta=100)
    assert isinstance(kp, list)
    assert len(kp) > 0
    assert des.dtype == np.uint8
    assert des.shape[1] == 32
    assert sure >= 0.0


def test_hog_cikar_vektor_ve_harita():
    """HOG çıktısının 1D float32 öznitelik vektörü ve 2D görselleştirme haritası olduğunu doğrular."""
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img, (32, 32), 20, 200, -1)

    vektor, harita, sure = GorselOznitelikCikarici.hog_cikar(img, hucre_boyutu=(8, 8), blok_boyutu=(2, 2))
    assert vektor.ndim == 1
    assert vektor.dtype == np.float32
    assert harita.shape == img.shape
    assert sure >= 0.0


def test_lbp_cikar_histogram_toplami():
    """LBP doku haritası ve normalize edilmiş histogramın toplamının 1.0 olduğunu doğrular."""
    img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

    harita, hist, sure = GorselOznitelikCikarici.lbp_cikar(img, yari_cap=1, nokta_sayisi=8)
    assert harita.shape == img.shape
    assert len(hist) == 10  # Uniform LBP (P=8 için 8+2 = 10 bin)
    assert np.isclose(np.sum(hist), 1.0, atol=1e-3)
    assert sure >= 0.0


def test_gecersiz_girdi_kanali():
    """3 kanallı renkli görüntü verildiğinde ValueError fırlatıldığını test eder."""
    renkli = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        GorselOznitelikCikarici.sift_cikar(renkli)

    with pytest.raises(ValueError):
        GorselOznitelikCikarici.orb_cikar(renkli)

    with pytest.raises(ValueError):
        GorselOznitelikCikarici.hog_cikar(renkli)

    with pytest.raises(ValueError):
        GorselOznitelikCikarici.lbp_cikar(renkli)


def test_duz_goruntu_dayanikliligi():
    """Tamamen düz (noktasız) bir görüntüde algoritmanın çökmeden boş tanımlayıcı döndürdüğünü test eder."""
    duz_img = np.zeros((80, 80), dtype=np.uint8)

    kp_sift, des_sift, _ = GorselOznitelikCikarici.sift_cikar(duz_img)
    assert len(kp_sift) == 0
    assert des_sift.shape == (0, 128)

    kp_orb, des_orb, _ = GorselOznitelikCikarici.orb_cikar(duz_img)
    assert len(kp_orb) == 0
    assert des_orb.shape == (0, 32)


def test_analiz_paneli_png_kaydetme(tmp_path):
    """Görselleştiricinin diske geçerli ve dolu bir PNG dosyası kaydettiğini doğrular."""
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (50, 50), 200, -1)

    sift_kp, _, _ = GorselOznitelikCikarici.sift_cikar(img)
    orb_kp, _, _ = GorselOznitelikCikarici.orb_cikar(img)
    _, hog_harita, _ = GorselOznitelikCikarici.hog_cikar(img)
    lbp_harita, lbp_hist, _ = GorselOznitelikCikarici.lbp_cikar(img)

    hedef_dosya = tmp_path / "test_oznitelik_paneli.png"
    cikti = OznitelikGorsellestirici.analiz_paneli_ciz(
        img, sift_kp, orb_kp, hog_harita, lbp_harita, lbp_hist, hedef_dosya
    )

    assert cikti.exists()
    assert cikti.stat().st_size > 0
