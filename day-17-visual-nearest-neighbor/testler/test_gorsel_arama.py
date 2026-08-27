"""Vektör Tabanlı Görsel Arama ve k-NN Birim Testleri.

Bu dosya; hibrit vektör çıkarımını, L2/Cosine mesafe metriklerini, Top-K sıralama doğruluğunu,
hata yakalama mekanizmalarını ve görselleştirme çıktısını test eder.
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

from src.vektor_cikarici import GorselVektorCikarici
from src.knn_arama_motoru import GorselAramaMotoru
from src.gorsellestirici import AramaGorsellestirici


def test_vektor_cikar_boyut_ve_norm():
    """Görsel vektörünün 1642 boyutlu olduğunu ve L2 normunun 1.0 olduğunu doğrular."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.circle(img, (32, 32), 15, (0, 200, 100), -1)

    cikarici = GorselVektorCikarici(standart_boyut=(128, 128))
    vec = cikarici.vektor_cikar(img)

    assert vec.ndim == 1
    assert len(vec) == 1642  # 64 (Renk) + 10 (LBP) + 1568 (HOG)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_katalog_ekle_ve_matris_boyutu():
    """Kataloğa görsel eklendikçe katalog matrisinin (N, D) boyutunda güncellendiğini test eder."""
    cikarici = GorselVektorCikarici(standart_boyut=(64, 64))
    motor = GorselAramaMotoru(vektor_cikarici=cikarici)

    img1 = np.full((50, 50, 3), 100, dtype=np.uint8)
    img2 = np.full((50, 50, 3), 200, dtype=np.uint8)

    motor.katalog_ekle("urun_1", img1)
    assert motor.katalog_matrisi.shape[0] == 1

    motor.katalog_ekle("urun_2", img2)
    assert motor.katalog_matrisi.shape[0] == 2
    assert len(motor.katalog_etiketler) == 2


def test_en_yakin_k_siralamasi():
    """Top-K sonuçlarının küçükten büyüğe artan mesafe sırasına sahip olduğunu doğrular."""
    motor = GorselAramaMotoru()
    for i in range(5):
        img = np.full((50, 50, 3), i * 40, dtype=np.uint8)
        motor.katalog_ekle(f"katalog_{i}", img)

    sorgu = np.full((50, 50, 3), 50, dtype=np.uint8)
    sonuclar = motor.en_yakin_k_ara(sorgu, k=4, metrik="cosine")

    assert len(sonuclar) == 4
    mesafeler = [r.mesafe for r in sonuclar]
    assert mesafeler == sorted(mesafeler)


def test_birebir_ayni_gorsel_mesafe_sifir():
    """Katalogdaki bir görsel doğrudan sorgulandığında mesafenin ~0 ve benzerliğin %100 olduğunu doğrular."""
    motor = GorselAramaMotoru()
    hedef_img = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(hedef_img, (40, 40), 20, (10, 150, 240), -1)

    motor.katalog_ekle("hedef_urun", hedef_img)
    motor.katalog_ekle("diger_urun", np.full((80, 80, 3), 120, dtype=np.uint8))

    sonuclar = motor.en_yakin_k_ara(hedef_img, k=1, metrik="cosine")
    assert sonuclar[0].etiket == "hedef_urun"
    assert np.isclose(sonuclar[0].mesafe, 0.0, atol=1e-4)
    assert np.isclose(sonuclar[0].benzerlik_yuzdesi, 100.0, atol=1e-2)


def test_bos_katalog_hatasi():
    """Boş katalogda arama yapılmak istendiğinde ValueError fırlatıldığını doğrular."""
    motor = GorselAramaMotoru()
    sorgu = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        motor.en_yakin_k_ara(sorgu, k=3)


def test_gecersiz_k_degeri_hatasi():
    """Negatif veya sıfır k değeri verildiğinde ValueError fırlatıldığını doğrular."""
    motor = GorselAramaMotoru()
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    motor.katalog_ekle("urun_1", img)

    with pytest.raises(ValueError):
        motor.en_yakin_k_ara(img, k=0)

    with pytest.raises(ValueError):
        motor.en_yakin_k_ara(img, k=-2)


def test_arama_raporu_png_kaydetme(tmp_path):
    """Arama raporunun diske geçerli ve dolu bir PNG dosyası kaydettiğini doğrular."""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    motor = GorselAramaMotoru()
    motor.katalog_ekle("test_1", img)
    motor.katalog_ekle("test_2", img)

    sonuclar = motor.en_yakin_k_ara(img, k=2)
    hedef_dosya = tmp_path / "test_arama_raporu.png"

    cikti = AramaGorsellestirici.arama_raporu_ciz(img, sonuclar, "Cosine", hedef_dosya)
    assert cikti.exists()
    assert cikti.stat().st_size > 0
