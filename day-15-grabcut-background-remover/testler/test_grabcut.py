"""GrabCut Segmentasyon ve Arka Plan Kaldırma Birim Testleri.

Bu dosya; GrabCut algoritmasını, 4-durumlu maske değerlerini, geçersiz kutu fırlatmalarını,
şeffaf BGRA üretimini, arka plan kompozisyonunu ve görsel rapor çıktısını test eder.
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

from src.grabcut_ayristirici import GrabCutAyristirici
from src.gorsellestirici import GrabCutGorsellestirici


def test_dikdortgen_ile_ayristir_maske_degerleri():
    """GrabCut maske değerlerinin {0, 1, 2, 3} ve ikili maskenin {0, 255} olduğunu doğrular."""
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(img, (40, 40), 20, (50, 150, 200), -1)

    ayristirici = GrabCutAyristirici()
    on_plan, ikili, ham = ayristirici.dikdortgen_ile_ayristir(img, (15, 15, 50, 50), iterasyon_sayisi=1)

    assert set(np.unique(ham)).issubset({0, 1, 2, 3})
    assert set(np.unique(ikili)).issubset({0, 255})
    assert on_plan.shape == img.shape


def test_gecersiz_dikdortgen_hatasi():
    """Görüntü sınırlarını aşan veya sıfır boyutlu sınırlayıcı kutuların ValueError fırlattığını doğrular."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    ayristirici = GrabCutAyristirici()

    with pytest.raises(ValueError):
        ayristirici.dikdortgen_ile_ayristir(img, (-5, 10, 30, 30))

    with pytest.raises(ValueError):
        ayristirici.dikdortgen_ile_ayristir(img, (10, 10, 60, 30))  # x + w > 50


def test_seffaf_png_olustur_sekil_ve_kanallar():
    """Şeffaf görselin 4 kanallı (BGRA) olduğunu ve alfa kanalının ikili maske olduğunu doğrular."""
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    maske = np.zeros((40, 40), dtype=np.uint8)
    maske[10:30, 10:30] = 255

    bgra = GrabCutAyristirici.seffaf_png_olustur(img, maske)
    assert bgra.shape == (40, 40, 4)
    assert np.array_equal(bgra[:, :, 3], maske)


def test_arka_plan_degistir_sekil():
    """Yeni arka planla birleştirilen kompozit görselin geçerli boyut ve tipte olduğunu test eder."""
    on_plan = np.zeros((50, 50, 3), dtype=np.uint8)
    maske = np.zeros((50, 50), dtype=np.uint8)
    yeni_arkaplan = np.full((50, 50, 3), 200, dtype=np.uint8)

    kompozit = GrabCutAyristirici.arka_plan_degistir(on_plan, maske, yeni_arkaplan)
    assert kompozit.shape == (50, 50, 3)
    assert kompozit.dtype == np.uint8


def test_maske_ile_iyilestir():
    """İnteraktif fırça noktaları eklendiğinde maskenin güncellendiğini doğrular."""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (40, 40), (200, 50, 50), -1)

    ayristirici = GrabCutAyristirici()
    _, _, ham = ayristirici.dikdortgen_ile_ayristir(img, (10, 10, 40, 40), iterasyon_sayisi=1)

    # Ön plan fırçası ekle
    _, ikili_yeni, ham_yeni = ayristirici.maske_ile_iyilestir(
        img, ham, kesin_on_plan_noktalari=[(30, 30)], iterasyon_sayisi=1
    )
    assert ikili_yeni[30, 30] == 255


def test_izole_on_plan_arka_plan_karartma():
    """İkili maskenin sıfır olduğu piksellerin izole ön planda siyah (0, 0, 0) kaldığını test eder."""
    img = np.full((40, 40, 3), 255, dtype=np.uint8)
    ayristirici = GrabCutAyristirici()
    on_plan, ikili, _ = ayristirici.dikdortgen_ile_ayristir(img, (10, 10, 20, 20), iterasyon_sayisi=1)

    # Maske 0 olan yerlerde pikseller siyah olmalı
    assert np.all(on_plan[ikili == 0] == 0)


def test_analiz_paneli_png_kaydetme(tmp_path):
    """Görsel analiz panelinin diske fiziksel geçerli PNG dosyası yazdığını doğrular."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    ham_maske = np.zeros((50, 50), dtype=np.uint8)
    kompozit = np.zeros((50, 50, 3), dtype=np.uint8)
    hedef = tmp_path / "grabcut_test.png"

    cikti = GrabCutGorsellestirici.analiz_paneli_ciz(
        img, (10, 10, 30, 30), ham_maske, img, kompozit, hedef
    )
    assert cikti.exists()
    assert cikti.stat().st_size > 0
