"""Motif Segmentasyonu ve Kontur Ayrıştırma Birim Testleri.

Bu dosya; Otsu eşiklemeyi, morfolojik temizliği, dairesellik ve doluluk (solidity)
hesaplamalarını, sınırlayıcı kutuların sınırlarını ve görsel raporun üretimini test eder.
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

from src.motif_ayristirici import MotifAyristirici, MotifBilgisi
from src.gorsellestirici import MotifGorsellestirici


def test_otsu_esikleme_temel():
    """Otsu eşiklemenin ikili maske ürettiğini ve eşik değerinin [0, 255] arasında olduğunu test eder."""
    gri = np.full((60, 60), 30, dtype=np.uint8)
    gri[15:45, 15:45] = 200  # Ortada parlak kare

    maske, esik = MotifAyristirici.otsu_esikleme(gri)
    assert maske.shape == (60, 60)
    assert set(np.unique(maske)).issubset({0, 255})
    assert 0.0 < esik < 255.0


def test_morfolojik_temizleme():
    """İzole 1 piksellik tuz gürültüsünün morfolojik açma ile silindiğini doğrular."""
    maske = np.zeros((40, 40), dtype=np.uint8)
    maske[10, 10] = 255  # Tek gürültü pikseli

    temiz = MotifAyristirici.morfolojik_temizleme(maske, cekirdek_boyutu=3)
    assert np.sum(temiz) == 0


def test_daire_dairesellik_skoru():
    """Kusursuz bir dairenin dairesellik metriğinin 0.85'in üzerinde çıktığını doğrular."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(img, (60, 60), 30, (255, 255, 255), -1)

    motifler, _ = MotifAyristirici.motifleri_ayristir(img, min_alan=500.0)
    assert len(motifler) == 1
    assert motifler[0].dairesellik > 0.85


def test_motif_ayristirici_filtreleme():
    """Eşik altında kalan çok küçük lekelerin elendiğini doğrular."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (20, 20), 2, (255, 255, 255), -1)  # Çok küçük leke
    cv2.rectangle(img, (40, 40), (80, 80), (255, 255, 255), -1)  # Büyük motif

    motifler, _ = MotifAyristirici.motifleri_ayristir(img, min_alan=200.0)
    assert len(motifler) == 1
    assert motifler[0].alan > 500.0


def test_sinirlayici_kutu_koordinatlari():
    """Sınırlayıcı kutunun pozitif genişlik/yükseklikte olduğunu doğrular."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 25), (75, 80), (255, 255, 255), -1)

    motifler, _ = MotifAyristirici.motifleri_ayristir(img, min_alan=200.0)
    assert len(motifler) == 1
    x, y, w, h = motifler[0].sinirlayici_kutu
    assert w > 0 and h > 0
    assert x >= 0 and y >= 0
    assert x + w <= 100 and y + h <= 100


def test_kirpilmis_gorsel_ve_maske_boyutlari():
    """Kırpılan ROI görüntüsünün sınırlayıcı kutu boyutlarıyla birebir eşleştiğini test eder."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(img, (60, 60), 25, (100, 150, 200), -1)

    motifler, _ = MotifAyristirici.motifleri_ayristir(img, min_alan=200.0)
    assert len(motifler) == 1
    m = motifler[0]
    _, _, w, h = m.sinirlayici_kutu
    assert m.kirpilmis_gorsel.shape == (h, w, 3)
    assert m.kirpilmis_maske.shape == (h, w)


def test_analiz_paneli_png_kaydetme(tmp_path):
    """Görsel analiz panelinin diske geçerli ve dolu bir PNG dosyası kaydettiğini doğrular."""
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(img, (40, 40), 20, (200, 200, 200), -1)

    motifler, maske = MotifAyristirici.motifleri_ayristir(img, min_alan=100.0)
    hedef = tmp_path / "motif_test.png"

    cikti = MotifGorsellestirici.analiz_paneli_ciz(img, maske, motifler, hedef)
    assert cikti.exists()
    assert cikti.stat().st_size > 0
