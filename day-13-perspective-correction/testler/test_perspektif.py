"""Perspektif Düzeltme ve Geometrik Dönüşümler Birim Testleri.

Bu dosya; 4 köşe sıralama algoritmasını, hedef boyut hesaplamasını,
homografi matrisinin tersinirliğini, açı döndürmeyi ve görsel rapor üretimini test eder.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.perspektif_duzeltici import PerspektifDuzeltici
from src.gorsellestirici import PerspektifGorsellestirici


def test_noktalari_sirala_karisik_sira():
    """Karışık verilen 4 noktanın [Sol-Üst, Sağ-Üst, Sağ-Alt, Sol-Alt] sırasına dizildiğini test eder."""
    # Bilinen köşe noktaları
    tl = np.array([20.0, 30.0])
    tr = np.array([180.0, 25.0])
    br = np.array([190.0, 210.0])
    bl = np.array([15.0, 205.0])

    karisik = np.array([br, tl, bl, tr])
    sirali = PerspektifDuzeltici.noktalari_sirala(karisik)

    assert np.allclose(sirali[0], tl)
    assert np.allclose(sirali[1], tr)
    assert np.allclose(sirali[2], br)
    assert np.allclose(sirali[3], bl)


def test_noktalari_sirala_gecersiz_sekil():
    """(4, 2) boyutunda olmayan nokta dizilerinde ValueError fırlatıldığını doğrular."""
    with pytest.raises(ValueError):
        PerspektifDuzeltici.noktalari_sirala(np.zeros((3, 2)))

    with pytest.raises(ValueError):
        PerspektifDuzeltici.noktalari_sirala(np.zeros((4, 3)))


def test_hedef_boyutlari_hesapla_pozitif():
    """Hesaplanan hedef genişlik ve yükseklik değerlerinin pozitif tamsayı olduğunu doğrular."""
    noktalar = np.array([
        [0.0, 0.0],
        [100.0, 10.0],
        [95.0, 150.0],
        [5.0, 140.0]
    ], dtype=np.float32)

    w, h = PerspektifDuzeltici.hedef_boyutlari_hesapla(noktalar)
    assert isinstance(w, int) and isinstance(h, int)
    assert w > 0 and h > 0
    assert w == pytest.approx(100, abs=10)
    assert h == pytest.approx(145, abs=10)


def test_birim_donusum_ve_homografi_sekli():
    """Homografi matrisinin (3, 3) boyutunda çıktığını ve istenen çözünürlüğü ürettiğini test eder."""
    gorsel = np.zeros((200, 200, 3), dtype=np.uint8)
    noktalar = np.array([[20, 20], [180, 30], [170, 180], [10, 170]], dtype=np.float32)

    duzeltilmis, H = PerspektifDuzeltici.dort_nokta_donusumu(
        gorsel, noktalar, hedef_genislik=100, hedef_yukseklik=120
    )

    assert H.shape == (3, 3)
    assert duzeltilmis.shape == (120, 100, 3)


def test_homografi_tersinirlik():
    """Hesaplanan homografi matrisinin tersinir olduğunu (determinant != 0) doğrular."""
    gorsel = np.zeros((100, 100, 3), dtype=np.uint8)
    noktalar = np.array([[10, 10], [90, 15], [85, 95], [5, 85]], dtype=np.float32)

    _, H = PerspektifDuzeltici.dort_nokta_donusumu(gorsel, noktalar)
    det_h = np.linalg.det(H)

    assert abs(det_h) > 1e-7


def test_egim_acisi_duzelt_boyutlar():
    """Eğim düzeltme fonksiyonunun görüntüyü döndürdüğünü ve geçerli şekil ürettiğini test eder."""
    gorsel = np.ones((80, 100, 3), dtype=np.uint8) * 128
    dondurulmus = PerspektifDuzeltici.egim_acisi_duzelt(gorsel, aci_derece=15.0)

    assert dondurulmus.ndim == 3
    assert dondurulmus.shape[2] == 3
    assert dondurulmus.shape[0] > 0 and dondurulmus.shape[1] > 0


def test_analiz_paneli_png_kaydetme(tmp_path):
    """Analiz görsel panelinin diske fiziksel geçerli PNG dosyası yazdığını doğrular."""
    orijinal = np.zeros((100, 100, 3), dtype=np.uint8)
    sirali = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    duzeltilmis = np.zeros((80, 80, 3), dtype=np.uint8)
    H = np.eye(3, dtype=np.float64)

    hedef = tmp_path / "perspektif_test.png"
    cikti = PerspektifGorsellestirici.analiz_paneli_ciz(orijinal, sirali, duzeltilmis, H, hedef)

    assert cikti.exists()
    assert cikti.stat().st_size > 0
