"""Görüntü İşleme Araç Seti (Filtreleme, Sobel, Morfoloji) Birim Testleri.

Bu dosya; konvolüsyon operatörlerini, Gauss yumuşatmasını, Sobel kenar gradyanlarını
ve matematiksel morfolojik dönüşümlerin doğruluğunu test eder.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.filtreler import KonvolusyonFiltresi, GaussBulaniklastirici, SobelKenarTespitEdici
from src.morfoloji import MorfolojikIslemci
from src.gorsellestirici import IslemePaneliUreteci


def test_konvolusyon_boyut_korunumu():
    """Özel 3x3 çekirdekle konvolüsyonun boyutları koruduğunu doğrular."""
    resim = np.ones((64, 64), dtype=np.uint8) * 100
    cekirdek = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    sonuc = KonvolusyonFiltresi.ozel_cekirdek_uygula(resim, cekirdek)
    assert sonuc.shape == resim.shape
    assert sonuc.dtype == resim.dtype


def test_gauss_bulaniklastirma_gurultu_azaltma():
    """Gauss bulanıklaştırmanın yüksek frekanslı gürültüyü ve yerel varyansı düşürdüğünü test eder."""
    np.random.seed(42)
    temiz = np.ones((50, 50), dtype=np.uint8) * 128
    gurultulu = temiz + np.random.randint(-20, 20, (50, 50)).astype(np.uint8)

    bulanik = GaussBulaniklastirici.bulaniklastir(gurultulu, cekirdek_boyutu=(5, 5), sigma_x=1.5)

    assert np.std(bulanik) < np.std(gurultulu)


def test_gauss_gecersiz_cekirdek_hatasi():
    """Çift boyutlu çekirdek verildiğinde ValueError fırlatıldığını doğrular."""
    resim = np.ones((30, 30), dtype=np.uint8)
    with pytest.raises(ValueError, match="pozitif tek sayılar"):
        GaussBulaniklastirici.bulaniklastir(resim, cekirdek_boyutu=(4, 4))


def test_sobel_kenar_yakalama():
    """Dikey bir geçişte Sobel Gx gradyanının yüksek kenar tepkisi verdiğini test eder."""
    resim = np.zeros((50, 50), dtype=np.uint8)
    resim[:, 25:] = 255  # Tam ortada dikey kenar

    gx, gy, magnitut = SobelKenarTespitEdici.gradyan_hesapla(resim, cekirdek_boyutu=3)

    assert np.max(gx) > 200  # Yatay türev dikey kenarı yakalamalıdır
    assert np.max(magnitut) > 200


def test_morfoloji_acma_gurultu_temizleme():
    """Açma (Opening) işleminin izole beyaz tekil pikseli yok ettiğini test eder."""
    resim = np.zeros((30, 30), dtype=np.uint8)
    resim[15, 15] = 255  # 1 piksellik izole nokta

    cekirdek = MorfolojikIslemci.yapisal_element_olustur((3, 3), "dikdortgen")
    sonuc = MorfolojikIslemci.acma(resim, cekirdek)

    assert sonuc[15, 15] == 0  # İzole nokta süpürülmüş olmalıdır


def test_morfoloji_kapatma_delik_doldurma():
    """Kapatma (Closing) işleminin nesne içindeki tek piksellik deliği tıkadığını test eder."""
    resim = np.ones((30, 30), dtype=np.uint8) * 255
    resim[15, 15] = 0  # 1 piksellik delik

    cekirdek = MorfolojikIslemci.yapisal_element_olustur((3, 3), "dikdortgen")
    sonuc = MorfolojikIslemci.kapatma(resim, cekirdek)

    assert sonuc[15, 15] == 255  # Delik doldurulmuş olmalıdır


def test_isleme_paneli_kaydetme(tmp_path):
    """Karşılaştırma panelinin geçerli bir PNG dosyası olarak kaydedildiğini doğrular."""
    resim = np.zeros((20, 20), dtype=np.uint8)
    adimlar = {"Orijinal": resim, "Test": resim}
    hedef_dosya = tmp_path / "panel_test.png"

    cikti = IslemePaneliUreteci.panel_olustur_ve_kaydet(adimlar, hedef_dosya)
    assert cikti.exists()
    assert cikti.stat().st_size > 0
