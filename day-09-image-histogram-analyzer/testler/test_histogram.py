"""Görüntü Histogramı ve Kontrast İyileştirme Birim Testleri.

Bu dosya; histogram hesaplamalarını, CDF kümülatif eğri özelliklerini,
Global Histogram Eşitleme ve CLAHE operatörlerini test eder.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.histogram_motoru import HistogramHesaplayici, KontrastIyilestirici
from src.gorsellestirici import HistogramGorsellestirici


def test_kanal_histogrami_toplami_ve_boyutu():
    """Histogram boyutunun 256 olduğunu ve eleman toplamının piksel sayısına eşit olduğunu test eder."""
    resim = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    hist = HistogramHesaplayici.kanal_histogrami(resim)

    assert len(hist) == 256
    assert np.isclose(np.sum(hist), 64 * 64)


def test_renkli_histogram_kanallari():
    """3 kanallı BGR görselden 3 ayrı renk histogramı çıkarıldığını doğrular."""
    resim_bgr = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
    hist_sozluk = HistogramHesaplayici.renkli_histogramlar(resim_bgr)

    assert set(hist_sozluk.keys()) == {"Mavi", "Yeşil", "Kırmızı"}
    for kanal_adi, hist in hist_sozluk.items():
        assert len(hist) == 256


def test_cdf_monotonik_artisi_ve_sinirlari():
    """CDF eğrisinin monotonik olarak azalan olmadığını ve son değerinin 1.0 olduğunu doğrular."""
    hist = np.array([10, 20, 30, 40], dtype=np.float32)
    cdf = HistogramHesaplayici.kumulatif_dagilim_cdf(hist, normalize_et=True)

    assert cdf[-1] == pytest.approx(1.0)
    # Monotonik artış kontrolü: her adım bir öncekinden büyük veya eşit olmalıdır
    assert np.all(np.diff(cdf) >= 0)


def test_kontrast_metrikleri_hesaplama():
    """Kontrast metriklerinin tüm alanlarının eksiksiz ve mantıklı olduğunu test eder."""
    resim = np.full((50, 50), 100, dtype=np.uint8)
    resim[0, 0] = 150
    metrikler = HistogramHesaplayici.kontrast_metrikleri(resim)

    assert "dinamik_aralik" in metrikler
    assert "rms_kontrast" in metrikler
    assert "shannon_entropisi" in metrikler
    assert metrikler["dinamik_aralik"] == 50.0


def test_global_esitleme_gri_ve_renkli():
    """Hem 2B gri hem de 3B renkli görüntünün boyutu bozulmadan eşitlendiğini doğrular."""
    gri = np.random.randint(50, 100, (50, 50), dtype=np.uint8)
    esit_gri = KontrastIyilestirici.global_histogram_esitle(gri)
    assert esit_gri.shape == gri.shape

    renkli = np.random.randint(50, 100, (50, 50, 3), dtype=np.uint8)
    esit_renkli = KontrastIyilestirici.global_histogram_esitle(renkli)
    assert esit_renkli.shape == renkli.shape


def test_clahe_iyilestirmesi():
    """CLAHE uygulamasının dinamik aralığı başarıyla genişlettiğini doğrular."""
    dar_aralik = np.random.randint(40, 60, (64, 64), dtype=np.uint8)
    clahe_resim = KontrastIyilestirici.clahe_uygula(dar_aralik, kirpma_limiti=2.0)

    assert clahe_resim.shape == dar_aralik.shape
    assert np.max(clahe_resim) > np.max(dar_aralik)


def test_histogram_raporu_png_kaydetme(tmp_path):
    """Karşılaştırma panelinin diske fiziksel geçerli PNG dosyası yazdığını test eder."""
    resim = np.full((30, 30), 100, dtype=np.uint8)
    gorseller = {"Test1": resim, "Test2": resim, "Test3": resim}
    hedef = tmp_path / "hist_rapor.png"

    cikti = HistogramGorsellestirici.analiz_raporu_ciz(gorseller, hedef)
    assert cikti.exists()
    assert cikti.stat().st_size > 0
