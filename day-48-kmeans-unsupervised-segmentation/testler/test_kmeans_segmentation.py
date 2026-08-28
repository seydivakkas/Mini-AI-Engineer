"""
Day 48: K-Means ile Denetimsiz Görüntü & Özellik Bölütleme Birim Testleri.
"""

import os
import pytest
import numpy as np
from src.kmeans_bolutleyici import KMeansGorselBolutleyici
from src.kume_optimizasyonu import KumeOptimizatoru
from src.gorsellestirici import KMeansBolutlemeGorsellestirici


@pytest.fixture
def ornek_gorsel():
    np.random.seed(42)
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[:30, :30] = [255, 0, 0]    # Kırmızı
    img[:30, 30:] = [0, 255, 0]    # Yeşil
    img[30:, :30] = [0, 0, 255]    # Mavi
    img[30:, 30:] = [255, 255, 0]  # Sarı
    return img


def test_sentetik_gorsel_boyutu(ornek_gorsel):
    """Görüntünün doğru boyut ve kanallara sahip olduğunu test eder."""
    assert ornek_gorsel.shape == (60, 60, 3)
    assert ornek_gorsel.dtype == np.uint8


def test_kume_optimizasyonu_en_iyi_k(ornek_gorsel):
    """Elbow ve Silhouette analizinin doğru K değerleri ürettiğini test eder."""
    duz = (ornek_gorsel.reshape(-1, 3).astype(np.float32)) / 255.0
    sonuc = KumeOptimizatoru.en_iyi_k_bul(duz, k_araligi=(2, 5))

    assert "en_iyi_k" in sonuc
    assert 2 <= sonuc["en_iyi_k"] <= 5
    assert len(sonuc["wcss_degerleri"]) == 4
    assert len(sonuc["silhouette_degerleri"]) == 4


def test_renk_kuantalama_cikti_boyutlari(ornek_gorsel):
    """Renk kuantalama çıktılarının boyut ve tiplerini test eder."""
    bolutleyici = KMeansGorselBolutleyici(k_kume=4)
    kuant_img, maske, merkezler = bolutleyici.renk_kuantalama_uygula(ornek_gorsel)

    assert kuant_img.shape == ornek_gorsel.shape
    assert maske.shape == (60, 60)
    assert merkezler.shape == (4, 3)
    assert len(np.unique(maske)) <= 4


def test_uzamsal_bolutleme_alan_yuzdeleri(ornek_gorsel):
    """Uzamsal bölütlemenin alan yüzdeleri toplamının %100 olduğunu test eder."""
    bolutleyici = KMeansGorselBolutleyici(k_kume=4, uzamsal_agirlik=0.5)
    sonuc = bolutleyici.uzamsal_bolutleme_uygula(ornek_gorsel)

    assert "alan_yuzdeleri" in sonuc
    toplam_alan = sum(sonuc["alan_yuzdeleri"].values())
    assert pytest.approx(toplam_alan, abs=0.2) == 100.0
    assert sonuc["bolutlenmis_gorsel"].shape == ornek_gorsel.shape


def test_uzamsal_agirlik_etkisi(ornek_gorsel):
    """Farklı uzamsal ağırlıkların bölütleyici tarafından kabul edildiğini test eder."""
    bolutleyici = KMeansGorselBolutleyici(k_kume=3, uzamsal_agirlik=0.1)
    sonuc = bolutleyici.uzamsal_bolutleme_uygula(ornek_gorsel)
    assert sonuc["uzamsal_agirlik"] == 0.1


def test_kmeans_sinir_durumlari_tek_renk():
    """Tamamen tek renkli bir görüntüde bölütleyicinin çökmediğini test eder."""
    tek_renk = np.full((40, 40, 3), 128, dtype=np.uint8)
    bolutleyici = KMeansGorselBolutleyici(k_kume=2)
    sonuc = bolutleyici.uzamsal_bolutleme_uygula(tek_renk)
    assert sonuc["bolutlenmis_gorsel"].shape == (40, 40, 3)


def test_gorsellestirici_panel_cizimi(ornek_gorsel, tmp_path):
    """6 panelli görselleştiricinin PNG dosyası ürettiğini test eder."""
    bolutleyici = KMeansGorselBolutleyici(k_kume=4)
    kuant_img, _, _ = bolutleyici.renk_kuantalama_uygula(ornek_gorsel)
    uzamsal_sonuc = bolutleyici.uzamsal_bolutleme_uygula(ornek_gorsel)
    kume_analizi = {
        "k_degerleri": [2, 3, 4, 5],
        "wcss_degerleri": [500.0, 300.0, 100.0, 80.0],
        "silhouette_degerleri": [0.55, 0.62, 0.88, 0.75],
        "en_iyi_k": 4,
        "en_iyi_silhouette": 0.88
    }

    cikis_yolu = str(tmp_path / "test_kmeans_paneli.png")
    yol = KMeansBolutlemeGorsellestirici.panel_ciz(
        orijinal_gorsel=ornek_gorsel,
        kuantalanmis_gorsel=kuant_img,
        uzamsal_sonuc=uzamsal_sonuc,
        kume_analizi=kume_analizi,
        hedef_path=cikis_yolu
    )

    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
