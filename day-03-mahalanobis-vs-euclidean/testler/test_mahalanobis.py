"""Mahalanobis ve Kovaryans Analizi Birim Testleri.

Bu dosya; kovaryans matrisi hesaplama doğruluğunu, birim kovaryans durumunda Öklid'e
eşitlenmeyi, tekil matris regülarizasyonunu ve Ki-kare anomali testini doğrular.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Proje kök dizinini ekler
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.kovaryans_ve_mesafe import KovaryansAnalizoru, MahalanobisMesafeOlcer
from src.anomali_tespit_edici import MahalanobisAnomaliDedektoru


def test_kovaryans_matrisi_numpy_ile_uyumu():
    """Özel kovaryans fonksiyonumuzun np.cov ile tam örtüştüğünü doğrular."""
    np.random.seed(123)
    veri = np.random.randn(100, 3)
    bizim_kovaryans = KovaryansAnalizoru.kovaryans_matrisi_hesapla(veri)
    numpy_kovaryans = np.cov(veri, rowvar=False)

    assert np.allclose(bizim_kovaryans, numpy_kovaryans, atol=1e-8)


def test_korelasyon_matrisi_ozellikleri():
    """Korelasyon matrisi köşegeninin 1.0 olduğunu ve [-1, 1] aralığında kaldığını test eder."""
    kovaryans = np.array([
        [4.0, 1.8],
        [1.8, 1.0]
    ])
    korelasyon = KovaryansAnalizoru.korelasyon_matrisi_hesapla(kovaryans)

    assert np.isclose(korelasyon[0, 0], 1.0)
    assert np.isclose(korelasyon[1, 1], 1.0)
    assert np.all(korelasyon >= -1.0) and np.all(korelasyon <= 1.0)


def test_merkeze_mesafe_sifir():
    """Dağılımın tam ortalama noktası için Mahalanobis mesafesinin 0.0 olduğunu denetler."""
    veri = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    olcer = MahalanobisMesafeOlcer(veri)
    mesafe = olcer.tekil_mahalanobis_mesafesi(olcer.ortalama)
    assert np.isclose(mesafe, 0.0, atol=1e-6)


def test_birim_kovaryans_oklide_esitlenir():
    """Kovaryans matrisi Birim Matris (I) olduğunda Mahalanobis == Öklid olmalıdır."""
    # Birbirinden bağımsız standart normal dağılım
    np.random.seed(42)
    ornek_sayisi = 50000
    veri = np.random.randn(ornek_sayisi, 3)
    olcer = MahalanobisMesafeOlcer(veri)

    test_noktasi = np.array([1.5, 2.0, -1.0])
    maha = olcer.tekil_mahalanobis_mesafesi(test_noktasi)
    oklid = olcer.tekil_oklid_mesafesi(test_noktasi)

    # Yeterince büyük örneklemde kovaryans I'ya yakınsar, mesafeler örtüşür
    assert np.isclose(maha, oklid, rtol=0.05)


def test_toplu_mahalanobis_vektorizasyonu():
    """Toplu vektörize hesaplama ile tek tek hesaplamanın aynı sonucu verdiğini doğrular."""
    np.random.seed(99)
    referans = np.random.randn(200, 4)
    olcer = MahalanobisMesafeOlcer(referans)

    test_matrisi = np.random.randn(10, 4)
    toplu_sonuc = olcer.toplu_mahalanobis_mesafesi(test_matrisi)

    for i in range(10):
        tekil_sonuc = olcer.tekil_mahalanobis_mesafesi(test_matrisi[i])
        assert np.isclose(toplu_sonuc[i], tekil_sonuc, atol=1e-7)


def test_tekil_matris_duzenlemesi():
    """Mükemmel doğrusal bağımlı (singular) veride çökme olmadan regülarizasyon yapıldığını test eder."""
    # İkinci sütun birincinin tam 2 katı (determinant = 0)
    x = np.linspace(1, 10, 20)
    veri = np.column_stack([x, 2 * x])

    # Tikhonov düzenlemesi sayesinde hata vermeden başlatılabilmeli
    olcer = MahalanobisMesafeOlcer(veri, duzenleme_katsayisi=1e-4)
    mesafe = olcer.tekil_mahalanobis_mesafesi(np.array([5.0, 10.0]))
    assert mesafe >= 0.0


def test_anomali_dedektoru_siniflandirma():
    """Normal ve aşırı aykırı noktaların doğru etiketlendiğini test eder."""
    np.random.seed(42)
    normal_veri = np.random.randn(500, 2)
    dedektor = MahalanobisAnomaliDedektoru(anlamlilik_duzeyi=0.01)
    dedektor.egit(normal_veri)

    normal_nokta = np.array([[0.1, -0.1]])
    asiri_aykiri = np.array([[10.0, 10.0]])

    tahmin_normal = dedektor.tahmin_et(normal_nokta)
    tahmin_aykiri = dedektor.tahmin_et(asiri_aykiri)

    assert not tahmin_normal[0].anomali_mi
    assert tahmin_aykiri[0].anomali_mi
