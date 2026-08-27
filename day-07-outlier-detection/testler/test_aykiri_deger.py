"""İstatistiksel ve ML Tabanlı Aykırı Değer Tespiti Birim Testleri.

Bu dosya; Z-Skoru, Modifiye Z-Skoru, IQR, İzolasyon Ormanı,
LOF ve karşılaştırma motorunun doğruluğunu test eder.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.istatistiksel_tespit import ZSkoruTespitEdici, IqrAykiriDegerTespitEdici
from src.makine_ogrenmesi_tespiti import IzolasyonOrmaniTespitEdici, LokalAykiriFaktorTespitEdici
from src.karsilastirma_ve_gorsellestirme import AykiriDegerKarsilastirici, AykiriDegerGorsellestirici


def test_z_skoru_tespiti():
    """Z-Skorunun aşırı uç değerleri başarıyla tespit ettiğini doğrular."""
    veri = np.array([10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 100.0])  # 100.0 aşırı uç
    bulucu = ZSkoruTespitEdici(esik_degeri=2.0)
    maske = bulucu.tespit_et(veri)

    assert maske[-1] == True
    assert np.sum(maske[:-1]) == 0


def test_modifiye_z_skoru_mad():
    """Modifiye Z-Skorunun medyan/MAD ile maskelenmeden ucu yakaladığını doğrular."""
    veri = np.array([5.0, 5.1, 4.9, 5.0, 5.2, 50.0])
    bulucu = ZSkoruTespitEdici(esik_degeri=3.5, modifiye_kullan=True)
    maske = bulucu.tespit_et(veri)

    assert maske[-1] == True
    assert maske[0] == False


def test_iqr_aykiri_tespiti():
    """IQR yönteminin çeyreklik sınırları doğru hesapladığını test eder."""
    veri = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
    iqr_bulucu = IqrAykiriDegerTespitEdici(carpan=1.5)
    maske = iqr_bulucu.tespit_et(veri)

    assert maske[-1] == True
    assert iqr_bulucu.ust_sinir < 100


def test_izolasyon_ormani_tespiti():
    """İzolasyon ormanının 2B uzaydaki uzak anomaliyi tespit ettiğini doğrular."""
    np.random.seed(42)
    normal_veri = np.random.normal(0, 1, (100, 2))
    aykiri_nokta = np.array([[20.0, 20.0]])
    tum_veri = np.vstack([normal_veri, aykiri_nokta])

    model = IzolasyonOrmaniTespitEdici(kirlilik_orani=0.02, rastgele_durum=42)
    maske = model.egit_ve_tespit_et(tum_veri)

    assert maske[-1] == True


def test_lokal_aykiri_faktor_tespiti():
    """LOF algoritmasının yerel yoğunluk farkını yakaladığını test eder."""
    np.random.seed(42)
    kume = np.random.normal(0, 0.5, (80, 2))
    aykiri = np.array([[5.0, 5.0]])
    veri = np.vstack([kume, aykiri])

    model = LokalAykiriFaktorTespitEdici(komsu_sayisi=15, kirlilik_orani=0.03)
    maske = model.egit_ve_tespit_et(veri)

    assert maske[-1] == True


def test_karsilastirici_ve_mutabakat():
    """Karşılaştırıcının 4 yöntemi de hatasız koşturduğunu doğrular."""
    veri = np.random.randn(50, 2)
    karsilastirici = AykiriDegerKarsilastirici(veri)
    sonuclar = karsilastirici.tum_yontemleri_calistir(kirlilik_orani=0.05)

    assert len(sonuclar) == 4
    mutabakat = karsilastirici.mutabakat_analizi(sonuclar)
    assert "oy_dagilimi" in mutabakat
    assert len(mutabakat["toplam_oylar"]) == 50


def test_gorsellestirme_png_uretimi(tmp_path):
    """Karşılaştırma grafiğinin fiziksel olarak diske kaydedildiğini test eder."""
    veri = np.random.randn(30, 2)
    karsilastirici = AykiriDegerKarsilastirici(veri)
    sonuclar = karsilastirici.tum_yontemleri_calistir(kirlilik_orani=0.1)

    hedef_png = tmp_path / "test_aykiri.png"
    cikti = AykiriDegerGorsellestirici.karsilastirma_grafigi_ciz(veri, sonuclar, hedef_png)

    assert cikti.exists()
    assert cikti.stat().st_size > 0
