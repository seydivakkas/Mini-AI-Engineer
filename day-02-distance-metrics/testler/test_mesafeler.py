"""Mesafe ve Benzerlik Metrikleri Birim Testleri.

Bu dosya; Öklid, Manhattan, Chebyshev, Kosinüs ve Minkowski metriklerinin
matematiksel doğruluğunu, sınır durumlarını ve toplu işlem vektörizasyonunu test eder.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Proje kök dizinini ekler
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.mesafe_olcer import MesafeOlcer
from src.gorsel_eslestirici import GorselBenzerlikEslestirici


def test_oklid_mesafesi_pisagor():
    """3-4-5 dik üçgeni koordinatlarında Öklid mesafesinin tam 5.0 çıktığını doğrular."""
    v1 = np.array([0.0, 0.0])
    v2 = np.array([3.0, 4.0])
    mesafe = MesafeOlcer.oklid_mesafesi(v1, v2)
    assert np.isclose(mesafe, 5.0)


def test_manhattan_mesafesi_toplami():
    """Koordinat mutlak farkları toplamının Manhattan formülüne uyduğunu kontrol eder."""
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 0.0, -1.0])
    # |1-4| + |2-0| + |3 - (-1)| = 3 + 2 + 4 = 9.0
    assert np.isclose(MesafeOlcer.manhattan_mesafesi(v1, v2), 9.0)


def test_chebyshev_mesafesi_maksimumu():
    """En büyük mutlak farkın Chebyshev mesafesini oluşturduğunu doğrular."""
    v1 = np.array([1.0, 2.0, 10.0])
    v2 = np.array([4.0, 6.0, 3.0])
    # max(|1-4|, |2-6|, |10-3|) = max(3, 4, 7) = 7.0
    assert np.isclose(MesafeOlcer.chebyshev_mesafesi(v1, v2), 7.0)


def test_kosinus_benzerligi_ozel_durumlar():
    """Paralel, dik ve zıt vektörlerde kosinüs benzerliği sınırlarını test eder."""
    # 1. Tam paralel (aynı yön, farklı büyüklük) -> 1.0
    p1 = np.array([2.0, 4.0])
    p2 = np.array([20.0, 40.0])
    assert np.isclose(MesafeOlcer.kosinus_benzerligi(p1, p2), 1.0)

    # 2. Tam dik (ortogonal) -> 0.0
    d1 = np.array([1.0, 0.0])
    d2 = np.array([0.0, 1.0])
    assert np.isclose(MesafeOlcer.kosinus_benzerligi(d1, d2), 0.0)

    # 3. Tam zıt vektörler -> -1.0
    z1 = np.array([1.0, 5.0])
    z2 = np.array([-1.0, -5.0])
    assert np.isclose(MesafeOlcer.kosinus_benzerligi(z1, z2), -1.0)


def test_minkowski_ozellesmeleri():
    """p=1 için Manhattan'a, p=2 için Öklid'e eşitlendiğini doğrular."""
    v1 = np.array([1.5, 3.2, 5.8])
    v2 = np.array([4.1, 1.1, 8.4])

    p1 = MesafeOlcer.minkowski_mesafesi(v1, v2, p_derecesi=1.0)
    manhattan = MesafeOlcer.manhattan_mesafesi(v1, v2)
    assert np.isclose(p1, manhattan)

    p2 = MesafeOlcer.minkowski_mesafesi(v1, v2, p_derecesi=2.0)
    oklid = MesafeOlcer.oklid_mesafesi(v1, v2)
    assert np.isclose(p2, oklid)


def test_gecersiz_vektor_hatalari():
    """Boyut uyuşmazlığı ve geçersiz norm derecelerinde hata fırlatıldığını denetler."""
    v1 = np.array([1.0, 2.0])
    v2 = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        MesafeOlcer.oklid_mesafesi(v1, v2)

    with pytest.raises(ValueError):
        MesafeOlcer.minkowski_mesafesi(v1, v1, p_derecesi=0.5)


def test_toplu_hesaplama_tutarliligi():
    """Toplu döngüsüz hesaplama ile tekil hesaplamaların birebir örtüştüğünü test eder."""
    sorgu = np.array([1.0, 2.0, 3.0])
    veri_kumesi = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 6.0, 8.0],
        [0.0, 0.0, 0.0]
    ])

    toplu_oklid = MesafeOlcer.toplu_oklid_mesafesi(sorgu, veri_kumesi)
    assert np.isclose(toplu_oklid[0], 0.0)
    assert np.isclose(toplu_oklid[1], MesafeOlcer.oklid_mesafesi(sorgu, veri_kumesi[1]))
    assert np.isclose(toplu_oklid[2], MesafeOlcer.oklid_mesafesi(sorgu, veri_kumesi[2]))


def test_gorsel_eslestirici_katalog_aramasi():
    """Katalog eşleştiricinin en yakın öğeyi ilk sırada bulduğunu test eder."""
    eslestirici = GorselBenzerlikEslestirici(metrik="oklid")
    eslestirici.katalog_ekle("hedef_oge", np.array([10.0, 20.0]))
    eslestirici.katalog_ekle("uzak_oge", np.array([100.0, 200.0]))

    sorgu = np.array([10.5, 20.2])
    sonuclar = eslestirici.en_yakin_k_bul(sorgu, k=1)

    assert len(sonuclar) == 1
    assert sonuclar[0].oge_kimligi == "hedef_oge"
