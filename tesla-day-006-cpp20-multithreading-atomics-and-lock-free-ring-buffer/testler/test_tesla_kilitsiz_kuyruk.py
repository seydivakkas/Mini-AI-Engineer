"""
Tesla C++20 Kilitsiz Kuyruk Birim Testleri (PyTest)
==================================================
Bu test paketi; Lock-Free SPSC Halka Kuyrugun FIFO siralamasini,
kapasite tasma korumasini ve eszamanli is parcacigi guvenligini dogrular.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import pytest
import sys
import os
import threading
import time

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_kilitsiz_kuyruk import (
    TeslaTekerlekHizPaketi,
    TeslaSPSCKilitsizHalkaKuyruk
)


def test_spsc_kuyruk_ekleme_ve_alma():
    """Kuyruğa eklenen elemanların FIFO sırasıyla kayıpsız alındığı test edilir."""
    kuyruk = TeslaSPSCKilitsizHalkaKuyruk(kapasite=16)
    p1 = TeslaTekerlekHizPaketi(1, 1000, 50.0, 50.0, 49.8, 49.9)
    p2 = TeslaTekerlekHizPaketi(2, 2000, 51.0, 51.0, 50.8, 50.9)

    assert kuyruk.kuyruga_ekle(p1) is True
    assert kuyruk.kuyruga_ekle(p2) is True

    al_p1 = kuyruk.kuyruktan_al()
    al_p2 = kuyruk.kuyruktan_al()

    assert al_p1 is not None and al_p1.darbe_sayaci == 1
    assert al_p2 is not None and al_p2.darbe_sayaci == 2
    assert kuyruk.kuyruktan_al() is None


def test_spsc_kuyruk_tasma_korumasi():
    """Kapasite dolduğunda taşmanın engellendiği doğrulanır."""
    kuyruk = TeslaSPSCKilitsizHalkaKuyruk(kapasite=4)
    for i in range(4):
        p = TeslaTekerlekHizPaketi(i, i*100, 10.0, 10.0, 10.0, 10.0)
        assert kuyruk.kuyruga_ekle(p) is True

    # 5. ekleme başarısız olmalı (taşma engeli)
    tasma_p = TeslaTekerlekHizPaketi(99, 9900, 10.0, 10.0, 10.0, 10.0)
    assert kuyruk.kuyruga_ekle(tasma_p) is False


def test_spsc_kuyruk_bosken_alma():
    """Boş kuyruktan çekme None dönmelidir."""
    kuyruk = TeslaSPSCKilitsizHalkaKuyruk(kapasite=8)
    assert kuyruk.kuyruktan_al() is None


def test_spsc_eszamanli_uretici_tuketici():
    """Eşzamanlı üretici ve tüketici iş parçacığında veri yarışı olmadan tüm paketler aktarılmalıdır."""
    kuyruk = TeslaSPSCKilitsizHalkaKuyruk(kapasite=1024)
    toplam_paket = 2000
    alinan_paketler = []

    def uretici():
        for i in range(toplam_paket):
            p = TeslaTekerlekHizPaketi(i, time.time_ns(), 60.0, 60.0, 60.0, 60.0)
            while not kuyruk.kuyruga_ekle(p):
                time.sleep(0.0001)

    def tuketici():
        while len(alinan_paketler) < toplam_paket:
            p = kuyruk.kuyruktan_al()
            if p is not None:
                alinan_paketler.append(p)
            else:
                time.sleep(0.0001)

    t_prod = threading.Thread(target=uretici)
    t_cons = threading.Thread(target=tuketici)

    t_prod.start()
    t_cons.start()

    t_prod.join()
    t_cons.join()

    assert len(alinan_paketler) == toplam_paket
    assert [p.darbe_sayaci for p in alinan_paketler] == list(range(toplam_paket))
