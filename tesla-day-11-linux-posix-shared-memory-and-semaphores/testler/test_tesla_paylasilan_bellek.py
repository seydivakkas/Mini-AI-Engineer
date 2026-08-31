"""
Tesla POSIX Paylasilan Bellek Birim Testleri (PyTest)
=====================================================
Bu test paketi; Linux POSIX Shared Memory olusturmasini, semafor tabanli
senkronizasyonu ve uctan uca sifir-kopyalama veri butunlugunu dogrular.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import pytest
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_paylasilan_bellek import (
    TeslaPOSIXPaylasilanBellek,
    TeslaPOSIXSemafor,
    TeslaSifirKopyaGoruntuHatti
)


def test_shm_olusturma_ve_mmap():
    """Paylaşılan bellek oluşturma ve mmap haritalaması doğrulanır."""
    shm = TeslaPOSIXPaylasilanBellek("/test_shm", 1024)
    assert shm.shm_open_ve_mmap() is True
    assert shm.eslendi_mi is True

    veri = b"TESLA_FSD_DATA"
    assert shm.sifir_kopya_yaz(0, veri) is True
    
    gorunum = shm.sifir_kopya_oku_gorunumu(0, len(veri))
    assert bytes(gorunum) == veri

    shm.shm_unlink()
    assert shm.eslendi_mi is False


def test_semafor_senkronizasyonu():
    """Semafor wait/post mantığı doğrulanır."""
    sem = TeslaPOSIXSemafor("/test_sem", baslangic_degeri=0)
    assert sem.bekle_sem_wait() is False  # 0 olduğu için geçemez

    sem.sinyal_ver_sem_post()
    assert sem.deger == 1
    assert sem.bekle_sem_wait() is True   # Şimdi geçer ve 0 yapar
    assert sem.deger == 0


def test_sifir_kopya_goruntu_akti_butunlugu():
    """Kamera üretici ve FSD tüketici süreçleri arasında sıfır kopyalama bütünlüğü test edilir."""
    hat = TeslaSifirKopyaGoruntuHatti(frame_boyutu_bayt=100)
    
    test_frame = b"\x42" * 100
    assert hat.uretici_kamera_frame_yaz(test_frame) is True

    alinan_gorunum = hat.tuketici_fsd_frame_oku_gorunumu()
    assert alinan_gorunum is not None
    assert bytes(alinan_gorunum) == test_frame


def test_shm_sinir_kontrolu():
    """Sınır aşımı durumunda yazma ve okumanın güvenle engellendiği doğrulanır."""
    shm = TeslaPOSIXPaylasilanBellek("/test_sinir_shm", 64)
    shm.shm_open_ve_mmap()

    veri_buyuk = b"\x00" * 128
    assert shm.sifir_kopya_yaz(0, veri_buyuk) is False
    assert len(shm.sifir_kopya_oku_gorunumu(0, 128)) == 0
