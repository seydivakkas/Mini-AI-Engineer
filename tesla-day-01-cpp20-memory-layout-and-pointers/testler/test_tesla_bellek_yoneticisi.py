"""
Tesla Bellek Yoneticisi Birim Testleri (PyTest Suite)
=====================================================
Bu test paketi; Tesla C++20 bellek duzeni, 64-byte cache line hizalamasi,
zero-allocation bellek havuzu ve lock-free halka kuyruk mekanizmalarini test eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import pytest
import time
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_bellek_yoneticisi import (
    TeslaTelemetriPaketi,
    CacheHizaliBellekHavuzu,
    SifirTahsilliHalkaKuyruk
)
from src.tesla_bellek_profilleyici import TeslaBellekProfilleyici



def test_tesla_telemetri_paketi_64_bayt_boyutu():
    """Telemetri paketinin ikili formati tam olarak 64 bayt (1 Cache Line) olmalidir."""
    paket = TeslaTelemetriPaketi(
        paket_id=42,
        zaman_damgasi_ns=time.time_ns(),
        can_id=0x280,
        direksiyon_acisi_rad=0.087,
        arac_hizi_kmh=105.4,
        batarya_gerilimi_v=401.5,
        motor_torku_nm=380.0,
        fren_basinci_bar=0.0,
        kontrol_checksum=0x1234
    )
    baytlar = paket.baytlara_donustur()
    assert len(baytlar) == 64, f"Paket boyutu 64 bayt olmali, ancak {len(baytlar)} bayt cikti!"
    
    # Geri cozum testi
    cozulen = TeslaTelemetriPaketi.baytlardan_coz(baytlar)
    assert cozulen.paket_id == 42
    assert cozulen.can_id == 0x280
    assert pytest.approx(cozulen.arac_hizi_kmh, rel=1e-3) == 105.4
    assert pytest.approx(cozulen.batarya_gerilimi_v, rel=1e-3) == 401.5


def test_cache_hizali_bellek_havuzu_tahsis_ve_serbest_birakma():
    """Bellek havuzundan O(1) tahsis ve iade islemleri dogrulanmalidir."""
    havuz = CacheHizaliBellekHavuzu(blok_sayisi=10, blok_boyutu=64)
    veri = b"TESLA_HW4_FSD_DATA_STREAM" * 2  # 50 bayt
    
    idx0 = havuz.tahsis_et(veri)
    assert idx0 is not None
    assert idx0 in havuz.dolu_indeksler
    assert havuz.doluluk_orani() == 0.1
    
    okunan = havuz.blok_oku(idx0)
    assert okunan is not None
    assert okunan[:len(veri)] == veri
    
    # Serbest birakma
    basarili = havuz.serbest_birak(idx0)
    assert basarili is True
    assert idx0 not in havuz.dolu_indeksler
    assert havuz.doluluk_orani() == 0.0


def test_bellek_havuzu_kapasite_siniri_oom_korumasi():
    """Havuz doldugunda yeni tahsisler None dondurerek OOM hatasini engellemelidir."""
    havuz = CacheHizaliBellekHavuzu(blok_sayisi=3, blok_boyutu=64)
    idx1 = havuz.tahsis_et(b"DATA1")
    idx2 = havuz.tahsis_et(b"DATA2")
    idx3 = havuz.tahsis_et(b"DATA3")
    
    assert idx1 is not None and idx2 is not None and idx3 is not None
    assert havuz.doluluk_orani() == 1.0
    
    # 4. tahsis basarisiz olmali
    idx4 = havuz.tahsis_et(b"DATA4")
    assert idx4 is None


def test_sifir_tahsilli_halka_kuyruk_fifo_davranisi():
    """Halka kuyruk FIFO (Ilk Giren Ilk Cikar) kuralina uymali ve tasma korumasi sunmalidir."""
    kuyruk = SifirTahsilliHalkaKuyruk(kapasite=3)
    assert kuyruk.bos_mu() is True
    
    p1 = TeslaTelemetriPaketi(1, 1000, 0x10, 0.0, 50.0, 400.0, 100.0, 0.0, 1)
    p2 = TeslaTelemetriPaketi(2, 2000, 0x20, 0.1, 60.0, 399.0, 150.0, 0.0, 2)
    p3 = TeslaTelemetriPaketi(3, 3000, 0x30, 0.2, 70.0, 398.0, 200.0, 0.0, 3)
    p4 = TeslaTelemetriPaketi(4, 4000, 0x40, 0.3, 80.0, 397.0, 250.0, 0.0, 4)
    
    assert kuyruk.ekle(p1) is True
    assert kuyruk.ekle(p2) is True
    assert kuyruk.ekle(p3) is True
    assert kuyruk.dolu_mu() is True
    
    # Kapasite asimi
    assert kuyruk.ekle(p4) is False
    assert kuyruk.toplam_tasan == 1
    
    # Cikarma islemi
    c1 = kuyruk.cikar()
    assert c1 is not None and c1.paket_id == 1
    c2 = kuyruk.cikar()
    assert c2 is not None and c2.paket_id == 2
    c3 = kuyruk.cikar()
    assert c3 is not None and c3.paket_id == 3
    assert kuyruk.bos_mu() is True


def test_tesla_bellek_profilleyici_benchmark():
    """Profilleyici benchmark analizini basariyla tamamlamalidir."""
    profilleyici = TeslaBellekProfilleyici(havuz_boyutu=100, ornek_sayisi=50)
    sonuclar = profilleyici.benchmark_tahsis_gecikmesi()
    
    assert "havuz_ortalama_ns" in sonuclar
    assert "heap_ortalama_ns" in sonuclar
    assert sonuclar["havuz_ortalama_ns"] > 0
    assert sonuclar["l1_cache_hit_havuz"] > sonuclar["l1_cache_hit_heap"]
    
    verim = profilleyici.halka_kuyruk_verim_testi()
    assert verim["paket_saniye"] > 1000
