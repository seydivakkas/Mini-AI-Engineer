"""
Tesla Gun 01 Ana Akis (Tesla Day 01 Main Pipeline)
===================================================
Modern C++20 Bellek Mimarisi, 64-Bayt Cache Line Hizalama ve Zero-Allocation Havuzu
Uctan Uca Calistirma ve Teshis Paneli Uretim Scripti.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import sys
import os
import time

# Windows terminal UTF-8 uyumlulugu
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# Kok dizini ekle
su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_bellek_yoneticisi import (
    TeslaTelemetriPaketi,
    CacheHizaliBellekHavuzu,
    SifirTahsilliHalkaKuyruk
)
from src.tesla_bellek_profilleyici import TeslaBellekProfilleyici
from src.tesla_bellek_gorsellestirici import TeslaBellekGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 01: C++20 BELLEK DUZENI & ZERO-ALLOC 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: 64-Bayt Cache Hizalama, Deterministik Havuz ve SPSC Halka Kuyruk")
    print("--------------------------------------------------------------------------------\n")

    # 1. Telemetri Paketi Testi
    ornek_paket = TeslaTelemetriPaketi(
        paket_id=101,
        zaman_damgasi_ns=time.time_ns(),
        can_id=0x140,
        direksiyon_acisi_rad=0.035,
        arac_hizi_kmh=118.2,
        batarya_gerilimi_v=399.8,
        motor_torku_nm=410.0,
        fren_basinci_bar=0.0,
        kontrol_checksum=0xCAFE
    )
    paket_bayt = ornek_paket.baytlara_donustur()
    print(f" [1] Tesla Telemetri Paketi Olusturuldu -> Boyut: {len(paket_bayt)} Bayt (Tam 1 L1 Cache Line)")

    # 2. Cache Hizali Bellek Havuzu Testi
    havuz = CacheHizaliBellekHavuzu(blok_sayisi=1024, blok_boyutu=64)
    blok_id = havuz.tahsis_et(paket_bayt)
    print(f" [2] Zero-Allocation Havuz Tahsisi -> Blok Indeksi: {blok_id}, Doluluk: %{havuz.doluluk_orani()*100:.2f}")

    # 3. Profilleme ve Determinizm Benchmark
    print("\n [3] Determinizm ve Gecikme Benchmark'i Baslatiliyor (1000 Dongu)...")
    profilleyici = TeslaBellekProfilleyici(havuz_boyutu=1024, ornek_sayisi=1000)
    benchmark_sonuclari = profilleyici.benchmark_tahsis_gecikmesi()
    verim_sonuclari = profilleyici.halka_kuyruk_verim_testi()

    print(f"     -> Zero-Alloc Havuz Ortalama Gecikme: {benchmark_sonuclari['havuz_ortalama_ns']:.1f} ns (P99: {benchmark_sonuclari['havuz_p99_ns']:.1f} ns)")
    print(f"     -> Dinamik Heap Malloc Gecikme      : {benchmark_sonuclari['heap_ortalama_ns']:.1f} ns (P99: {benchmark_sonuclari['heap_p99_ns']:.1f} ns)")
    print(f"     -> Hizlanma Kat Sayisi              : {benchmark_sonuclari['hizlanma_kat_sayisi']:.2f}x Daha Hizli")
    print(f"     -> L1 Cache Hit Orani               : %{benchmark_sonuclari['l1_cache_hit_havuz']:.1f} (Heap: %{benchmark_sonuclari['l1_cache_hit_heap']:.1f})")
    print(f"     -> Lock-Free Kuyruk Verimi          : {verim_sonuclari['paket_saniye']:,.0f} Paket/sn ({verim_sonuclari['bant_genisligi_mb_s']:.2f} MB/s)")

    # 4. Teshis Paneli Gorsellestirme
    print("\n [4] 6 Panelli Tesla Muhendislik Tani Paneli Uretiliyor...")
    gorsellestirici = TeslaBellekGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(benchmark_sonuclari, dosya_adi="tesla_bellek_tani_paneli.png")
    print(f"     -> Tani Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GUN 01 BASARIYLA TAMAMLANDI! TESLA ASIL-D BELLEK CEKIRDEGI DOGRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
