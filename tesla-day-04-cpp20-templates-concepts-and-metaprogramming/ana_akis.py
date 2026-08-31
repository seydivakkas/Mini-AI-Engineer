"""
Tesla Gun 04 Ana Akis (Tesla Day 04 Main Pipeline)
===================================================
C++20 Sablonlar (Templates), Kavramlar (Concepts) ve Derleme Zamani Meta-Programlama
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

from src.tesla_kavramlar_meta import (
    ConstexprCRC32,
    TeslaSensorPaketiKavrami,
    TeslaBataryaTelemetrisi,
    TeslaMotorTelemetrisi,
    GecersizPaketOrnegi,
    TeslaTipGuvenliSerilestirici
)
from src.tesla_kavram_profilleyici import TeslaKavramProfilleyici
from src.tesla_kavram_gorsellestirici import TeslaKavramGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 04: C++20 CONCEPTS & METAPROGRAMLAMA 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: Derleme Zamani Tur Guvenligi, Requires Kisitlamalari & Constexpr CRC32")
    print("--------------------------------------------------------------------------------\n")

    # 1. Concept Dogrulama Testi
    print(" [1] C++20 Concept Dogrulama Motoru Calistiriliyor...")
    batarya = TeslaBataryaTelemetrisi(
        can_id=0x150,
        zaman_damgasi_ns=time.time_ns(),
        paket_gerilimi_v=403.2,
        akim_amper=-85.4,
        sicaklik_c=32.0,
        sarj_orani_soc=88.2
    )
    b_gecerli, b_msg = TeslaSensorPaketiKavrami.dogrula(batarya)
    print(f"     -> [Batarya Telemetrisi] : {b_msg}")

    gecersiz = GecersizPaketOrnegi(veri_metni="EKSİK_ID", deger=50.0)
    g_gecerli, g_msg = TeslaSensorPaketiKavrami.dogrula(gecersiz)
    print(f"     -> [Gecersiz Paket     ] : {g_msg}")

    # 2. Tip Guvenli Serilestirme ve CRC32 Ekleme
    print("\n [2] Constexpr CRC32 ile Serilestirme...")
    serilestirici = TeslaTipGuvenliSerilestirici()
    seri_veri = serilestirici.serilestir_ve_crc_ekle(batarya)
    print(f"     -> Serilestirilmis CAN-FD Paketi Boyutu: {len(seri_veri)} Bayt (Son 4 Bayt: CRC32)")

    # 3. Profilleme ve Performans Benchmark'i
    print("\n [3] Constexpr vs Naive CRC-32 ve Serilestirme Benchmark Baslatiliyor...")
    profilleyici = TeslaKavramProfilleyici(dongu_sayisi=10000)
    crc_sonuclari = profilleyici.benchmark_constexpr_vs_naive_crc32()
    seri_sonuclari = profilleyici.benchmark_serilestirme_verimi()

    print(f"     -> Constexpr Tablolu CRC-32 : {crc_sonuclari['constexpr_tablolu_ns']:.1f} ns")
    print(f"     -> Naive Bitwise CRC-32     : {crc_sonuclari['naive_bitwise_ns']:.1f} ns ({crc_sonuclari['hizlanma_orani']:.1f}x Hizlanma)")
    print(f"     -> Serilestirme Gecikmesi   : {seri_sonuclari['ortalama_ns']:.1f} ns (P99: {seri_sonuclari['p99_ns']:.1f} ns)")
    print(f"     -> Saniyedeki Paket Hacmi   : {seri_sonuclari['paket_saniye']:,.0f} Paket/sn")

    # 4. Teshis Paneli Gorsellestirme
    print("\n [4] 6 Panelli Tesla Concepts Tani Paneli Uretiliyor...")
    gorsellestirici = TeslaKavramGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    metrikler_paketi = {
        "crc_metrik": crc_sonuclari,
        "seri_metrik": seri_sonuclari
    }
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler_paketi, dosya_adi="tesla_kavram_tani_paneli.png")
    print(f"     -> Tani Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GUN 04 BASARIYLA TAMAMLANDI! C++20 CONCEPTS CEKIRDEGI DOGRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
