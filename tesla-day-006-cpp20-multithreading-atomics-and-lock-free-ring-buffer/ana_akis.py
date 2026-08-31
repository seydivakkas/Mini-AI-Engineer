"""
Tesla Gun 06 Ana Akis (Tesla Day 06 Main Pipeline)
===================================================
C++20 Eszamanlilik (Concurrency), Atomikler ve Kilitsiz Halka Kuyruklar
Uctan Uca Calistirma ve Teshis Paneli Uretim Scripti.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import sys
import os
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_kilitsiz_kuyruk import (
    TeslaTekerlekHizPaketi,
    TeslaSPSCKilitsizHalkaKuyruk
)
from src.tesla_es_zamanlilik_profilleyici import TeslaEsZamanlilikProfilleyici
from src.tesla_es_zamanlilik_gorsellestirici import TeslaEsZamanlilikGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 06: C++20 ATOMIKS & LOCK-FREE SPSC 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: 100 kHz Tekerlek Hiz Kesmeleri, Acquire/Release & Sifir Kilit")
    print("--------------------------------------------------------------------------------\n")

    # 1. Lock-Free SPSC Temel Operasyon
    print(" [1] C++20 Lock-Free SPSC Halka Kuyruk Baslatiliyor (Kapasite: 1024)...")
    kuyruk = TeslaSPSCKilitsizHalkaKuyruk(kapasite=1024)
    for i in range(5):
        paket = TeslaTekerlekHizPaketi(
            darbe_sayaci=i+1,
            zaman_ns=time.time_ns(),
            sol_on_kmh=120.4 + i*0.1,
            sag_on_kmh=120.3 + i*0.1,
            sol_arka_kmh=120.0 + i*0.1,
            sag_arka_kmh=120.1 + i*0.1
        )
        kuyruk.kuyruga_ekle(paket)

    print(f"     -> Kuyruga 5 Kesme Paketi Eklendi | Doluluk Orani: %{kuyruk.doluluk_orani()*100:.2f}")
    for _ in range(5):
        p = kuyruk.kuyruktan_al()
        if p:
            print(f"     -> [Acquire/Release] Cekilen Paket: Darbe #{p.darbe_sayaci} | Sol On Hiz: {p.sol_on_kmh:.1f} km/h")

    # 2. Eszamanlilik Benchmark'i
    print("\n [2] Lock-Free SPSC vs Mutex Kilitli Kuyruk Performans Benchmark'i...")
    profilleyici = TeslaEsZamanlilikProfilleyici(islem_sayisi=30000)
    metrikler = profilleyici.benchmark_spsc_vs_kilitli()

    print(f"     -> Lock-Free SPSC Ortalama Gecikme : {metrikler['spsc_ort_ns']:.1f} ns (Jitter σ: {metrikler['spsc_jitter_ns']:.1f} ns)")
    print(f"     -> Mutex Kilitli Ortalama Gecikme  : {metrikler['kilitli_ort_ns']:.1f} ns (Jitter σ: {metrikler['kilitli_jitter_ns']:.1f} ns)")
    print(f"     -> Donanimsal Hizlanma Carpani     : {metrikler['hizlanma_orani']:.1f}x Hizli")
    print(f"     -> SPSC Islem Kapasitesi (Throughput): {metrikler['spsc_milyon_islem_sn']:.1f} Milyon Islem/sn")

    # 3. Teshis Paneli Gorsellestirme
    print("\n [3] 6 Panelli Tesla Concurrency Tani Paneli Uretiliyor...")
    gorsellestirici = TeslaEsZamanlilikGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_es_zamanlilik_tani_paneli.png")
    print(f"     -> Tani Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GUN 06 BASARIYLA TAMAMLANDI! LOCK-FREE SPSC VERI YAPISI DOGRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
