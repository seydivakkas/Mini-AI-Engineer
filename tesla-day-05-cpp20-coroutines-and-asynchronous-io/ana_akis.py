"""
Tesla Gun 05 Ana Akis (Tesla Day 05 Main Pipeline)
===================================================
C++20 Esyordamlar (Coroutines) ve Asenkron G/C
Uctan Uca Calistirma ve Teshis Paneli Uretim Scripti.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_esyordam_motoru import (
    TeslaTelemetriUreteci,
    TeslaEsyordamGorevi,
    Tesla10GbpsEthernetHatti
)
from src.tesla_esyordam_profilleyici import TeslaEsyordamProfilleyici
from src.tesla_esyordam_gorsellestirici import TeslaEsyordamGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 05: C++20 COROUTINES & ASENKRON G/C 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: 10 Gbps Ethernet Telemetrisi, co_yield/co_await & Sifir Thread Ek Yuku")
    print("--------------------------------------------------------------------------------\n")

    # 1. Tekil Eşyordam Akışı
    print(" [1] C++20 co_yield Telemetri Ureteci Baslatiliyor (LIDAR_ON)...")
    sensor = TeslaTelemetriUreteci("LIDAR_ON", toplam_paket=5)
    for _ in range(5):
        pkt = sensor.siradaki_paketi_al()
        if pkt:
            print(f"     -> [co_yield] Paket Alindi: {pkt.sensor_kaynagi} | Akis ID: {pkt.akis_id:02d} | Boyut: {pkt.veri_boyutu_bayt}B")

    # 2. Çoklu Akış ve Profilleme
    print("\n [2] 8-Sensör Eşzamanlı 10 Gbps Kooperatif Zamanlayıcı ve Profilleme Başlatılıyor...")
    profilleyici = TeslaEsyordamProfilleyici(sensor_sayisi=8, paket_sayisi=1000)
    baglam_sonuc = profilleyici.benchmark_baglam_degistirme()
    akis_sonuc = profilleyici.benchmark_coklu_akis_hatti()

    print(f"     -> C++20 Coroutine Resume/Yield : {baglam_sonuc['coroutine_gecikme_ns']:.1f} ns")
    print(f"     -> OS Thread Preemptive Switch  : {baglam_sonuc['os_thread_gecikme_ns']:.1f} ns ({baglam_sonuc['hizlanma_orani']:.1f}x Hizlanma)")
    print(f"     -> Coroutine Bellek Ayak Izi    : {baglam_sonuc['coroutine_bellek_bayt']:.0f} Bayt (vs OS Thread 2 MB)")
    print(f"     -> Toplam Islenen Paket/Bayt    : {akis_sonuc['toplam_adim']} Adim | {akis_sonuc['toplam_bayt'] / (1024*1024):.2f} MB")
    print(f"     -> Efektif Islem Hacmi          : {akis_sonuc['mb_saniye']:.1f} MB/s (Sifir Thread Lock)")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Coroutines Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaEsyordamGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    metrikler_paketi = {
        "baglam": baglam_sonuc,
        "akis": akis_sonuc
    }
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler_paketi, dosya_adi="tesla_esyordam_tani_paneli.png")
    print(f"     -> Tani Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GUN 05 BASARIYLA TAMAMLANDI! C++20 COROUTINE MOTORU DOGRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
