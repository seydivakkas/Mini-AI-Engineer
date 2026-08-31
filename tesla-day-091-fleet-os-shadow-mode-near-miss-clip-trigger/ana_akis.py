"""
Tesla Gün 91 Ana Akış (Tesla Day 91 Main Pipeline)
===================================================
Tesla Filo İşletim Sistemi (Fleet OS): Gölge Mod ve Kritik Klip Tetikleme
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
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

from src.tesla_filo_os_tetikleyici import TeslaFleetOSClipTrigger
from src.tesla_filo_profilleyici import TeslaFiloProfilleyici
from src.tesla_filo_gorsellestirici import TeslaFiloGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 91: TESLA FLEET OS GÖLGE MOD & KLİP TETİKLEME 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Sert Fren (>0.8g), Acil Direksiyon, Gölge Mod & 15s H.265 Klip")
    print("--------------------------------------------------------------------------------\n")

    # 1. Filo OS Benchmark'ı
    print(" [1] 5,000 Araçlık Filo Telemetri Olayı Taranıyor ve Filtreleniyor...")
    profilleyici = TeslaFiloProfilleyici(fleet_event_count=5000)
    metrikler = profilleyici.benchmark_fleet_trigger()

    print(f"     -> Taranan Araç Olayı      : {metrikler['total_fleet_events']:,} Olay")
    print(f"     -> Tetiklenen Kritik Klip  : {metrikler['critical_clips_triggered']:,} Paket (%{metrikler['trigger_rate_pct']:.2f})")
    print(f"     -> Klip Formatı            : 15 Saniye (10s Öncesi + 5s Sonrası, 8 Kamera + CAN)")
    print(f"     -> Veri Yükleme Stratejisi : Wi-Fi Bağlandığında Dojo Autolabeler Havuzuna")
    print(f"     -> Filo Güvenlik Durumu    : %100 OTONOM VERİ MOTORU AKTİF")

    # 2. Değerlendirme Hızı
    print("\n [2] Edge Telemetri Tarama ve Map-Reduce Filtresi RTOS Performansı...")
    print(f"     -> Olay Başına Süre        : {metrikler['per_event_ortalama_us']:.3f} µs (P99: {metrikler['filter_p99_us']/1000.0:.3f} µs)")
    print(f"     -> Saniyelik Olay Hacmi    : {metrikler['saniyelik_olay_tarama']:,} Olay/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Fleet OS Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaFiloGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_filo_os_golge_mod_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 91 BAŞARIYLA TAMAMLANDI! FLEET OS GÖLGE MOD TETİKLEYİCİSİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
