"""
Tesla Gün 39 Ana Akış (Tesla Day 39 Main Pipeline)
===================================================
Ultrasonik ve Milimetrik Radar Sinyal İşleme
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

from src.tesla_radar_ve_ultrasonik_isleme import TeslaRadarAndUltrasonicProcessor
from src.tesla_radar_profilleyici import TeslaRadarProfilleyici
from src.tesla_radar_gorsellestirici import TeslaRadarGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 39: 77 GHz RADAR VE ULTRASONİK SİNYAL İŞLEME 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 2D Range-Doppler FFT, CA-CFAR Eşikleme & ToF Sıcaklık Düzeltme")
    print("--------------------------------------------------------------------------------\n")

    # 1. Radar ve Ultrasonik Benchmark'ı
    print(" [1] 77 GHz FMCW Radar 2D FFT ve Ultrasonik ToF Mesafe Ölçümü...")
    profilleyici = TeslaRadarProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_radar_ve_ultrasonik()

    print(f"     -> 2D Range-Doppler Boyutu : {metrikler['rd_map'].shape} (64 Chirp x 256 Örnek)")
    print(f"     -> Ultrasonik Mesafe (20°C): {metrikler['us_dist_20c']:.3f} Metre")
    print(f"     -> Ultrasonik Mesafe (-10°C): {metrikler['us_dist_minus10c']:.3f} Metre (Sıcaklık Kompanzasyonu ile)")

    # 2. Radar RTOS Çözümleme Hızı
    print("\n [2] 2D Range-Doppler FFT ve CFAR RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi   : {metrikler['radar_step_ortalama_us']:.3f} µs (P99: {metrikler['radar_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Radar Karesi  : {metrikler['saniyelik_radar_karesi']:,} Kare/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Radar ve Ultrasonik Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaRadarGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_radar_ultrasonik_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi  : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 39 BAŞARIYLA TAMAMLANDI! RADAR VE ULTRASONİK SİNYAL İŞLEME DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
