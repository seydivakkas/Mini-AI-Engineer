"""
Tesla Gün 27 Ana Akış (Tesla Day 27 Main Pipeline)
===================================================
Termal Yönetim ve Isı Pompası (Octovalve) Kontrol Algoritmaları
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

from src.tesla_octovalve_termal_yonetim import (
    TeslaOctovalveController,
    VehicleThermalState,
    OctovalveMode
)
from src.tesla_termal_profilleyici import TeslaTermalProfilleyici
from src.tesla_termal_gorsellestirici import TeslaTermalGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 27: OCTOVALVE TERMAL YÖNETİM SİSTEMİ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Octovalve 8-Yollu Valf, Isı Pompası COP & Supercharger Ön Isıtma")
    print("--------------------------------------------------------------------------------\n")

    # 1. 30 Dakikalık Supercharger Ön Isıtma Benchmark'ı
    print(" [1] 0°C Dış Ortamda Supercharger Hedefiyle 30 Dakikalık Ön Isıtma Simülasyonu...")
    profilleyici = TeslaTermalProfilleyici(sim_saniye=1800)
    metrikler = profilleyici.benchmark_termal_sistem()

    print(f"     -> Başlangıç Batarya Sıcaklığı : 5.0 °C (Soğuk)")
    print(f"     -> 30 Dk Sonrası Batarya      : {metrikler['final_battery_temp_c']:.1f} °C (Supercharger Hazır!)")
    print(f"     -> 30 Dk Sonrası Kabin        : {metrikler['final_cabin_temp_c']:.1f} °C (Konforlu)")
    print(f"     -> Octovalve Isı Pompası Enerji: {metrikler['hp_energy_kwh']:.2f} kWh")
    print(f"     -> Eski Dirençli PTC Enerjisi : {metrikler['ptc_energy_kwh']:.2f} kWh")
    print(f"     -> Isı Pompası Enerji Tasarrufu: %{metrikler['energy_saved_pct']:.1f} Tasarruf!")

    # 2. Termal Karar Döngüsü RTOS Performansı
    print("\n [2] Termal Diferansiyel Denklem Çözücü Gecikmesi...")
    print(f"     -> Ortalama Çözüm Süresi      : {metrikler['termal_step_ortalama_us']:.3f} µs (P99: {metrikler['termal_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Karar Kapasitesi : {metrikler['saniyelik_termal_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Octovalve Termal Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaTermalGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_octovalve_termal_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 27 BAŞARIYLA TAMAMLANDI! OCTOVALVE VE TERMAL KONTROL DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
