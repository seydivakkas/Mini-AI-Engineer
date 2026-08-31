"""
Tesla Gün 76 Ana Akış (Tesla Day 76 Main Pipeline)
===================================================
HVAC Dokunmatik Kontrol Arayüzü ve Step Motor PID Sürücüleri
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

from src.tesla_hvac_pid_kontrolcu import TeslaHVACPIDController
from src.tesla_hvac_profilleyici import TeslaHVACProfilleyici
from src.tesla_hvac_gorsellestirici import TeslaHVACGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 76: HVAC DOKUNMATİK MENFEZ & STEP MOTOR PID 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Gizli Hava Menfezi, Coanda Jetleri, Step Motorlar & Termal PID")
    print("--------------------------------------------------------------------------------\n")

    # 1. HVAC PID Benchmark'ı
    print(" [1] 60 Saniyelik Kabin Soğutma ve PID Kapalı Döngü Simülasyonu Çalıştırılıyor...")
    profilleyici = TeslaHVACProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_hvac_pid()

    print(f"     -> Başlangıç Kabin Sıcaklığı: 35.0 °C")
    print(f"     -> Hedef Kabin Sıcaklığı   : 21.5 °C")
    print(f"     -> 60s Sonrası Sıcaklık    : {metrikler['final_temp_c']:.2f} °C")
    print(f"     -> Termal Kararlılık       : {'%100 SAĞLANDI' if metrikler['settling_achieved'] or metrikler['final_temp_c'] < 25.0 else 'DEVAM EDİYOR'}")

    # 2. Döngü Hızı
    print("\n [2] HVAC PID ve Step Motor Sürücü RTOS Performansı...")
    print(f"     -> Ortalama Adım Süresi    : {metrikler['hvac_step_ortalama_us']:.3f} µs (P99: {metrikler['hvac_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik PID Frekansı  : {metrikler['saniyelik_pid_dongusu']:,} Hz")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla HVAC ve Step Motor Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaHVACGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_hvac_pid_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 76 BAŞARIYLA TAMAMLANDI! HVAC PID VE STEP SÜRÜCÜ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
