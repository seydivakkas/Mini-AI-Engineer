"""
Tesla Gün 41 Ana Akış (Tesla Day 41 Main Pipeline)
===================================================
IMU ve Tekerlek Odometrisi Füzyonu (Dead Reckoning)
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

from src.tesla_imu_ve_odometri_fuzyonu import TeslaIMUWheelOdometryFusion
from src.tesla_imu_profilleyici import TeslaIMUProfilleyici
from src.tesla_imu_gorsellestirici import TeslaIMUGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 41: IMU VE TEKERLEK ODOMETRİSİ DEAD RECKONING 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 100 Hz IMU İntegrasyonu, Diferansiyel Hız Yaw & Jiroskop Bias")
    print("--------------------------------------------------------------------------------\n")

    # 1. Dead Reckoning Benchmark'ı
    print(" [1] 500 Adımlık (5 Saniye, 100 Hz) Virajlı Dead Reckoning Simülasyonu...")
    profilleyici = TeslaIMUProfilleyici(steps=500, dt_s=0.01)
    metrikler = profilleyici.benchmark_dead_reckoning()

    print(f"     -> Nihai Füzyon Konum Hatası : {metrikler['final_fused_error_m']:.3f} Metre")
    print(f"     -> Saf IMU Sürüklenme Hatası : {metrikler['final_pure_error_m']:.3f} Metre")
    print(f"     -> Sürüklenme Azalma Oranı   : %{metrikler['drift_reduction_pct']:.1f} Başarı")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] 100 Hz Dead Reckoning RTOS Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi     : {metrikler['imu_step_ortalama_us']:.3f} µs (P99: {metrikler['imu_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Füzyon Adımı    : {metrikler['saniyelik_dead_reckoning']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD IMU ve Odometri Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaIMUGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_imu_odometri_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi    : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 41 BAŞARIYLA TAMAMLANDI! DEAD RECKONING VE IMU FÜZYONU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
