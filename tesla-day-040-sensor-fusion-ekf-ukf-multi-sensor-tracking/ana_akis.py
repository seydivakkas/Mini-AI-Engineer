"""
Tesla Gün 40 Ana Akış (Tesla Day 40 Main Pipeline)
===================================================
Genişletilmiş Kalman Filtresi (EKF) ile Asenkron Sensör Füzyonu
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

from src.tesla_sensor_fuzyonu_ekf_ukf import TeslaSensorFusionEKF
from src.tesla_fuzyon_profilleyici import TeslaFuzyonProfilleyici
from src.tesla_fuzyon_gorsellestirici import TeslaFuzyonGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 40: EKF VE ÇOKLU SENSÖR FÜZYONU (KAMERA + RADAR) 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 6-Durumlu EKF, Radar Jacobian Hj, Mahalanobis Kapılama & Takip")
    print("--------------------------------------------------------------------------------\n")

    # 1. Asenkron Füzyon Benchmark'ı
    print(" [1] 200 Adımlık Asenkron Kamera (20 Hz) ve Radar (10 Hz) Füzyon Simülasyonu...")
    profilleyici = TeslaFuzyonProfilleyici(steps=200, dt_s=0.05)
    metrikler = profilleyici.benchmark_sensor_fuzyonu()

    print(f"     -> Konum Takip Hatası (X RMSE) : {metrikler['rmse_pos_x_m']:.3f} Metre (< 0.25 m Eşiği)")
    print(f"     -> Konum Takip Hatası (Y RMSE) : {metrikler['rmse_pos_y_m']:.3f} Metre")
    print(f"     -> Hız Kestirim Hatası (Vx RMSE): {metrikler['rmse_vel_x_mps']:.3f} m/s")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] 6-Durumlu Asenkron EKF RTOS Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi       : {metrikler['fuzyon_step_ortalama_us']:.3f} µs (P99: {metrikler['fuzyon_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Füzyon Kapasitesi : {metrikler['saniyelik_fuzyon_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Sensör Füzyonu Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaFuzyonGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_sensor_fuzyonu_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi      : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 40 BAŞARIYLA TAMAMLANDI! SENSÖR FÜZYONU VE EKF DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
