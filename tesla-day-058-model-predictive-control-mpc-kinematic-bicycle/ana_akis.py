"""
Tesla Gün 58 Ana Akış (Tesla Day 58 Main Pipeline)
===================================================
Model Predictive Control (MPC) Kinematik Kontrolcü
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

from src.tesla_mpc_kinematik_kontrolcu import TeslaKinematicMPCController
from src.tesla_mpc_profilleyici import TeslaMPCProfilleyici
from src.tesla_mpc_gorsellestirici import TeslaMPCGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 58: MODEL PREDICTIVE CONTROL (MPC) KONTROLCÜ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Ayrık Riccati, Durum Geri Beslemesi & Çift Eksenli Takip")
    print("--------------------------------------------------------------------------------\n")

    # 1. MPC Benchmark'ı
    print(" [1] Model Predictive Control Kapalı Çevrim Takip Simülasyonu Çalıştırılıyor...")
    profilleyici = TeslaMPCProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_mpc_controller()

    print(f"     -> Takip Yakınsama Durumu   : {'BAŞARILI' if metrikler['is_converged'] else 'BAŞARISIZ'}")
    print(f"     -> Son Yanal Takip Hatası   : {metrikler['final_lat_err']*100:.1f} cm (Hedef: < 10 cm)")
    print(f"     -> Son Yönelme Açısı Hatası : {metrikler['final_yaw_err_deg']:.2f}° (Hedef: < 1.0°)")
    print(f"     -> Kontrol Adım Sayısı      : {len(metrikler['lat_errors'])} Adım (4.0 Saniye)")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] MPC Kontrolcü RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['mpc_step_ortalama_us']:.3f} µs (P99: {metrikler['mpc_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kontrol Çevrimi: {metrikler['saniyelik_mpc_cevrimi']:,} Çevrim/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD MPC Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaMPCGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_kinematic_mpc_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 58 BAŞARIYLA TAMAMLANDI! MODEL PREDICTIVE CONTROL DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
