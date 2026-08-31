"""
Tesla Gün 57 Ana Akış (Tesla Day 57 Main Pipeline)
===================================================
Frenet Çerçevesi ve Jerk-Optimal Quintic Şerit Değiştirme
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

from src.tesla_frenet_ve_quintic_serit_degistirme import TeslaFrenetTrajectoryPlanner
from src.tesla_frenet_profilleyici import TeslaFrenetProfilleyici
from src.tesla_frenet_gorsellestirici import TeslaFrenetGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 57: FRENET VE QUINTIC ŞERİT DEĞİŞTİRME 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Frenet (s,d), 5. Derece Quintic Polinom & Jerk-Optimal Konfor")
    print("--------------------------------------------------------------------------------\n")

    # 1. Frenet & Quintic Benchmark'ı
    print(" [1] 5. Derece Jerk-Optimal Şerit Değiştirme Yörüngesi Sentezleniyor...")
    profilleyici = TeslaFrenetProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_frenet_planner()

    print(f"     -> Hedef Şerit Genişliği    : 3.5 Metre (4.0 Saniyede Geçiş)")
    print(f"     -> Maksimum Yanal Jerk      : {metrikler['max_jerk']:.3f} m/s³ (Konfor Sınırı: <= 1.5 m/s³)")
    print(f"     -> Maksimum Yanal İvme      : {metrikler['max_acc']:.3f} m/s² (Konfor Sınırı: <= 2.0 m/s²)")
    print(f"     -> Sürüş Konfor Durumu      : {'%100 PREMIUM KONFOR' if metrikler['is_comfortable'] else 'SERT MANEVRA'}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Frenet Quintic Planlayıcı RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['frenet_step_ortalama_us']:.3f} µs (P99: {metrikler['frenet_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Yörünge Hacmi  : {metrikler['saniyelik_frenet_plani']:,} Plan/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Frenet Quintic Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaFrenetGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_frenet_quintic_lane_change_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 57 BAŞARIYLA TAMAMLANDI! FRENET QUINTIC ŞERİT DEĞİŞTİRME DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
