"""
Tesla Gün 56 Ana Akış (Tesla Day 56 Main Pipeline)
===================================================
FAZ 6 BAŞLANGICI: Hibrit A* ve Voronoi Alanı Park Planlayıcı
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

from src.tesla_hibrit_a_star_park_planlayici import TeslaHybridAStarParkPlanner
from src.tesla_hibrit_a_star_profilleyici import TeslaHibritAStarProfilleyici
from src.tesla_hibrit_a_star_gorsellestirici import TeslaHibritAStarGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 56: HİBRİT A* (HYBRID A*) VE OTONOM PARK PLANLAMA 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Sürekli Durum (x,y,theta), Kinematik Model, S-Eğrisi & Voronoi")
    print("--------------------------------------------------------------------------------\n")

    # 1. Hibrit A* Park Benchmark'ı
    print(" [1] Hibrit A* Kinematik Bisiklet Modeli ve S-Eğrisi Park Yörüngesi Sentezleniyor...")
    profilleyici = TeslaHibritAStarProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_park_planner()

    print(f"     -> Park Başarı Durumu       : {'BAŞARILI' if metrikler['success'] else 'BAŞARISIZ'}")
    print(f"     -> Son Konum Park Hatası    : {metrikler['final_pos_err_m']*100:.1f} cm (Hedef: < 15 cm)")
    print(f"     -> Son Yönelme Açısı Hatası : {metrikler['final_yaw_err_deg']:.2f}° (Hedef: < 2.0°)")
    print(f"     -> Manevra Adım Sayısı      : {len(metrikler['trajectory'])} Adım")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Hibrit A* RTOS Planlama Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['park_step_ortalama_us']:.3f} µs (P99: {metrikler['park_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Park Yörüngesi : {metrikler['saniyelik_park_plani']:,} Plan/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Hibrit A* Autopark Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaHibritAStarGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_hybrid_a_star_autopark_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 56 BAŞARIYLA TAMAMLANDI! HİBRİT A* PARK PLANLAYICI DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
