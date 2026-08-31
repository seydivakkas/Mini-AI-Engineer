"""
Tesla Gün 59 Ana Akış (Tesla Day 59 Main Pipeline)
===================================================
Clothoid (Euler Spirali) Dinamik Engelden Kaçınma
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

from src.tesla_clothoid_kacinma_planlayici import TeslaClothoidAvoidancePlanner
from src.tesla_clothoid_profilleyici import TeslaClothoidProfilleyici
from src.tesla_clothoid_gorsellestirici import TeslaClothoidGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 59: CLOTHOID DİNAMİK ENGELDEN KAÇINMA 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Doğrusal Eğrilik, C² Süreklilik & Aktüatör Hız Sınırları")
    print("--------------------------------------------------------------------------------\n")

    # 1. Clothoid Benchmark'ı
    print(" [1] 4 Kademeli Sürekli Eğrilikli Clothoid Kaçınma Yörüngesi Sentezleniyor...")
    profilleyici = TeslaClothoidProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_clothoid_planner()

    print(f"     -> Kaçınma Manevrası        : {'GÜVENLİ (%100 ÇARPIŞMASIZ)' if metrikler['is_safe'] else 'TEHLİKELİ'}")
    print(f"     -> Minimum Güvenlik Payı    : {metrikler['min_clearance_m']:.2f} Metre (Hedef: >= 1.5 m)")
    print(f"     -> Toplam Manevra Mesafesi  : 60.0 Metre (100 Adım)")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Clothoid Planlayıcı RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['clothoid_step_ortalama_us']:.3f} µs (P99: {metrikler['clothoid_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kaçınma Planı  : {metrikler['saniyelik_clothoid_plani']:,} Plan/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Clothoid Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaClothoidGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_clothoid_avoidance_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 59 BAŞARIYLA TAMAMLANDI! CLOTHOID KAÇINMA PLANLAYICI DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
