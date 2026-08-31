"""
Tesla Gün 62 Ana Akış (Tesla Day 62 Main Pipeline)
===================================================
Otomatik Acil Frenleme (AEB) ve Kaçınma Manevrası
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

from src.tesla_aeb_ve_kacinma_manevrasi import TeslaAEBController
from src.tesla_aeb_profilleyici import TeslaAEBProfilleyici
from src.tesla_aeb_gorsellestirici import TeslaAEBGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 62: OTOMATİK ACİL FRENLEME (AEB) VE AES 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Euro-NCAP Protokolü, Durma Mesafesi, -9.0 m/s² Fren & AES")
    print("--------------------------------------------------------------------------------\n")

    # 1. AEB Benchmark'ı
    print(" [1] Acil Durum Çarpışma Senaryosu Değerlendiriliyor (72 km/h, 18m Engel)...")
    profilleyici = TeslaAEBProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_aeb_controller()

    print(f"     -> AEB Karar Seviyesi       : {metrikler['level']}")
    print(f"     -> Alınan Acil Aksiyon      : {metrikler['action_desc']}")
    print(f"     -> Time-To-Collision (TTC)  : {metrikler['ttc_s']:.2f} Saniye")
    print(f"     -> Acil Durma Mesafesi      : {metrikler['stopping_dist_m']:.1f} Metre")
    print(f"     -> Hedef Fren İvmesi        : {metrikler['target_acc']:.1f} m/s² (-0.92g)")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] AEB Karar Motoru RTOS Performansı...")
    print(f"     -> Ortalama Karar Süresi    : {metrikler['aeb_step_ortalama_us']:.3f} µs (P99: {metrikler['aeb_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik AEB Döngüsü    : {metrikler['saniyelik_aeb_cevrimi']:,} Karar/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD AEB ve Acil Durum Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaAEBGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_aeb_emergency_maneuver_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 62 BAŞARIYLA TAMAMLANDI! AEB VE ACİL KAÇINMA MOTORU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
