"""
Tesla Gün 66 Ana Akış (Tesla Day 66 Main Pipeline)
===================================================
FAZ 6 BÜYÜK CAPSTONE: C++ ile Otonom Otoyol Şerit Değiştirme & MPC Yörünge Takipçisi
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

from src.tesla_faz6_capstone_planner_controller import TeslaFullStackFSDPlannerController
from src.tesla_faz6_capstone_profilleyici import TeslaFaz6CapstoneProfilleyici
from src.tesla_faz6_capstone_gorsellestirici import TeslaFaz6CapstoneGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 66: FAZ 6 BÜYÜK CAPSTONE (PLANNER & CONTROLLER) 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Quintic Şerit Değişimi, MPC/Stanley Takip, AEB/AES & ASIL-D")
    print("--------------------------------------------------------------------------------\n")

    # 1. Faz 6 Capstone Benchmark'ı
    print(" [1] Full-Stack FSD Otonom Sürüş Motoru Simüle Ediliyor...")
    profilleyici = TeslaFaz6CapstoneProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_pipeline()
    c = metrikler["ciktilar"]

    print(f"     -> Otoyol Seyir Hızı        : 90.0 km/h (25.0 m/s)")
    print(f"     -> Şerit Değiştirme Hedefi  : 3.50 m (4.0 Saniyede Jerk-Optimal Geçiş)")
    print(f"     -> Maksimum Yanal Jerk      : {c['max_jerk']:.2f} m/s³ (Limit: <= 3.50 m/s³)")
    print(f"     -> Son Yanal Takip Hatası   : {c['final_lat_err_m']*100:.2f} cm (Hedef: < 8.0 cm)")
    print(f"     -> Son Gövde Açı Hatası     : {c['final_yaw_err_deg']:.2f}° (Hedef: < 1.50°)")
    print(f"     -> Euro-NCAP AEB Durumu     : {c['aeb_status']} (TTC: {c['ttc_s']:.1f}s)")
    print(f"     -> ISO 26262 ASIL-D Durumu  : {'ONAYLANDI (SAĞLAM)' if c['asil_d_verified'] else 'HATA'}")
    print(f"     -> FSD Çift Düğüm (Node A/B): {'TAM UZLAŞI (CONSENSUS)' if c['arbiter_consensus'] else 'AYRIŞMA'}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Full-Stack FSD Motoru RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['capstone_step_ortalama_us']:.3f} µs (P99: {metrikler['capstone_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik FSD Döngüsü    : {metrikler['saniyelik_fsd_döngüsü']:,} Döngü/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Faz 6 Capstone Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaFaz6CapstoneGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_faz6_capstone_fsd_planner_controller_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🏆 TEBRİKLER! FAZ 6 BÜYÜK CAPSTONE %100 BAŞARIYLA TAMAMLANDI! 🏆")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
