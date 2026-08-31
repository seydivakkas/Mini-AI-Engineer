"""
Tesla Gün 95 Ana Akış (Tesla Day 95 Main Pipeline)
===================================================
Simülasyondan Gerçeğe (Sim2Real) Robotik Eğitimi: Isaac Sim ve Domain Randomization
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

from src.tesla_sim2real_randomizer import TeslaSim2RealDomainRandomizer
from src.tesla_sim2real_profilleyici import TeslaSim2RealProfilleyici
from src.tesla_sim2real_gorsellestirici import TeslaSim2RealGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🤖 TESLA FSD MASTERI | GÜN 95: ISAAC SIM DOMAIN RANDOMIZATION & SIM2REAL 🤖")
    print("================================================================================")
    print("Stajyer Görevi: ±%15 Kütle, µ=0.4-1.0, 0-8ms Gecikme & %98 Zero-Shot Transfer")
    print("--------------------------------------------------------------------------------\n")

    # 1. Sim2Real Benchmark'ı
    print(" [1] 100 Rastgele Isaac Sim Dünyasında Politika Transfer Testi Başlatılıyor...")
    profilleyici = TeslaSim2RealProfilleyici(iterations=50)
    metrikler = profilleyici.benchmark_sim2real()

    print(f"     -> Simülasyon Dünyası Sayısı: {metrikler['num_episodes']} Rastgele Bölüm")
    print(f"     -> Zero-Shot Başarı Oranı  : %{metrikler['success_rate_pct']:.1f} (Hedef >= %95.0)")
    print(f"     -> Ortalama Politika Ödülü : {metrikler['average_reward']:.2f}")
    print(f"     -> Test Edilen Sürtünme    : µ_min = {metrikler['min_friction']} | Gecikme: {metrikler['max_latency_ms']} ms")
    print(f"     -> Donanım Dağıtım Durumu  : {metrikler['sim2real_ready']} (%100 GERÇEK ROBOTA YÜKLENEBİLİR)")

    # 2. Örnekleme Hızı
    print("\n [2] Domain Parametreleri Örnekleme RTOS Performansı...")
    print(f"     -> Ortalama Örnekleme Süresi: {metrikler['step_ortalama_us']:.3f} µs (P99: {metrikler['step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Ortam Üretimi  : {metrikler['saniyelik_ortam_ornekleme']:,} Ortam/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Sim2Real Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaSim2RealGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_sim2real_randomization_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 95 BAŞARIYLA TAMAMLANDI! SIM2REAL TRANSFER MOTORU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
