"""
Tesla Gün 82 Ana Akış (Tesla Day 82 Main Pipeline)
===================================================
Tesla Autobidder Algoritmik Enerji Ticareti ve Arbitraj
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

from src.tesla_autobidder_ticaret_motoru import TeslaAutobidderTrader
from src.tesla_autobidder_profilleyici import TeslaAutobidderProfilleyici
from src.tesla_autobidder_gorsellestirici import TeslaAutobidderGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 82: TESLA AUTOBIDDER ENERJİ TİCARETİ & ARBİTRAJ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Spot Piyasa Fiyat Tahmini, Batarya Amortismanı & Kar Maksimizasyonu")
    print("--------------------------------------------------------------------------------\n")

    # 1. Autobidder Benchmark'ı
    print(" [1] 24 Saatlik Spot Piyasa Arbitraj Simülasyonu Başlatılıyor...")
    profilleyici = TeslaAutobidderProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_autobidder()

    print(f"     -> Toplam Brüt Satış Geliri: ${metrikler['revenue_usd']:,.2f} USD")
    print(f"     -> Şarj Elektrik Maliyeti  : -${metrikler['cost_usd']:,.2f} USD")
    print(f"     -> Batarya Yıpranma Kaybı  : -${metrikler['deg_cost_usd']:,.2f} USD ($40/MWh)")
    print(f"     -> Günlük Net Arbitraj Karı: +${metrikler['profit_usd']:,.2f} USD / Gün")
    print(f"     -> Yıllık Tahmini Getiri   : +${metrikler['profit_usd'] * 365:,.0f} USD / Megapack")

    # 2. Karar Hızı
    print("\n [2] Autobidder Arbitraj Karar Verme RTOS Performansı...")
    print(f"     -> Saat Başına Karar Süresi: {metrikler['trading_step_ortalama_us']:.3f} µs (P99: {metrikler['sim_24h_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Karar Hacmi   : {metrikler['saniyelik_karar_kapasitesi']:,} Teklif/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Autobidder Arbitraj Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaAutobidderGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_autobidder_arbitraj_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 82 BAŞARIYLA TAMAMLANDI! AUTOBIDDER ENERJİ TİCARETİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
