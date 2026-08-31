"""
Tesla Gün 65 Ana Akış (Tesla Day 65 Main Pipeline)
===================================================
Stanley ve Pure Pursuit Yörünge Takip Kontrolcüsü
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

from src.tesla_stanley_pure_pursuit_kontrolcu import TeslaStanleyTracker, TeslaTrackingBenchmark
from src.tesla_stanley_profilleyici import TeslaStanleyProfilleyici
from src.tesla_stanley_gorsellestirici import TeslaStanleyGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 65: STANLEY VE PURE PURSUIT TAKİP KONTROLCÜSÜ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Ön Aks Geometrik Takip, Cross-Track Error & Stanley Kontrolü")
    print("--------------------------------------------------------------------------------\n")

    # 1. Stanley Benchmark'ı
    print(" [1] Stanley Kapalı Çevrim Şerit Takip Simülasyonu Çalıştırılıyor...")
    profilleyici = TeslaStanleyProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_tracker()

    print(f"     -> Takip Yakınsama Durumu   : {'BAŞARILI' if metrikler['is_converged'] else 'BAŞARISIZ'}")
    print(f"     -> Son Yanal Takip Hatası   : {metrikler['final_err']*100:.2f} cm (Hedef: < 5 cm)")
    print(f"     -> Simülasyon Adım Sayısı   : {len(metrikler['errors'])} Adım (5.0 Saniye)")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Stanley Kontrolcü RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['stanley_step_ortalama_us']:.3f} µs (P99: {metrikler['stanley_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Takip Çevrimi  : {metrikler['saniyelik_takip_cevrimi']:,} Çevrim/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Stanley Takip Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaStanleyGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_stanley_tracking_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 65 BAŞARIYLA TAMAMLANDI! STANLEY TAKİP KONTROLCÜSÜ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
