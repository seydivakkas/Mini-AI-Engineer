"""
Tesla Gün 26 Ana Akış (Tesla Day 26 Main Pipeline)
===================================================
Hücre Dengeleme Algoritmaları: Pasif ve Aktif Dengeleme Kontrolü
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

from src.tesla_hucre_dengeleme_kontrolcusu import (
    BatteryCell,
    TeslaBalancingController,
    BalancingStrategy
)
from src.tesla_dengeleme_profilleyici import TeslaDengelemeProfilleyici
from src.tesla_dengeleme_gorsellestirici import TeslaDengelemeGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 26: HÜCRE DENGELEME KONTROLCÜSÜ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Pasif Bleeding Dirençleri, Aktif Endüktif Enerji Aktarımı & ΔV")
    print("--------------------------------------------------------------------------------\n")

    # 1. 12S Batarya Modülü Dengeleme Benchmark'ı
    print(" [1] 12S Hücre Grubu İçin Pasif vs Aktif Dengeleme Simülasyonu...")
    profilleyici = TeslaDengelemeProfilleyici(num_cells=12)
    metrikler = profilleyici.benchmark_dengeleme()

    print(f"     -> Pasif Dengeleme Süresi   : {metrikler['passive_duration_s']} sn ({metrikler['passive_duration_s']/60:.1f} dakika)")
    print(f"     -> Aktif Dengeleme Süresi   : {metrikler['active_duration_s']} sn ({metrikler['active_duration_s']/60:.1f} dakika)")
    print(f"     -> Aktif Dengeleme Hızlanması: {metrikler['speedup_factor']:.2f}x Daha Hızlı!")
    print(f"     -> Pasif Yakılan Isı Enerjisi: {metrikler['passive_total_heat_j']:.1f} Joule")
    print(f"     -> Aktif Harcanan Isı Enerjisi: {metrikler['active_total_heat_j']:.1f} Joule ({metrikler['heat_saving_factor']:.1f}x Isı Tasarrufu!)")

    # 2. Gerçek Zamanlı RTOS Gecikme Analizi
    print("\n [2] Dengeleme Karar Döngüsü RTOS Performansı...")
    print(f"     -> Ortalama Döngü Süresi   : {metrikler['dengeleme_step_ortalama_us']:.3f} µs (P99: {metrikler['dengeleme_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Karar Hacmi   : {metrikler['saniyelik_dengeleme_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Hücre Dengeleme Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaDengelemeGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_hucre_dengeleme_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 26 BAŞARIYLA TAMAMLANDI! HÜCRE DENGELEME & TERMAL KONTROL DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
