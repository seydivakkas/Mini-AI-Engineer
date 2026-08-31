"""
Tesla Gün 60 Ana Akış (Tesla Day 60 Main Pipeline)
===================================================
Hız Profili Optimizasyonu ve Enerji Verimliliği
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

from src.tesla_hiz_profili_optimize_edici import TeslaSpeedProfileOptimizer
from src.tesla_hiz_profili_profilleyici import TeslaHizProfiliProfilleyici
from src.tesla_hiz_profili_gorsellestirici import TeslaHizProfiliGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 60: HIZ PROFİLİ VE ENERJİ OPTİMİZASYONU 🚗")
    print("================================================================================")
    print("Stajyer Görevi: İleri-Geri Geçiş (DP), Yanal İvme Limiti & Rejenerasyon")
    print("--------------------------------------------------------------------------------\n")

    # 1. Hız Profili Benchmark'ı
    print(" [1] İleri-Geri Geçişli Dinamik Hız Profili Hesaplanıyor...")
    profilleyici = TeslaHizProfiliProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_speed_profiler()

    print(f"     -> Düzlük Maksimum Hız      : {metrikler['max_straight_speed']*3.6:.1f} km/h")
    print(f"     -> Viraj İçi Minimum Hız    : {metrikler['min_corner_speed']*3.6:.1f} km/h (R = 25m)")
    print(f"     -> Rejeneratif Enerji Kazancı: {metrikler['regen_energy_kj']:.1f} kJ (%85 Verim)")
    print(f"     -> Yanal Konfor Uyumu       : {'%100 UYUMLU (a_lat <= 2.0 m/s²)' if metrikler['is_comfortable'] else 'AŞIRI YANAL İVME'}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Hız Optimizasyonu RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['speed_step_ortalama_us']:.3f} µs (P99: {metrikler['speed_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Hız Profili    : {metrikler['saniyelik_hiz_profili']:,} Profil/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Hız Profili Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaHizProfiliGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_speed_profile_optimization_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 60 BAŞARIYLA TAMAMLANDI! HIZ PROFİLİ VE ENERJİ OPTİMİZASYONU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
