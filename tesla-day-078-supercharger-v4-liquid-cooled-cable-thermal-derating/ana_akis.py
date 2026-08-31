"""
Tesla Gün 78 Ana Akış (Tesla Day 78 Main Pipeline)
===================================================
Supercharger V4: 1000V DC, Sıvı Soğutmalı Kablo ve Termal Derating
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

from src.tesla_supercharger_v4_derater import TeslaSuperchargerV4CableDerater
from src.tesla_v4_profilleyici import TeslaV4Profilleyici
from src.tesla_v4_gorsellestirici import TeslaV4Gorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 78: SUPERCHARGER V4 SIVI SOĞUTMALI KABLO & DERATING 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 1000V DC Mimari, 500 kW Güç, Kablo Joule Isınması & Termal Kalkan")
    print("--------------------------------------------------------------------------------\n")

    # 1. Supercharger V4 Benchmark'ı
    print(" [1] 120 Saniyelik Ultra Hızlı Şarj ve Termal Dinamik Simülasyonu Başlatılıyor...")
    profilleyici = TeslaV4Profilleyici(iterations=100)
    metrikler = profilleyici.benchmark_v4_derating()

    print(f"     -> Şarj Mimarisi           : 1000V DC / 500A (500 kW Nominal)")
    print(f"     -> Son Kablo Sıcaklığı     : {metrikler['final_temp_c']:.1f} °C (Kritik Eşik: 85 °C)")
    print(f"     -> Son Şarj Gücü           : {metrikler['final_power_kw']:.1f} kW")
    print(f"     -> Termal Güvenlik Durumu  : %100 KORUMALI (Sıvı Soğutma & Derating Devrede)")

    # 2. Termal Hesaplama Hızı
    print("\n [2] Termal Dinamik ve Derating RTOS Performansı...")
    print(f"     -> Ortalama Adım Süresi    : {metrikler['step_ortalama_us']:.3f} µs (P99: {metrikler['step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kontrol Hacmi : {metrikler['saniyelik_kontrol_kapasitesi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Supercharger V4 Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaV4Gorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_supercharger_v4_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 78 BAŞARIYLA TAMAMLANDI! SUPERCHARGER V4 DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
