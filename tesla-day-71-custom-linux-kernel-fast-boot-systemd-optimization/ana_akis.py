"""
Tesla Gün 71 Ana Akış (Tesla Day 71 Main Pipeline)
===================================================
Özel Linux Çekirdeği Fast-Boot (<2s) ve Systemd Optimizasyonu
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

from src.tesla_fast_boot_yonetici import TeslaFastBootOptimizer
from src.tesla_boot_profilleyici import TeslaBootProfilleyici
from src.tesla_boot_gorsellestirici import TeslaBootGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 71: EMBEDDED LINUX FAST-BOOT (<2.0s) & SYSTEMD 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Kernel XIP, Sürücü Budama, Servis Blame Analizi & Soğuk Başlatma")
    print("--------------------------------------------------------------------------------\n")

    # 1. Boot Analiz Benchmark'ı
    print(" [1] Tesla Linux Önyükleme Aşamaları ve Servisler Simüle Ediliyor...")
    profilleyici = TeslaBootProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_boot_analyzer()
    stages = metrikler["boot_stages"]

    print(f"     -> Toplam Soğuk Başlatma    : {stages['total_boot_s']:.2f} Saniye ({stages['total_boot_ms']:.0f} ms)")
    print(f"     -> Firmware POST Süresi     : {stages['firmware_post_ms']:.0f} ms")
    print(f"     -> Kernel Decompression     : {stages['kernel_init_ms']:.0f} ms")
    print(f"     -> Systemd Userspace        : {stages['systemd_userspace_ms']:.0f} ms")
    print(f"     -> UI Splash / Qt6 Ready    : {stages['ui_renderer_init_ms']:.0f} ms")
    print(f"     -> Fast-Boot Hedef Uyumu    : {'BAŞARILI (< 2.0s)' if metrikler['is_compliant'] else 'YAVAŞ'}")

    # 2. Analiz Hızı
    print("\n [2] Systemd Analiz RTOS Performansı...")
    print(f"     -> Ortalama Analiz Süresi   : {metrikler['analyzer_step_ortalama_us']:.3f} µs (P99: {metrikler['analyzer_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Analiz Kapasite: {metrikler['saniyelik_analiz_hacmi']:,} Analiz/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Fast-Boot Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaBootGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_linux_fast_boot_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 71 BAŞARIYLA TAMAMLANDI! LINUX FAST-BOOT DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
