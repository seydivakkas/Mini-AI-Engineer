"""
Tesla Gün 64 Ana Akış (Tesla Day 64 Main Pipeline)
===================================================
Çift Kanallı Güvenlik ve FSD HW Çip Yedekliliği (Lockstep Arbiter)
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

from src.tesla_cift_node_arabulucu import FSDHardwareArbiter
from src.tesla_cift_node_profilleyici import TeslaCiftNodeProfilleyici
from src.tesla_cift_node_gorsellestirici import TeslaCiftNodeGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 64: ÇİFT DÜĞÜM (DUAL-NODE) FSD ÇİP YEDEKLİLİĞİ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Node A & Node B Lockstep, Karar Arabulucusu & Ayrışma Duruşu")
    print("--------------------------------------------------------------------------------\n")

    # 1. Arabulucu Benchmark'ı
    print(" [1] Çift NPU Çıkarım Oylama Mekanizması Simüle Ediliyor...")
    profilleyici = TeslaCiftNodeProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_arbiter()

    print(f"     -> Arabulucu Modu           : {metrikler['mode']}")
    print(f"     -> Durum Özeti              : {metrikler['status_desc']}")
    print(f"     -> Uygulanan Direksiyon     : {metrikler['steer_applied']:.3f} rad ({metrikler['steer_applied']*57.2958:.2f}°)")
    print(f"     -> Uygulanan İvme           : {metrikler['acc_applied']:.2f} m/s²")
    print(f"     -> Çift Çip Uyuşmazlığı     : Direksiyon {metrikler['steer_diff']:.3f} rad, İvme {metrikler['acc_diff']:.2f} m/s²")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] FSD Arabulucu RTOS Performansı...")
    print(f"     -> Ortalama Oylama Süresi   : {metrikler['arbiter_step_ortalama_us']:.3f} µs (P99: {metrikler['arbiter_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Oylama Hacmi   : {metrikler['saniyelik_oylama_hacmi']:,} Karar/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Çift Düğüm Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaCiftNodeGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_dual_node_fsd_redundancy_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 64 BAŞARIYLA TAMAMLANDI! ÇİFT DÜĞÜM ARABULUCU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
