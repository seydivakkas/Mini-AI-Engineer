"""
Tesla Gün 53 Ana Akış (Tesla Day 53 Main Pipeline)
===================================================
Filo Gölge Modu (Shadow Mode), A/B Testleri ve Veri Motoru
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

from src.tesla_golge_modu_ve_veri_motoru import TeslaShadowModeDataEngine
from src.tesla_golge_profilleyici import TeslaGolgeProfilleyici
from src.tesla_golge_gorsellestirici import TeslaGolgeGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 53: GÖLGE MODU (SHADOW MODE) VE VERİ MOTORU 🚗")
    print("================================================================================")
    print("Stajyer Görevi: İnsan-Model Uyuşmazlık Tetikleyicisi, Uç Klip Paketleme & A/B Z-Testi")
    print("--------------------------------------------------------------------------------\n")

    # 1. Shadow Mode Benchmark'ı
    print(" [1] İnsan-Model Karar Farkları ve Uç Klip Tetikleyicisi Çözümleniyor...")
    profilleyici = TeslaGolgeProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_shadow_mode()

    print(f"     -> Tetikleme Durumu         : {'TETİKLENDİ (UYUŞMAZLIK)' if metrikler['is_triggered'] else 'UYUMLU'}")
    print(f"     -> Direksiyon Farkı         : {metrikler['steer_diff']:.1f}° (Eşik: 5.0°)")
    print(f"     -> İvme / Fren Farkı        : {metrikler['accel_diff']:.1f} m/s² (Eşik: 1.5 m/s²)")
    print(f"     -> Tetikleme Nedenleri      : {metrikler['trigger_reasons']}")

    ab = metrikler["ab_test"]
    print(f"\n     -> A/B Testi Sonucu (MPI)   : Model A: {ab['mpi_model_a']:.1f} Mil vs Model B: {ab['mpi_model_b']:.1f} Mil")
    print(f"     -> Performans İyileşmesi    : %{ab['improvement_pct']:.1f} (Z: {ab['z_score']:.2f}, p = {ab['p_value']:.5f})")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Shadow Mode RTOS Denetim Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['shadow_step_ortalama_us']:.3f} µs (P99: {metrikler['shadow_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Denetim Hacmi  : {metrikler['saniyelik_denetim_adimi']:,} Denetim/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Shadow Mode Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaGolgeGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_shadow_mode_data_engine_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 53 BAŞARIYLA TAMAMLANDI! GÖLGE MODU VE VERİ MOTORU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
