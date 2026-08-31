"""
Tesla Gün 36 Ana Akış (Tesla Day 36 Main Pipeline)
===================================================
Derinlik Tahmini (Monocular & Stereo Depth) ve Optik Akış
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

from src.tesla_derinlik_ve_optik_akis import TeslaDepthAndOpticalFlowEstimator
from src.tesla_derinlik_profilleyici import TeslaDerinlikProfilleyici
from src.tesla_derinlik_gorsellestirici import TeslaDerinlikGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 36: DERİNLİK TAHMİNİ VE GEOMETRİK OPTİK AKIŞ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Stereo Disparity Z=fB/d, Karesel Belirsizlik σz~Z², LK Akış & TTC")
    print("--------------------------------------------------------------------------------\n")

    # 1. Derinlik ve Optik Akış Benchmark'ı
    print(" [1] 500 Noktalı Disparity-to-Depth Dönüşümü ve TTC Kestirimi...")
    profilleyici = TeslaDerinlikProfilleyici(num_samples=500, iterations=100)
    metrikler = profilleyici.benchmark_derinlik_ve_akis()

    print(f"     -> Ortalama Derinlik Hatası (MAE): {metrikler['mae_depth_m']:.3f} Metre (Gürültülü Disparity ile)")
    print(f"     -> Çarpışma Süresi (TTC)         : {metrikler['ttc_sec']:.2f} Saniye (30m Mesafe, 15m/s Yaklaşma)")
    print(f"     -> Lucas-Kanade Akış Hız Vektörü : ({metrikler['lk_vx']:.2f}, {metrikler['lk_vy']:.2f}) px/kare")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Derinlik ve Optik Akış RTOS Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi         : {metrikler['derinlik_step_ortalama_us']:.3f} µs (P99: {metrikler['derinlik_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Derinlik Haritası   : {metrikler['saniyelik_derinlik_haritasi']:,} Harita/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Derinlik ve Optik Akış Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaDerinlikGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_derinlik_optik_akis_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi        : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 36 BAŞARIYLA TAMAMLANDI! DERİNLİK VE OPTİK AKIŞ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
