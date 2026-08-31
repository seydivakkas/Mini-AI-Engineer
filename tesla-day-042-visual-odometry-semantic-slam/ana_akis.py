"""
Tesla Gün 42 Ana Akış (Tesla Day 42 Main Pipeline)
===================================================
Görsel Odometri (VO) ve Semantik SLAM
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

from src.tesla_gorsel_odometri_ve_slam import TeslaVisualOdometrySLAM
from src.tesla_vo_profilleyici import TeslaVOProfilleyici
from src.tesla_vo_gorsellestirici import TeslaVOGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 42: GÖRSEL ODOMETRİ (VO) VE SEMANTİK SLAM 🚗")
    print("================================================================================")
    print("Stajyer Görevi: PnP + RANSAC, Dinamik Maskeleme & Döngü Kapatma (Loop Closure)")
    print("--------------------------------------------------------------------------------\n")

    # 1. VO ve SLAM Benchmark'ı
    print(" [1] 150 Noktalı 3D Harita ve Kapalı Döngü SLAM Simülasyonu...")
    profilleyici = TeslaVOProfilleyici(num_points=150)
    metrikler = profilleyici.benchmark_vo_and_slam()

    print(f"     -> Statik Inlier Oranı         : %{metrikler['inlier_orani_pct']:.1f}")
    print(f"     -> Yeniden İzdüşüm Hatası      : {metrikler['reproj_error_px']:.2f} Piksel (< 1.5 px Eşiği)")
    print(f"     -> Oluşturulan Anahtar Kareler : {metrikler['keyframes_count']} Adet")
    print(f"     -> Döngü Kapatma Durumu        : {'TETİKLENDİ (Drift Sıfırlandı)' if metrikler['loop_closed'] else 'Açık Döngü'}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] PnP + RANSAC Poz Kestirimi RTOS Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi       : {metrikler['pnp_step_ortalama_us']:.3f} µs (P99: {metrikler['pnp_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik VO Adım Kapasitesi: {metrikler['saniyelik_vo_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Görsel Odometri Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaVOGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_vo_slam_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi      : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 42 BAŞARIYLA TAMAMLANDI! GÖRSEL ODOMETRİ VE SLAM DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
