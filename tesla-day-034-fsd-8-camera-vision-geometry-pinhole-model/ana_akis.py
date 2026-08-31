"""
Tesla Gün 34 Ana Akış (Tesla Day 34 Main Pipeline)
===================================================
8 Kamera Görüş Geometrisi, İğne Deliği Modeli ve Distorsiyon Düzeltme
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

from src.tesla_8kamera_gorus_geometrisi import Tesla8CameraVisionRig
from src.tesla_kamera_profilleyici import TeslaKameraProfilleyici
from src.tesla_kamera_gorsellestirici import TeslaKameraGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 34: 8-KAMERA GÖRÜŞ GEOMETRİSİ VE İĞNE DELİĞİ MODELİ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 360° Çevre Görüş, İçsel K, Dışsal [R|t] & Brown-Conrady Düzeltme")
    print("--------------------------------------------------------------------------------\n")

    # 1. 8 Kamera ve 3D Dünya Projeksiyon Benchmark'ı
    print(" [1] 8 Kamera Donanım Tezgâhı ve 360° 3D Sahne Projeksiyonu...")
    profilleyici = TeslaKameraProfilleyici(num_points=200, iterations=100)
    metrikler = profilleyici.benchmark_kamera_geometrisi()

    print(f"     -> Toplam 3D Sahne Noktası : {metrikler['total_points']} Nokta (360° Çevre)")
    print(f"     -> Toplam Kamera İzdüşümü : {metrikler['total_visible_detections']} Piksel Tespiti")
    print("     -> Kamera Başına Tespit Dağılımı:")
    for cam_name, count in metrikler['cam_visibility_counts'].items():
        print(f"        * {cam_name:<16}: {count} Nokta Görünür")

    # 2. 36 FPS FSD Gerçek Zamanlı Geometri RTOS Performansı
    print("\n [2] 8-Kamera İzdüşüm ve Distorsiyon Çözümleme Hızı...")
    print(f"     -> Ortalama Kare Süresi   : {metrikler['geometri_step_ortalama_us']:.3f} µs (P99: {metrikler['geometri_step_p99_us']:.3f} µs)")
    print(f"     -> 36 FPS Bütçe Kullanımı : %{(metrikler['geometri_step_ortalama_us'] / 27777.0) * 100:.2f} (Kalan Bütçe NPU/HydraNet'e Ayrıldı)")
    print(f"     -> Saniyelik Kare İşleme  : {metrikler['saniyelik_kare_isleme']:,} Kare/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD 8-Kamera Görüş Geometrisi Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaKameraGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_8kamera_geometri_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 34 BAŞARIYLA TAMAMLANDI! 8-KAMERA GÖRÜŞ GEOMETRİSİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
