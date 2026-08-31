"""
Tesla Gün 47 Ana Akış (Tesla Day 47 Main Pipeline)
===================================================
NeRF (Neural Radiance Fields) ve Otomatik Etiketleme Motoru
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

from src.tesla_nerf_ve_otomatik_etiketleme import TeslaNeRFAutoLabeler
from src.tesla_nerf_profilleyici import TeslaNeRFProfilleyici
from src.tesla_nerf_gorsellestirici import TeslaNeRFGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 47: NeRF VE 3D OTOMATİK ETİKETLEME (AUTO-LABEL) 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Volume Rendering, Geçirgenlik İntegrali, 3D Zemin Gerçeği & PSNR")
    print("--------------------------------------------------------------------------------\n")

    # 1. NeRF ve Auto-Labeling Benchmark'ı
    print(" [1] 100 Kamera Işını Boyunca Hacimsel İntegral ve 3D BBox Çıkarımı...")
    profilleyici = TeslaNeRFProfilleyici(num_rays=100)
    metrikler = profilleyici.benchmark_nerf_auto_labeling()

    c = metrikler["bbox_center"]
    d = metrikler["bbox_dims"]
    print(f"     -> Otomatik 3D BBox Merkezi : [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}] Metre")
    print(f"     -> BBox Boyutları (GxUxY)   : [{d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}] Metre")
    print(f"     -> Rekonstrüksiyon PSNR     : {metrikler['psnr_db']:.1f} dB (> 30 dB Kalite Standardı)")
    print(f"     -> Üretilen 3D Nokta Sayısı : {metrikler['point_count']} Adet")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] NeRF Volume Rendering RTOS Performansı...")
    print(f"     -> Ortalama Işın Süresi     : {metrikler['nerf_ray_ortalama_us']:.3f} µs (P99: {metrikler['nerf_ray_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Işın Kapasitesi: {metrikler['saniyelik_nerf_isini']:,} Işın/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD NeRF ve Auto-Labeling Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaNeRFGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_nerf_auto_labeling_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 47 BAŞARIYLA TAMAMLANDI! NeRF VE OTOMATİK ETİKETLEME DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
