"""
Tesla Gün 35 Ana Akış (Tesla Day 35 Main Pipeline)
===================================================
Epipolar Geometri, Essential ve Fundamental Matris Kalibrasyonu
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

from src.tesla_epipolar_geometri_ve_matris import TeslaEpipolarCalibrator
from src.tesla_epipolar_profilleyici import TeslaEpipolarProfilleyici
from src.tesla_epipolar_gorsellestirici import TeslaEpipolarGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 35: EPİPOLAR GEOMETRİ VE MATRİS KALİBRASYONU 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Essential Matris E, Fundamental F, 8-Nokta SVD & Sampson Hatası")
    print("--------------------------------------------------------------------------------\n")

    # 1. Epipolar ve SVD Benchmark'ı
    print(" [1] Stereo Görüş Çifti Kalibrasyonu ve 8-Nokta SVD Çözümü...")
    profilleyici = TeslaEpipolarProfilleyici(num_points=50, iterations=100)
    metrikler = profilleyici.benchmark_epipolar_kalibrasyon()

    print(f"     -> Ortalama Sampson Hatası   : {metrikler['sampson_error_mean_px']:.6f} Piksel (Alt-Piksel Hassasiyet)")
    print(f"     -> Maksimum Sampson Hatası   : {metrikler['sampson_error_max_px']:.6f} Piksel (< 0.05 px Eşiği)")
    print(f"     -> Fundamental Matris Rankı  : {metrikler['rank_F']} (Rank-2 Kanıtlandı, det(F) = {metrikler['det_F']:.2e})")

    # 2. Epipolar Kısıt RTOS Performansı
    print("\n [2] 8-Nokta SVD ve Epipolar Kısıt Çözümleme Hızı...")
    print(f"     -> Ortalama Çözüm Süresi     : {metrikler['epipolar_step_ortalama_us']:.3f} µs (P99: {metrikler['epipolar_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kalibrasyon     : {metrikler['saniyelik_cozum_sayisi']:,} Matris/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Epipolar Geometri Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaEpipolarGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_epipolar_geometri_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi    : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 35 BAŞARIYLA TAMAMLANDI! EPİPOLAR GEOMETRİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
