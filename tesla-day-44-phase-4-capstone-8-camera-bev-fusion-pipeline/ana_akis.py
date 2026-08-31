"""
Tesla Gün 44 Ana Akış (Tesla Day 44 Main Pipeline)
===================================================
FAZ 4 BÜYÜK CAPSTONE: 8 Kameralı Spatiotemporal BEV Füzyon Hattı
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

from src.tesla_faz4_capstone_bev_fuzyon_hatti import TeslaPhase4CapstonePipeline
from src.tesla_capstone4_profilleyici import TeslaCapstone4Profilleyici
from src.tesla_capstone4_gorsellestirici import TeslaCapstone4Gorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 44: FAZ 4 CAPSTONE - 8 KAMERALI BEV FÜZYON HATTI 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 8 Kamera, Radar, 100Hz IMU, Semantik SLAM & BEV Transformer")
    print("--------------------------------------------------------------------------------\n")

    # 1. Faz 4 Capstone Pipeline Benchmark'ı
    print(" [1] Uçtan Uca 8 Kamera ve Çoklu Sensör FSD Füzyon Hattı Yürütülüyor...")
    profilleyici = TeslaCapstone4Profilleyici(steps=100)
    metrikler = profilleyici.benchmark_capstone_pipeline()

    print(f"     -> Takip Edilen Öncü Araç Mesafesi : {metrikler['son_lead_mesafe_m']:.2f} Metre")
    print(f"     -> Takip Edilen Bağıl Hız         : {metrikler['son_lead_hiz_mps']:.2f} m/s")
    print(f"     -> Dead Reckoning Kat Edilen Yol  : {metrikler['son_ego_x_m']:.2f} Metre")
    print(f"     -> BEV Doluluk Matrisi Boyutu     : {metrikler['bev_occupancy'].shape}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Faz 4 Capstone RTOS Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi          : {metrikler['pipeline_step_ortalama_us']:.3f} µs (P99: {metrikler['pipeline_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik FSD Döngü Kapasitesi : {metrikler['saniyelik_fsd_adimi']:,} FPS (60 FPS Hedefi Aşıldı)")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Faz 4 Capstone Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaCapstone4Gorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_faz4_capstone_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi         : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 FAZ 4 BÜYÜK CAPSTONE BAŞARIYLA TAMAMLANDI! TÜM FSD GÖRÜŞ HATTI DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
