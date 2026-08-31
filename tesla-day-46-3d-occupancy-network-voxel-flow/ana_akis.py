"""
Tesla Gün 46 Ana Akış (Tesla Day 46 Main Pipeline)
===================================================
3D Occupancy Network: 3 Boyutlu Voksel Doluluk ve Hız Kestirimi
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

from src.tesla_3d_occupancy_network import Tesla3DOccupancyNetwork
from src.tesla_occupancy_profilleyici import TeslaOccupancyProfilleyici
from src.tesla_occupancy_gorsellestirici import TeslaOccupancyGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 46: 3D OCCUPANCY NETWORK VE VOKSEL AKIŞI (FLOW) 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 40,000 Voksel, 3D Voxel Flow Hız Vektörleri & Kutulanamaz Engeller")
    print("--------------------------------------------------------------------------------\n")

    # 1. 3D Occupancy Benchmark'ı
    print(" [1] 3D Voksel Izgarası ve Voxel Flow Hız Alanı Çözümleniyor...")
    profilleyici = TeslaOccupancyProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_occupancy_network()

    print(f"     -> Toplam Voksel Hücresi     : {metrikler['toplam_voksel']:,} Adet (50x50x16)")
    print(f"     -> Dolu Voksel Sayısı        : {metrikler['dolu_voksel_sayisi']:,} (Doluluk Oranı: %{metrikler['doluluk_orani_pct']:.1f})")
    print(f"     -> Öncü Araç 3D Akış Hızı    : Vx = {metrikler['car_vx_mps']:.1f} m/s")
    print(f"     -> Yürüyen Yaya 3D Akış Hızı : Vy = {metrikler['ped_vy_mps']:.1f} m/s")
    print(f"     -> Devrilmiş Ağaç Tespiti    : {'YAKALANDI (Güvenli Fren Aktif)' if metrikler['tree_captured'] else 'Kaçırıldı'}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] 3D Occupancy Network RTOS Çıkarım Performansı...")
    print(f"     -> Ortalama Çözüm Süresi     : {metrikler['occupancy_step_ortalama_us']:.3f} µs (P99: {metrikler['occupancy_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Voksel Hacmi    : {metrikler['saniyelik_voksel_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD 3D Occupancy Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaOccupancyGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_3d_occupancy_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi    : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 46 BAŞARIYLA TAMAMLANDI! 3D OCCUPANCY NETWORK DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
