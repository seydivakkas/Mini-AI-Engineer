"""
Tesla Gün 43 Ana Akış (Tesla Day 43 Main Pipeline)
===================================================
Tesla Vision Park Asistanı ve Yüksek Çözünürlüklü Mesafe Kestirimi
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

from src.tesla_vision_park_asistani import TeslaVisionParkAssist
from src.tesla_park_profilleyici import TeslaParkProfilleyici
from src.tesla_park_gorsellestirici import TeslaParkGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 43: TESLA VISION YÜKSEK ÇÖZÜNÜRLÜKLÜ PARK ASİSTANI 🚗")
    print("================================================================================")
    print("Stajyer Görevi: USS'siz 3D Voxel Doluluk, 360° Işın Atma & Kör Nokta Belleği")
    print("--------------------------------------------------------------------------------\n")

    # 1. Park Asistanı Benchmark'ı
    print(" [1] 3D Voxel Izgara ve 360 Derece Işın Atma Park Simülasyonu...")
    profilleyici = TeslaParkProfilleyici(iterations=50)
    metrikler = profilleyici.benchmark_park_assist()

    print(f"     -> Tespit Edilen En Yakın Mesafe : {metrikler['min_mesafe_cm']:.1f} cm")
    print(f"     -> Park İkaz Durumu             : {metrikler['ikaz_metni']}")
    print(f"     -> Izgara Çözünürlüğü           : 5 cm Voxel Hücre Boyutu")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] 360° Mesafe Kestirimi RTOS Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi        : {metrikler['park_step_ortalama_us']:.3f} µs (P99: {metrikler['park_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Park Güncellemesi  : {metrikler['saniyelik_park_guncellemesi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Vision Park Asistanı Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaParkGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_vision_park_asistani_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi       : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 43 BAŞARIYLA TAMAMLANDI! TESLA VISION PARK ASİSTANI DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
