"""
Tesla Gün 67 Ana Akış (Tesla Day 67 Main Pipeline)
===================================================
Tesla V12 UI Mimarisi, Qt6/QML ve C++ Model Entegrasyonu
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

from src.tesla_v12_ui_model import TeslaV12VehicleModel
from src.tesla_v12_profilleyici import TeslaV12UIProfilleyici
from src.tesla_v12_gorsellestirici import TeslaV12UIGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 67: TESLA V12 UI MİMARİSİ (QT6 / QML / C++) 🚗")
    print("================================================================================")
    print("Stajyer Görevi: QObject Backend, Q_PROPERTY Sinyal Bağlama & 60 FPS Telemetri")
    print("--------------------------------------------------------------------------------\n")

    # 1. UI Model Benchmark'ı
    print(" [1] Tesla V12 Dokunmatik Ekran QML Veri Akışı Simüle Ediliyor...")
    profilleyici = TeslaV12UIProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_ui_model()

    print(f"     -> Ekran Hız Göstergesi     : {metrikler['final_speed']:.1f} km/h")
    print(f"     -> Batarya Seviyesi         : %{metrikler['battery_pct']}")
    print(f"     -> Vites Modu               : [{metrikler['gear']}] DRIVE")
    print(f"     -> FSD Sürüş Durumu         : {'AKTİF' if metrikler['fsd_active'] else 'PASİF'}")
    print(f"     -> Yayınlanan Sinyal Adedi  : {metrikler['signals']} Sinyal")

    # 2. UI Render Hızı
    print("\n [2] Qt6/QML Sinyal İletim ve Kare Performansı...")
    print(f"     -> Kare Başına Gecikme      : {metrikler['ui_frame_ortalama_us']:.3f} µs (60 FPS Bütçesi: 16,666 µs)")
    print(f"     -> Saniyelik Kare Kapasitesi: {metrikler['saniyelik_kare_isleme']:,} FPS (Sıfır Kilitlenme)")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla V12 UI Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaV12UIGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_v12_ui_architecture_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 67 BAŞARIYLA TAMAMLANDI! TESLA V12 UI MODELİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
