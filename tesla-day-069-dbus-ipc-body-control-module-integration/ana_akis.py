"""
Tesla Gün 69 Ana Akış (Tesla Day 69 Main Pipeline)
===================================================
D-Bus ve IPC ile Araç Gövde Kontrolcüleri (BCM) Haberleşmesi
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

from src.tesla_dbus_bcm_yonetici import TeslaDBusBodyController, LightMode
from src.tesla_dbus_profilleyici import TeslaDBusProfilleyici
from src.tesla_dbus_gorsellestirici import TeslaDBusGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 69: D-BUS SYSTEM BUS & BCM ENTEGRASYONU 🚗")
    print("================================================================================")
    print("Stajyer Görevi: com.tesla.BodyController, Asenkron IPC & RPC Metod Çağrıları")
    print("--------------------------------------------------------------------------------\n")

    # 1. D-Bus Benchmark'ı
    print(" [1] Tesla D-Bus IPC Servisi ve BCM Çağrıları Simüle Ediliyor...")
    profilleyici = TeslaDBusProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_dbus_service()

    print(f"     -> İşlenen RPC Çağrısı     : {metrikler['processed']} Çağrı/Batch")
    print(f"     -> Yayınlanan D-Bus Sinyali : {metrikler['total_signals']:,} Sinyal (Door, Light, Window)")
    print(f"     -> Far Durumu              : {metrikler['lights']}")
    print(f"     -> Şarj Portu Durumu       : {'AÇIK' if metrikler['charge_port'] else 'KAPALI'}")

    # 2. IPC Hızı
    print("\n [2] D-Bus IPC RTOS Performansı...")
    print(f"     -> Metod Başına Gecikme    : {metrikler['dbus_call_ortalama_us']:.3f} µs")
    print(f"     -> Saniyelik RPC Kapasitesi: {metrikler['saniyelik_rpc_kapasitesi']:,} Çağrı/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla D-Bus IPC Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaDBusGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_dbus_ipc_bcm_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 69 BAŞARIYLA TAMAMLANDI! D-BUS BCM SERVİSİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
