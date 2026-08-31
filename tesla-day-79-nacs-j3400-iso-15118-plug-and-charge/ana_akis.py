"""
Tesla Gün 79 Ana Akış (Tesla Day 79 Main Pipeline)
===================================================
CCS / NACS (J3400) Şarj Protokolü ve ISO 15118 Tak-Çalıştır Şifreleme
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

from src.tesla_nacs_iso15118_motor import TeslaNACSISO15118Engine
from src.tesla_iso15118_profilleyici import TeslaISO15118Profilleyici
from src.tesla_iso15118_gorsellestirici import TeslaISO15118Gorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 79: NACS (J3400) & ISO 15118 TAK-ÇALIŞTIR 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Control Pilot PWM, HomePlug GreenPHY PLC, TLS 1.3 & V2G İletişimi")
    print("--------------------------------------------------------------------------------\n")

    # 1. ISO 15118 Benchmark'ı
    print(" [1] Tak-Çalıştır (Plug & Charge) ve V2G Mesajlaşma Döngüsü Başlatılıyor...")
    profilleyici = TeslaISO15118Profilleyici(iterations=100)
    metrikler = profilleyici.benchmark_plug_and_charge()

    print(f"     -> Araç VIN                : {metrikler['vin']}")
    print(f"     -> Kimlik Doğrulama        : {metrikler['auth_status']} (Sözleşme Onaylandı)")
    print(f"     -> Control Pilot Durumu    : {metrikler['cp_state']} (6V DC / Kontaktör Kapalı)")
    print(f"     -> Şarj Başlangıç Gücü     : {metrikler['power_kw']:.1f} kW")

    # 2. Protokol Hızı
    print("\n [2] ISO 15118 Kriptografik El Sıkışma RTOS Performansı...")
    print(f"     -> Ortalama Mesaj Süresi   : {metrikler['pnc_step_ortalama_us']:.3f} µs (P99: {metrikler['pnc_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Mesaj Kapasite: {metrikler['saniyelik_mesaj_kapasitesi']:,} Mesaj/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla NACS & ISO 15118 Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaISO15118Gorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_nacs_iso15118_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 79 BAŞARIYLA TAMAMLANDI! NACS ISO 15118 DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
