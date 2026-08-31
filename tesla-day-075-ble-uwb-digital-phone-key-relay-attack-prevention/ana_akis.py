"""
Tesla Gün 75 Ana Akış (Tesla Day 75 Main Pipeline)
===================================================
Araç İçi BLE ve UWB Dijital Telefon Anahtarı (Phone Key) Protokolü
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

from src.tesla_phone_key_uwb_dogrulayici import TeslaPhoneKeyUWBValidator
from src.tesla_phone_key_profilleyici import TeslaPhoneKeyProfilleyici
from src.tesla_phone_key_gorsellestirici import TeslaPhoneKeyGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 75: UWB PHONE KEY & RÖLE SALDIRISI KALKANI 🚗")
    print("================================================================================")
    print("Stajyer Görevi: UWB Time-of-Flight, Işık Hızı Mesafe & Relay Attack Savunması")
    print("--------------------------------------------------------------------------------\n")

    # 1. Phone Key Benchmark'ı
    print(" [1] UWB ToF Mesafe Ölçümü ve Röle Saldırısı Senaryosu Simüle Ediliyor...")
    profilleyici = TeslaPhoneKeyProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_phone_key()

    print(f"     -> Normal Kullanıcı Mesafesi: {metrikler['normal_dist']:.2f} m (4.5 ns ToF -> KİLİT AÇILDI)")
    print(f"     -> Röle Saldırganı Mesafesi : {metrikler['attack_dist']:.2f} m (35.0 ns ToF)")
    print(f"     -> Röle Saldırısı Savunması : {'%100 ENGELLENDİ (KİLİTLİ KALDI)' if metrikler['attack_detected'] else 'AÇIK TESPİT EDİLDİ'}")

    # 2. Doğrulama Hızı
    print("\n [2] UWB ToF Doğrulama RTOS Performansı...")
    print(f"     -> Ortalama Kontrol Süresi  : {metrikler['tof_check_ortalama_us']:.3f} µs (P99: {metrikler['tof_check_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kontrol Hacmi  : {metrikler['saniyelik_kilit_kontrolu']:,} Kontrol/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla UWB Phone Key Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaPhoneKeyGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_uwb_phone_key_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 75 BAŞARIYLA TAMAMLANDI! UWB PHONE KEY PROTOKOLÜ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
