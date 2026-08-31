"""
Tesla Gün 63 Ana Akış (Tesla Day 63 Main Pipeline)
===================================================
ISO 26262 ASIL-D Fonksiyonel Güvenlik ve Fail-Operational
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

from src.tesla_asil_d_guvenlik_kalkani import TeslaASILDSafetyGuard
from src.tesla_asil_d_profilleyici import TeslaASILDProfilleyici
from src.tesla_asil_d_gorsellestirici import TeslaASILDGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 63: ISO 26262 ASIL-D VE FAIL-OPERATIONAL 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Çift Kanal Doğrulama, Debounce Filtresi & MRM Güvenli Durma")
    print("--------------------------------------------------------------------------------\n")

    # 1. ASIL-D Benchmark'ı
    print(" [1] Çift Kanallı Güvenlik Döngüsü Simüle Ediliyor...")
    profilleyici = TeslaASILDProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_safety_guard()

    print(f"     -> Güvenlik Durumu          : {metrikler['state']}")
    print(f"     -> Tork Kanal Uyuşmazlığı   : {metrikler['torque_diff']:.2f} Nm (Limit: <= 0.50 Nm)")
    print(f"     -> Hız Kanal Uyuşmazlığı    : {metrikler['speed_diff']:.2f} m/s (Limit: <= 0.40 m/s)")
    print(f"     -> Donanım Güvenlik Onayı   : {'ONAYLANDI (ASIL-D SAĞLAM)' if metrikler['is_safe'] else 'ARIZA AKTİF'}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] ASIL-D Güvenlik Kalkanı RTOS Performansı...")
    print(f"     -> Ortalama Döngü Süresi    : {metrikler['safety_step_ortalama_us']:.3f} µs (P99: {metrikler['safety_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Güvenlik Hacmi : {metrikler['saniyelik_guvenlik_dongusu']:,} Kontrol/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD ASIL-D Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaASILDGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_iso_26262_asil_d_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 63 BAŞARIYLA TAMAMLANDI! ISO 26262 ASIL-D KALKANI DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
