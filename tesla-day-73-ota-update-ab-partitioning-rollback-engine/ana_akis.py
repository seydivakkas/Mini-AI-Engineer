"""
Tesla Gün 73 Ana Akış (Tesla Day 73 Main Pipeline)
===================================================
OTA Güncelleme Mimarisi: A/B Bölümlendirme ve Geri Alma (Rollback)
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

from src.tesla_ota_ab_slot_yonetici import OTABootSlotManager
from src.tesla_ota_profilleyici import TeslaOTAProfilleyici
from src.tesla_ota_gorsellestirici import TeslaOTAGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 73: OTA GÜNCELLEME VE A/B ROLLBACK MİMARİSİ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Çift Slot A/B, 3-Fault Boot Sayacı & Otomatik Güvenli Rollback")
    print("--------------------------------------------------------------------------------\n")

    # 1. OTA Benchmark'ı
    print(" [1] Bozuk OTA Güncellemesi ve Otomatik Rollback Senaryosu Simüle Ediliyor...")
    profilleyici = TeslaOTAProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_ota_rollback()

    print(f"     -> Kurtarılan Aktif Slot   : Slot {metrikler['final_slot']}")
    print(f"     -> Çalışan Güvenli Sürüm   : v{metrikler['final_version']}")
    print(f"     -> Rollback Başarı Durumu  : {'%100 BAŞARILI (SIFIR BRICK RİSKİ)' if metrikler['rollback_success'] else 'HATA'}")

    # 2. Durum Makinesi Hızı
    print("\n [2] A/B Durum Makinesi RTOS Performansı...")
    print(f"     -> Ortalama Geçiş Süresi   : {metrikler['rollback_step_ortalama_us']:.3f} µs (P99: {metrikler['rollback_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Geçiş Hacmi   : {metrikler['saniyelik_gecis_hacmi']:,} Geçiş/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla OTA ve A/B Bölümlendirme Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaOTAGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_ota_ab_partitioning_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 73 BAŞARIYLA TAMAMLANDI! OTA A/B ROLLBACK DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
