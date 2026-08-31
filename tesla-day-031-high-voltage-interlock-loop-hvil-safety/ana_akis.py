"""
Tesla Gün 31 Ana Akış (Tesla Day 31 Main Pipeline)
===================================================
Yüksek Gerilim Kilidi (HVIL), İzolasyon ve Güvenlik Sistemleri
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

from src.tesla_hvil_ve_guvenlik_sistemi import (
    TeslaHVILSafetyManager,
    HighVoltageSystemState,
    ContactorState,
    HVILStatus
)
from src.tesla_hvil_profilleyici import TeslaHVILProfilleyici
from src.tesla_hvil_gorsellestirici import TeslaHVILGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 31: YÜKSEK GERİLİM GÜVENLİK SİSTEMİ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: HVIL Döngü Denetimi, Pyrofuse Patlatma & Precharge Sıralaması")
    print("--------------------------------------------------------------------------------\n")

    # 1. 400V Precharge ve HVIL Güvenlik Benchmark'ı
    print(" [1] 400V DC Precharge Sıralaması ve HVIL Arıza Simülasyonu...")
    profilleyici = TeslaHVILProfilleyici(sim_ms=500)
    metrikler = profilleyici.benchmark_hvil_guvenlik()

    print(f"     -> Precharge Tamamlanma Süresi: {metrikler['precharge_time_ms']} ms (%95 DC Link Şarjı)")
    print(f"     -> HVIL Kesilme Sonrası Durum : {metrikler['hvil_shutdown_status']} (Acil Güç Kesildi)")
    print(f"     -> Kaza Durumu Pyrofuse Tepkisi: {'BAŞARIYLA PATLATILDI' if metrikler['pyrofuse_blown'] else 'HATA'}")

    # 2. 1 kHz ASIL-D Güvenlik Döngüsü RTOS Performansı
    print("\n [2] 1 kHz ASIL-D Güvenlik Karar Döngüsü RTOS Performansı...")
    print(f"     -> Ortalama Döngü Süresi      : {metrikler['hvil_step_ortalama_us']:.3f} µs (P99: {metrikler['hvil_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Karar Kapasitesi : {metrikler['saniyelik_guvenlik_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Yüksek Gerilim Güvenlik Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaHVILGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_hvil_guvenlik_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 31 BAŞARIYLA TAMAMLANDI! HVIL VE YÜKSEK GERİLİM GÜVENLİĞİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
