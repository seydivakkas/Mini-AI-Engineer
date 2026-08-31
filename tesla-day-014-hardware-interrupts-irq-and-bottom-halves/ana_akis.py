"""
Tesla Gun 14 Ana Akis (Tesla Day 14 Main Pipeline)
===================================================
Donanim Kesmeleri (IRQ), Top-Half / Bottom-Half & AEB Radar TTC
Uctan Uca Calistirma ve Teshis Paneli Uretim Scripti.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
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

from src.tesla_kesme_yoneticisi import (
    TeslaTopHalfHardIRQ,
    TeslaBottomHalfThreadedIRQ,
    TeslaKesmeFirtinasiOnleyici,
    TeslaKesmeYonetimSistemi
)
from src.tesla_irq_profilleyici import TeslaIRQProfilleyici
from src.tesla_irq_gorsellestirici import TeslaIRQGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 14: HARDWARE IRQ & BOTTOM-HALF 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: HardIRQ ACK, request_threaded_irq, AEB Radar TTC & Kesme Fırtınası Koruması")
    print("--------------------------------------------------------------------------------\n")

    # 1. Kesme Yönetim Sistemi Başlatma
    print(" [1] Tesla AEB Radar IRQ Yönetim Sistemi Başlatılıyor...")
    yonetim = TeslaKesmeYonetimSistemi()

    # 2. Senaryo 1: Güvenli Takip Mesafesi (50 m, -10 m/s -> TTC = 5.0 s)
    print("\n [2] Senaryo 1: Güvenli Takip Mesafesi Donanım Kesmesi Tetikleniyor...")
    sonuc1 = yonetim.donanim_kesmesi_olustur(mesafe_m=50.0, hiz_mps=-10.0)
    b1 = sonuc1["bottom_half_sonuc"]
    print(f"     -> Top-Half Yanıtı      : {sonuc1['top_half_sonuc']}")
    print(f"     -> Radar TTC            : {b1['ttc_sn']:.2f} sn | Acil Fren: {'AKTİF' if b1['acil_fren_tetiklendi'] else 'PASİF'} (Durum: {b1['durum']})")

    # 3. Senaryo 2: Kritik Çarpışma Tehlikesi (12 m, -25 m/s -> TTC = 0.48 s)
    print("\n [3] Senaryo 2: Ani Engel Çarpışma Senaryosu Donanım Kesmesi Tetikleniyor...")
    sonuc2 = yonetim.donanim_kesmesi_olustur(mesafe_m=12.0, hiz_mps=-25.0)
    b2 = sonuc2["bottom_half_sonuc"]
    print(f"     -> Top-Half Yanıtı      : {sonuc2['top_half_sonuc']}")
    print(f"     -> Radar TTC            : {b2['ttc_sn']:.2f} sn | Acil Fren: {'🚨 DERHAL AKTİF EDİLDİ 🚨' if b2['acil_fren_tetiklendi'] else 'PASİF'} (Durum: {b2['durum']})")

    # 4. Profilleme ve Karşılaştırma
    print("\n [4] Top-Half HardIRQ vs Monolitik IRQ Bloklama Benchmark'ı...")
    profilleyici = TeslaIRQProfilleyici(ornek_sayisi=5000)
    metrikler = profilleyici.benchmark_tophalf_vs_monolitik()

    print(f"     -> Top-Half HardIRQ Gecikmesi (Ortalama): {metrikler['tophalf_ortalama_us']:.3f} µs (P99: {metrikler['tophalf_p99_us']:.3f} µs)")
    print(f"     -> Monolitik Bloklayıcı Kesme Gecikmesi : {metrikler['monolitik_ortalama_us']:.3f} µs")
    print(f"     -> Hızlanma / Tepkisellik Kazancı       : {metrikler['hizlanma_orani']:.1f}x Daha Hızlı")
    print(f"     -> Kesme Fırtınası Engellenen İstek Oranı: %{metrikler['firtina_red_orani']:.1f}")

    # 5. Tanı Paneli Görselleştirme
    print("\n [5] 6 Panelli Tesla IRQ Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaIRQGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_irq_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 14 BAŞARIYLA TAMAMLANDI! HAFTA 2 LINUX & RTOS TAMAMLANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
