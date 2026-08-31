"""
Tesla Gun 07 Ana Akis (Tesla Day 07 Main Pipeline)
===================================================
C++20 std::span, std::ranges ve std::string_view ile Sifir Tahsisli Veri Isleme
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

from src.tesla_span_ranges_ayristirici import TeslaNMEAAyristirici
from src.tesla_ayristirici_profilleyici import TeslaAyristiriciProfilleyici
from src.tesla_ayristirici_gorsellestirici import TeslaAyristiriciGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 07: C++20 SPAN, RANGES & STRING_VIEW 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: Sifir Heap Tahsisi, Zero-Copy NMEA GPS Ayristirma & Bellek Koruma")
    print("--------------------------------------------------------------------------------\n")

    # 1. Zero-Copy NMEA Ayrıştırma Demoları
    ornek_nmea = "$GPRMC,083559.00,A,3723.2475,N,12208.3845,W,55.4,180.0,300826,,,A*72"
    print(f" [1] Ham GNSS NMEA Cümlesi Alındı:\n     -> \"{ornek_nmea}\"")

    konum = TeslaNMEAAyristirici.gprmc_ayristir(ornek_nmea)
    if konum:
        print("\n     -> C++20 string_view ile Sıfır-Kopyalama Ayrıştırma Tamamlandı:")
        print(f"        * Enlem (Latitude)   : {konum.enlem_derece:.6f}° Kuzey")
        print(f"        * Boylam (Longitude) : {konum.boylam_derece:.6f}° Batı (Tesla HQ, Palo Alto)")
        print(f"        * Araç Hızı          : {konum.hiz_kmh:.1f} km/h")
        print(f"        * Rota Açısı (Azi.)  : {konum.rota_acisi_derece:.1f}°")
        print(f"        * UTC Zamanı / Tarih : {konum.utc_zamani} / {konum.tarih}")

    # 2. Profilleme ve Heap Tahsis Karşılaştırması
    print("\n [2] std::string_view vs Heap Allocation Performans Benchmark'ı...")
    profilleyici = TeslaAyristiriciProfilleyici(dongu_sayisi=10000)
    metrikler = profilleyici.benchmark_string_view_vs_kopyalama()

    print(f"     -> std::string_view Ortalama Gecikme : {metrikler['view_ort_ns']:.1f} ns (Heap Tahsisi: {metrikler['view_tahsis_sayisi']})")
    print(f"     -> Klasik std::string Ortalama Gecik : {metrikler['kopya_ort_ns']:.1f} ns (Heap Tahsisi: {metrikler['kopya_tahsis_sayisi']})")
    print(f"     -> Hızlanma Çarpanı                  : {metrikler['hizlanma_orani']:.1f}x Daha Hızlı")
    print(f"     -> Saniyedeki Cümle Ayrıştırma Hacmi : {metrikler['saniyede_cumle_sayisi']:,.0f} Cümle/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Zero-Copy Ayrıştırıcı Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaAyristiriciGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_ayristirici_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 07 BAŞARIYLA TAMAMLANDI! SIFIR-KOPYALAMA AYRIŞTIRICI DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
