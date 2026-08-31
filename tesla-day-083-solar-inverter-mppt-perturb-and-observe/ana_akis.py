"""
Tesla Gün 83 Ana Akış (Tesla Day 83 Main Pipeline)
===================================================
Güneş Enerjisi ve Solar Inverter MPPT (Perturb & Observe) Kontrolü
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

from src.tesla_solar_mppt_kontrolcu import TeslaSolarMPPTController
from src.tesla_mppt_profilleyici import TeslaSolarMPPTProfilleyici
from src.tesla_mppt_gorsellestirici import TeslaSolarMPPTGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 83: SOLAR INVERTER VE MPPT KONTROLÜ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Fotovoltaik P-V Eğrisi, Perturb & Observe, %99+ Güneş Verimi")
    print("--------------------------------------------------------------------------------\n")

    # 1. MPPT Benchmark'ı
    print(" [1] 60 İterasyonluk Maksimum Güç Noktası Takip (MPPT) Simülasyonu Başlatılıyor...")
    profilleyici = TeslaSolarMPPTProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_mppt()

    print(f"     -> Teorik Maksimum Güç     : {metrikler['optimal_p']:.2f} W")
    print(f"     -> Takip Edilen Güç        : {metrikler['tracked_p']:.2f} W")
    print(f"     -> MPPT Takip Verimliliği  : %{metrikler['efficiency']:.2f}")
    print(f"     -> MPPT Kilitlenme Durumu  : {'%100 KİLİTLENDİ (Maksimum Güneş Hasadı)' if metrikler['locked'] else 'ARANIYOR'}")

    # 2. MPPT Çözüm Hızı
    print("\n [2] MPPT P&O Kontrol Algoritması RTOS Performansı...")
    print(f"     -> Ortalama Adım Süresi    : {metrikler['step_ortalama_us']:.3f} µs (P99: {metrikler['step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik MPPT Frekansı : {metrikler['saniyelik_mppt_frekansi']:,} Hz")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Solar MPPT Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaSolarMPPTGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_solar_mppt_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 83 BAŞARIYLA TAMAMLANDI! SOLAR MPPT KONTROLÜ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
