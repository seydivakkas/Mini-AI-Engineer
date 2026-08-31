"""
Tesla Gün 29 Ana Akış (Tesla Day 29 Main Pipeline)
===================================================
Uzay Vektör Darbe Genişlik Modülasyonu (SVPWM) ve İnvertör Sürücüleri
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

from src.tesla_svpwm_modulatoru import TeslaSVPWMModulator
from src.tesla_svpwm_profilleyici import TeslaSVPWMProfilleyici
from src.tesla_svpwm_gorsellestirici import TeslaSVPWMGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 29: UZAY VEKTÖR PWM (SVPWM) SÜRÜCÜSÜ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 6 Sektör Tespiti, 7-Segment Simetrik PWM, +%15.47 DC Bara Kazancı")
    print("--------------------------------------------------------------------------------\n")

    # 1. 1 Tam Elektriksel Periyot Boyunca SVPWM Simülasyonu
    print(" [1] 400V DC Bara Gerilimi Altında 360 Derecelik SVPWM Sektör Çözümü...")
    profilleyici = TeslaSVPWMProfilleyici(sim_noktasi=360)
    metrikler = profilleyici.benchmark_svpwm()

    print(f"     -> Klasik SPWM Maks Gerilim : {metrikler['v_spwm_max']:.1f} V (Vdc / 2)")
    print(f"     -> Tesla SVPWM Maks Gerilim : {metrikler['v_svpwm_max']:.1f} V (Vdc / √3)")
    print(f"     -> DC Bara Verim Kazancı    : +%{metrikler['dc_gain_pct']:.2f} Daha Yüksek Çıkış Voltajı!")

    # 2. 10 kHz RTOS Modülasyon Performansı
    print("\n [2] 10 kHz (100 µs Periyotlu) SVPWM Hesaplama Gecikmesi...")
    print(f"     -> Ortalama SVPWM Adım Süresi : {metrikler['svpwm_step_ortalama_us']:.3f} µs (P99: {metrikler['svpwm_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Modülasyon Hacmi : {metrikler['saniyelik_svpwm_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla SVPWM ve İnvertör Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaSVPWMGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_svpwm_inverter_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 29 BAŞARIYLA TAMAMLANDI! UZAY VEKTÖR PWM (SVPWM) DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
