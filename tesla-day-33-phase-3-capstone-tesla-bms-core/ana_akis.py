"""
Tesla Gün 33 Ana Akış (Tesla Day 33 Main Pipeline)
===================================================
FAZ 3 BÜYÜK CAPSTONE: Tam Kapsamlı Tesla BMS ve Güç Aktarma Çekirdeği
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

from src.tesla_faz3_capstone_bms_ve_cekis_cekirdegi import CapstonePowertrainCore
from src.tesla_bms_capstone_profilleyici import TeslaCapstoneProfilleyici
from src.tesla_bms_capstone_gorsellestirici import TeslaCapstoneGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🏆 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 33: FAZ 3 BÜYÜK CAPSTONE BİRLEŞTİRME 🏆")
    print("================================================================================")
    print("Stajyer Görevi: 96S ECM, EKF SoC, RLS SoH, 10kHz FOC, SVPWM, Octovalve, Rejen & HVIL")
    print("--------------------------------------------------------------------------------\n")

    # 1. 0-120 km/h İvmelenme, Otoyol Seyri, Tek Pedallı Rejen ve HVIL Kesme Testi
    print(" [1] 1500 Adımlık Uçtan Uca Gerçek Zamanlı Powertrain & BMS Simülasyonu...")
    profilleyici = TeslaCapstoneProfilleyici(sim_adimlari=1500)
    metrikler = profilleyici.benchmark_capstone_surus_dongusu()

    print(f"     -> Ulaşılan Maksimum Hız       : {metrikler['max_speed_kmh']:.1f} km/h")
    print(f"     -> Maksimum Çekiş Gücü (Launch): {metrikler['max_power_kw']:.1f} kW (Bataryadan Çekilen)")
    print(f"     -> Maksimum Rejenerasyon Gücü : {metrikler['max_regen_power_kw']:.1f} kW (Bataryaya Basılan)")
    print(f"     -> Sürüş Döngüsü Sonu EKF SoC : %{metrikler['final_soc_pct']:.2f}")

    # 2. 100 Hz RTOS Karar Döngüsü
    print("\n [2] 100 Hz / 10 kHz Çok Katmanlı Powertrain Karar Gecikmesi...")
    print(f"     -> Ortalama Döngü Süresi      : {metrikler['capstone_step_ortalama_us']:.3f} µs (P99: {metrikler['capstone_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Karar Kapasitesi : {metrikler['saniyelik_capstone_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Faz 3 Büyük Capstone Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaCapstoneGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_faz3_capstone_bms_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🏆 TEBRİKLER! FAZ 3 BÜYÜK CAPSTONE BAŞARIYLA TAMAMLANDI! (GÜN 23 - 33) 🏆")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
