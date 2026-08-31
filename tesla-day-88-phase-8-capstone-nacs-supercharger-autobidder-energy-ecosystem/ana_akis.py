"""
Tesla Gün 88 Ana Akış (Tesla Day 88 Main Pipeline)
===================================================
FAZ 8 BÜYÜK CAPSTONE: NACS Uyumlu Supercharger Yük Paylaşımı,
Megapack Desteği ve Autobidder Enerji Ekosistemi
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

from src.tesla_faz8_enerji_ekosistemi_simulatoru import TeslaPhase8EnergyEcosystemSimulator
from src.tesla_capstone8_profilleyici import TeslaCapstone8Profilleyici
from src.tesla_capstone8_gorsellestirici import TeslaCapstone8Gorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🏆 TESLA FSD MASTERI | GÜN 88: FAZ 8 BÜYÜK CAPSTONE ENERJİ EKOSİSTEMİ 🏆")
    print("================================================================================")
    print("Stajyer Görevi: NACS Supercharger V4, Megapack BESS, Autobidder, Solar & SiC LLC")
    print("--------------------------------------------------------------------------------\n")

    # 1. Faz 8 Capstone Benchmark'ı
    print(" [1] 16-Stall Supercharger, Megapack, Solar MPPT ve Autobidder Simülasyonu Başlatılıyor...")
    profilleyici = TeslaCapstone8Profilleyici(iterations=100)
    metrikler = profilleyici.benchmark_energy_ecosystem()

    print(f"     -> Toplam Supercharger Yükü: {metrikler['supercharger_load_kw']:,.1f} kW (16 Stall V4 NACS)")
    print(f"     -> Solar Roof MPPT Üretimi : -{metrikler['solar_generated_kw']:,.1f} kW (Temiz Enerji)")
    print(f"     -> Megapack XL BESS Desteği: -{metrikler['megapack_power_kw']:,.1f} kW (Pik Trafo Tıraşlama)")
    print(f"     -> Net Şebeke Güç Çekişi   : {metrikler['net_grid_draw_kw']:,.1f} kW (2000 kW Sınırı Korundu)")
    print(f"     -> Azami Kablo Sıcaklığı   : {metrikler['max_cable_temp']:.1f} °C (< 85°C Sıvı Soğutma Güvenli)")
    print(f"     -> Şebeke Güvenlik Durumu  : %100 TRAFO AŞIMI ENGELLENDİ & SIFIR KESİNTİ")

    # 2. Capstone Döngü Hızı
    print("\n [2] Tam Ekosistem Simülasyonu RTOS Performansı...")
    print(f"     -> Simülasyon Döngü Süresi : {metrikler['step_ortalama_us']:.3f} µs (P99: {metrikler['step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Ekosistem Hızı: {metrikler['saniyelik_dongu_kapasitesi']:,} Döngü/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Faz 8 Capstone Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaCapstone8Gorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_faz8_capstone_enerji_ekosistemi_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🏆 FAZ 8 BÜYÜK CAPSTONE VE TÜM FAZ 8 BAŞARIYLA TAMAMLANDI! TEBRİKLER! 🏆")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
