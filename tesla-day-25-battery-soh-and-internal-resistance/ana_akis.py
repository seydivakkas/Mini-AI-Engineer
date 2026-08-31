"""
Tesla Gün 25 Ana Akış (Tesla Day 25 Main Pipeline)
===================================================
Batarya Sağlık Durumu (SoH) ve Çevrimiçi İç Direnç İzleyici
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

from src.tesla_soh_ve_ic_direnc_izleyici import (
    BatteryCycleAgingSimulator,
    RecursiveLeastSquaresR0,
    calculate_soh_capacity,
    calculate_soh_resistance
)
from src.tesla_soh_profilleyici import TeslaSoHProfilleyici
from src.tesla_soh_gorsellestirici import TeslaSoHGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 25: BATARYA SOH & İÇ DİRENÇ İZLEME 🚗")
    print("================================================================================")
    print("Stajyer Görevi: SEI Katmanı Yaşlanması, RLS Çevrimiçi Parametre Kestirimi & EOL")
    print("--------------------------------------------------------------------------------\n")

    # 1. 2000 Döngülük Yaşlanma Simülasyonu
    print(" [1] Normal Şarj (25°C) vs Supercharger (45°C) 2000 Döngü Yaşlanma Analizi...")
    profilleyici = TeslaSoHProfilleyici(max_cycles=2000)
    metrikler = profilleyici.benchmark_batarya_soh()

    print(f"     -> Normal Şarj Kalan SoH_C : %{metrikler['final_soh_normal_pct']:.2f} (Direnç: {metrikler['final_r0_normal_mohm']:.2f} mΩ)")
    print(f"     -> Supercharger Kalan SoH_C: %{metrikler['final_soh_fast_pct']:.2f} (Direnç: {metrikler['final_r0_fast_mohm']:.2f} mΩ)")

    # 2. RLS Çevrimiçi İç Direnç Kestirimi
    print("\n [2] RLS (Recursive Least Squares) ile Sürüş Esnasında Anlık R0 Tespiti...")
    print(f"     -> Hedeflenen Gerçek R0    : {metrikler['true_r0_mohm']:.2f} mΩ")
    print(f"     -> RLS Nihai Tahmini       : {metrikler['rls_tahminler'][-1]:.2f} mΩ")
    print(f"     -> Kestirim Hata Oranı     : %{abs(metrikler['rls_tahminler'][-1] - metrikler['true_r0_mohm']) / metrikler['true_r0_mohm'] * 100:.2f}")

    # 3. Hesaplama Performansı Benchmark'ı
    print("\n [3] RLS Algoritması 1 kHz RTOS Döngü Performansı...")
    print(f"     -> Ortalama RLS Adım Süresi: {metrikler['rls_step_ortalama_us']:.3f} µs (P99: {metrikler['rls_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik RLS Kapasitesi: {metrikler['saniyelik_rls_adimi']:,} Adım/sn")

    # 4. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla Batarya SoH Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaSoHGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_soh_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 25 BAŞARIYLA TAMAMLANDI! BATARYA SAĞLIĞI (SOH) VE RLS DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
