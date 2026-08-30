"""
Day 390: Climate Engineering & Carbon Capture Optimization with Causal AI
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 30 Günlük Endüstriyel Doğrudan Havadan Karbon Yakalama (DACCS) Simülasyonu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from carbon_capture_causal_motoru import CarbonCaptureBenchmark
from carbon_profilleyici import CarbonProfilleyici
from carbon_gorsellestirici import CarbonGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 390: NEDENSEL YAPAY ZEKA ILE ATMOSFERIK KARBON YAKALAMA (DACCS)")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = CarbonCaptureBenchmark(num_units=100)
    print("\n[1/4] 30 Gunluk Endustriyel DACCS Tesis Simulasynu Kosuluyor...")
    bench_res = bench.kos(num_days=30)

    print(f"  -> Yakalanan Toplam CO2  : {bench_res['total_co2_captured_tons']:,.2f} Ton Net CO2")
    print(f"  -> Ozgul Enerji Tuketimi : {bench_res['specific_energy_consumption_mwh_ton']:.2f} MWh/ton CO2")
    print(f"  -> Yakalama Verimi       : %{bench_res['capture_efficiency_pct']:.1f}")
    print(f"  -> Seviyelendirilmis Maliyet: ${bench_res['levelized_cost_usd_ton']:.2f} / ton")
    print(f"  -> Nedensel Verim Artisi : +%{bench_res['causal_efficiency_uplift_pct']:.1f}")

    # 2. Profilleme
    print("\n[2/4] Iklim Muhendisligi ve Nedensel Otonomi Profillemesi...")
    profilleyici = CarbonProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Karbon Yakalama Teshis Paneli Ciziliyor...")
    gorsellestirici = CarbonGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 390: ATMOSFERIK KARBON YAKALAMA OPTIMIZASYONU BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
