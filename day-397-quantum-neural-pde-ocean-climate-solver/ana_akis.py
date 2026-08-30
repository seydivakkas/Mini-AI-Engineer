"""
Day 397: Quantum-Assisted Neural PDE Ocean-Climate Solver
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 100 Yıllık Küresel Okyanus Termohalin Dolaşımı ve AMOC Çözümü.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from ocean_climate_pde_motoru import QuantumAcceleratedClimateBenchmark
from ocean_climate_profilleyici import OceanClimateProfilleyici
from ocean_climate_gorsellestirici import OceanClimateGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 397: NORAL PDE COZUCULERLE KUANTUM DESTEKLI KURESEL OKYANUS-IKLIM SIMULASYONU")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = QuantumAcceleratedClimateBenchmark(simulation_years=100)
    print("\n[1/4] 100 Yillik Okyanus Termohalin ve AMOC Dinamikleri FNO ile Cozuluyor...")
    bench_res = bench.kos()

    print(f"  -> Cozulen Simulasyon Yili : {bench_res['simulation_years']} Yil")
    print(f"  -> Hesaplama Hizlanmasi    : {bench_res['speedup_vs_fortran']:.0f}x KAT HIZLI")
    print(f"  -> Baslangic AMOC Akisi    : {bench_res['baseline_amoc_sv']:.1f} Sv")
    print(f"  -> 2050 AMOC Akisi         : {bench_res['final_amoc_sv']:.2f} Sv")
    print(f"  -> AMOC Zayiflama Orani    : -%{bench_res['amoc_weakening_pct']:.1f}")
    print(f"  -> Enerji Korunum Hatasi   : %{bench_res['avg_energy_conservation_error_pct']:.4f}")

    # 2. Profilleme
    print("\n[2/4] Gezegensel Iklim ve Fiziksel Korunum Profillemesi...")
    profilleyici = OceanClimateProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Okyanus-Iklim Teshis Paneli Ciziliyor...")
    gorsellestirici = OceanClimateGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 397: KUANTUM NORAL PDE IKLIM COZUCU BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
