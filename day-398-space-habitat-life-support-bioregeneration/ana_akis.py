"""
Day 398: Autonomous Deep-Space Habitat Life Support & Bioregeneration ECLSS AI
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 365 Günlük Mars Habitatı Kapalı Döngü Yaşam Destek ve Biyo-Rejenerasyon Simülasyonu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from space_life_support_motoru import SpaceHabitatBenchmark
from space_life_profilleyici import SpaceLifeProfilleyici
from space_life_gorsellestirici import SpaceLifeGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 398: UZAY ISTASYONU OTONOM YASAM DESTEK VE BIYO-REJENERASYON SISTEMI (ECLSS)")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = SpaceHabitatBenchmark(mission_days=365, crew_count=4)
    print("\n[1/4] 365 Gunluk Mars Yasam Destek ve Mikroalg Fotobiyoreaktoru Simule Ediliyor...")
    bench_res = bench.kos()

    print(f"  -> Gorev Suresi           : {bench_res['mission_days']} Gun (4 Astronot)")
    print(f"  -> Kapali Dongu Kapanmasi : %{bench_res['closure_loop_pct']:.1f}")
    print(f"  -> Ortalama PO2 Basinci   : {bench_res['avg_po2_kpa']:.2f} kPa")
    print(f"  -> Ortalama PCO2 Basinci  : {bench_res['avg_pco2_kpa']:.3f} kPa")
    print(f"  -> Kalan Su Rezervi       : {bench_res['final_water_liters']:.1f} L")
    print(f"  -> Hasat Edilen Biyokutle : {bench_res['final_algae_biomass_kg']:.1f} kg")

    # 2. Profilleme
    print("\n[2/4] Derin Uzay Biyo-Rejenerasyon ve ECLSS Guvenlik Profillemesi...")
    profilleyici = SpaceLifeProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Uzay Yasam Destek Teshis Paneli Ciziliyor...")
    gorsellestirici = SpaceLifeGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 398: UZAY YASAM DESTEK VE BIYO-REJENERASYON BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
