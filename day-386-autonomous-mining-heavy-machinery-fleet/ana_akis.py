"""
Day 386: Autonomous Mining & Heavy Machinery Fleet in GPS-Denied Environments
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 8 Araçlık Belden Kırmalı Otonom Maden Filosu, Yeraltı SLAM ve Çizelgeleme.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mining_fleet_motoru import MiningFleetBenchmark
from mining_profilleyici import MiningProfilleyici
from mining_gorsellestirici import MiningGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 386: GPS'SIZ YERALTI OTONOM MADENCILIK & AGIR IS MAKINESI FILOSU")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = MiningFleetBenchmark(num_trucks=8)
    print("\n[1/4] 8 Belden Kirmali Kamyon ile Yeralti Maden Sevk ve SLAM Simulasyonu...")
    bench_res = bench.kos(num_cycles=50)

    print(f"  -> Toplam Tasinan Cevher : {bench_res['total_ore_extracted_tons']:.1f} Ton")
    print(f"  -> Uretim Hizi (Debi)    : {bench_res['production_rate_tons_per_hr']:.1f} Ton/Saat")
    print(f"  -> Yeralti SLAM Hatasi   : {bench_res['avg_slam_positioning_error_m']:.3f} metre")
    print(f"  -> Toz Filtreleme Verimi : %{bench_res['dust_filtering_efficiency_pct']:.1f}")
    print(f"  -> Kaza/Carpisma Sayisi  : {bench_res['collision_count']}")

    # 2. Profilleme
    print("\n[2/4] Otonom Madencilik Verim ve Guvenlik Profillemesi...")
    profilleyici = MiningProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Otonom Madencilik Teshis Paneli Ciziliyor...")
    gorsellestirici = MiningGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 386: OTONOM MADENCILIK VE AGIR IS MAKINESI FILOSU BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
