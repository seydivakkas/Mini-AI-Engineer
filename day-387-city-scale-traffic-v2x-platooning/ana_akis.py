"""
Day 387: City-Scale Traffic Optimization & V2X Autonomous Vehicle Platooning
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: Şehir Trafiği MFD Akış Optimizasyonu, CACC Konvoy ve Dizi Kararlılığı Koşumu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from traffic_v2x_platooning_motoru import TrafficV2XBenchmark
from traffic_profilleyici import TrafficProfilleyici
from traffic_gorsellestirici import TrafficGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 387: SEHIR OLCEGINDE TRAFIK OPTIMIZASYONU & V2X OTONOM KONVOY YONETIMI")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = TrafficV2XBenchmark(platoon_size=8)
    print("\n[1/4] 8 Araclik CACC Konvoyu ve MFD Sehir Trafik Simulasyonu Kosuluyor...")
    bench_res = bench.kos(num_steps=80)

    print(f"  -> Dizi Kararliligi (String Stability): {bench_res['string_stability_ratio']:.3f} (<= 1.0 PASS)")
    print(f"  -> String Stable Durumu               : {bench_res['is_string_stable']}")
    print(f"  -> Seyahat Suresi Iyilestirmesi       : %{bench_res['travel_time_reduction_pct']:.1f}")
    print(f"  -> Aerodinamik Enerji Tasarrufu       : %{bench_res['energy_saving_pct']:.1f}")
    print(f"  -> Kavsak Kilitlenme Orani            : %{bench_res['intersection_deadlock_rate']:.1f}")

    # 2. Profilleme
    print("\n[2/4] V2X Konvoy ve Sehir Trafigi Otonomi Seviyesi Profillemesi...")
    profilleyici = TrafficProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu V2X Trafik Teshis Paneli Ciziliyor...")
    gorsellestirici = TrafficGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 387: V2X KONVOY VE SEHIR TRAFIK OPTIMIZASYONU BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
