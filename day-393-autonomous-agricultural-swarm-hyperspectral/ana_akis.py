"""
Day 393: Autonomous Precision Agriculture Swarm: Hyperspectral Health & Selective Harvesting
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 1000 Bitkili Akıllı Tarla Sürü Denetimi, Hiperspektral Teşhis ve Seçici Robotik Hasat.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from agricultural_swarm_motoru import AgriculturalSwarmBenchmark
from agricultural_profilleyici import AgriculturalProfilleyici
from agricultural_gorsellestirici import AgriculturalGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 393: OTONOM HASSAS TARIM SURUSU: HIPERSPEKTRAL TEHIS & SECICI HASAT")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = AgriculturalSwarmBenchmark(num_plants=1000)
    print("\n[1/4] 1000 Bitkili Tarla Hiperspektral Suru Taramasi ve Hasat Basliyor...")
    bench_res = bench.kos()

    print(f"  -> Denetlenen Bitki Sayisi: {bench_res['total_plants_inspected']}")
    print(f"  -> Tespit Edilen Hastalik  : {bench_res['diseased_plants_detected']}")
    print(f"  -> Pestisit Kimyasal Tasar.: %{bench_res['pesticide_chemical_reduction_pct']:.1f}")
    print(f"  -> Hasat Edilen Meyve      : {bench_res['ripe_fruits_harvested']}")
    print(f"  -> Meyve Zedelenme Orani   : %{bench_res['fruit_bruising_rate_pct']:.2f}")

    # 2. Profilleme
    print("\n[2/4] Hassas Tarim Otonomisi ve Suru Verimliligi Profillemesi...")
    profilleyici = AgriculturalProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Tarim Surusu Teshis Paneli Ciziliyor...")
    gorsellestirici = AgriculturalGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 393: OTONOM TARIM SURUSU VE SECICI HASAT BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
