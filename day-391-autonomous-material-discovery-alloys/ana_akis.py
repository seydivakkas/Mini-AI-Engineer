"""
Day 391: Autonomous Materials Discovery: High-Entropy Alloys & Superconductor Screening
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 1000+ Aday Alaşım Kompozisyonu Yüksek Hacimli Otonom Taraması.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from material_discovery_motoru import MaterialDiscoveryBenchmark
from material_profilleyici import MaterialProfilleyici
from material_gorsellestirici import MaterialGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 391: OTONOM MALZEME KESFI: HEA ALASIMLARI & SUPERILETKEN TARAMASI")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = MaterialDiscoveryBenchmark(num_candidates=1000)
    print("\n[1/4] 1000 Aday Kompozisyon CGCNN ve Termodinamik Taramaya Aliniyor...")
    bench_res = bench.kos()

    print(f"  -> Taranan Aday Sayisi    : {bench_res['total_candidates_screened']}")
    print(f"  -> Kararli HEA Alasimlari : {bench_res['stable_hea_alloys_found']}")
    print(f"  -> Kati Cozelti Verimi    : %{bench_res['hea_solid_solution_yield_pct']:.1f}")
    print(f"  -> Yuksek-Tc Superiletken : {bench_res['high_tc_candidates_count']} Aday (Tc > 77 K)")
    print(f"  -> Maksimum Tahmini Tc    : {bench_res['max_predicted_tc_kelvin']:.1f} K")

    # 2. Profilleme
    print("\n[2/4] Malzeme Bilimi Otonomi ve Faz Kararlilik Profillemesi...")
    profilleyici = MaterialProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Malzeme Teshis Paneli Ciziliyor...")
    gorsellestirici = MaterialGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 391: OTONOM MALZEME KESFI BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
