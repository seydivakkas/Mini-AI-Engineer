"""
Day 389: Global Supply Chain Disruption Forecaster & Dynamic Rerouting
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 90 Günlük Küresel Tedarik Zinciri Kriz Simülasyonu, ST-GNN ve Rota Yenileme.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from supply_chain_motoru import SupplyChainBenchmark
from supply_chain_profilleyici import SupplyChainProfilleyici
from supply_chain_gorsellestirici import SupplyChainGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 389: KURESEL TEDARIK ZINCIRI KRIZ TAHMINI & DINAMIK ROTA YENILEME")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = SupplyChainBenchmark()
    print("\n[1/4] 90 Gunluk Kuresel Kriz (Suveys Blokaji) Senaryosu Simule Ediliyor...")
    bench_res = bench.kos(num_days=90)

    print(f"  -> Yonetilen Kriz        : {bench_res['chokepoint_crisis_handled']}")
    print(f"  -> Nominal Transit Suresi: {bench_res['nominal_transit_days']} Gun")
    print(f"  -> Yeni Rota Suresi      : {bench_res['rerouted_transit_days']} Gun")
    print(f"  -> Stoksuz Kalmama Orani : %{bench_res['stockout_prevented_pct']:.1f}")
    print(f"  -> Tedarik Zinciri Direnci: %{bench_res['supply_chain_resilience_score']:.1f}")

    # 2. Profilleme
    print("\n[2/4] Tedarik Zinciri Otonomi ve Dayaniklilik Profillemesi...")
    profilleyici = SupplyChainProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Tedarik Zinciri Teshis Paneli Ciziliyor...")
    gorsellestirici = SupplyChainGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 389: KURESEL TEDARIK ZINCIRI VE ROTA YENILEME BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
