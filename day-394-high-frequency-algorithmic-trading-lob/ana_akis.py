"""
Day 394: Microsecond Algorithmic Trading with Limit Order Book Dynamics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 10.000 Mikrosaniye Seviye-3 LOB Algoritmik Ticaret Simülasyonu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hft_lob_trading_motoru import HFTTradingBenchmark
from hft_profilleyici import HFTProfilleyici
from hft_gorsellestirici import HFTGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 394: MIKROSANIYE HFT LOB ALGORITMIK TICARET & HAWKES SURECLERI")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = HFTTradingBenchmark(num_ticks=10000)
    print("\n[1/4] 10.000 Mikrosaniye Seviye-3 LOB Ticaret Simule Ediliyor...")
    bench_res = bench.kos()

    print(f"  -> Islenen Tik Sayisi     : {bench_res['num_ticks']}")
    print(f"  -> Net PnL (USD)          : ${bench_res['final_pnl_usd']:,.2f}")
    print(f"  -> Sharpe Orani           : {bench_res['sharpe_ratio']:.2f}")
    print(f"  -> Maksimum Drawdown      : %{bench_res['max_drawdown_pct']:.2f}")
    print(f"  -> Ortalama Tik Gecikmesi : {bench_res['avg_latency_us']:.2f} us")
    print(f"  -> Hawkes Dallanma Orani  : eta={bench_res['branching_ratio_eta']:.2f}")

    # 2. Profilleme
    print("\n[2/4] Kantitatif Finans ve HFT Risk Profillemesi...")
    profilleyici = HFTProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu HFT Teshis Paneli Ciziliyor...")
    gorsellestirici = HFTGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 394: MIKROSANIYE HFT LOB TICARET BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
