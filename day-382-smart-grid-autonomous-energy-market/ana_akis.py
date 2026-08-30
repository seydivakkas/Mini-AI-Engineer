"""
Day 382: Smart Grid Autonomous Energy Balancing & Decentralized Agent Market
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: 14-Baralı Akıllı Şebeke, Çift Yönlü Piyasa Takası ve Frekans Kararlılık Simülasyonu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from smart_grid_market_motoru import SmartGridBenchmark
from grid_profilleyici import GridProfilleyici
from grid_gorsellestirici import GridGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 382: AKILLI SEBEKE OTONOM ENERJI DENGELEME & COKLU-AJAN PIYASASI")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = SmartGridBenchmark()
    print("\n[1/4] 24 Saatlik 14-Barali IEEE Test Sebekesi ve Piyasa Simule Ediliyor...")
    bench_res = bench.kos(num_hours=24)

    print(f"  -> Ortalama Frekans Sapmasi  : {bench_res['avg_frequency_deviation_hz']:.4f} Hz")
    print(f"  -> Yenilenebilir Enerji Payi : %{bench_res['avg_renewable_penetration_pct']:.1f}")
    print(f"  -> Ortalama Takas Fiyati     : {bench_res['avg_mcp_usd_mwh']:.2f} $/MWh")
    print(f"  -> Sebeke Kararlilik Indeksi : %{bench_res['grid_stability_pct']:.1f}")

    # 2. Profilleme
    print("\n[2/4] Akilli Sebeke ve Piyasa Otonomisi Profillemesi Yapiliyor...")
    profilleyici = GridProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Akilli Sebeke Teshis Paneli Ciziliyor...")
    gorsellestirici = GridGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 382: AKILLI SEBEKE VE ENERJI PIYASASI BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
