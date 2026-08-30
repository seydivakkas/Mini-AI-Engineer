"""
Day 380: Integrated Photonic-Silicon Heterogeneous AI Supercomputer Architecture (Phase 19 Finale)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: FAZ 19 BÜYÜK FİNALİ Heterojen Fotonik-Silikon-Kuantum AI Süper-Bilgisayar SoC Koşumu.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from photonic_silicon_supercomputer_motoru import SupercomputerBenchmark
from supercomputer_profilleyici import SupercomputerProfilleyici
from supercomputer_gorsellestirici import SupercomputerGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 380: FAZ 19 BUYUK FINALI -- ENTEGRE FOTONIK-SILIKON-KUANTUM AI SUPER-BILGISAYAR")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = SupercomputerBenchmark()
    print("\n[1/4] 100 Heterojen Cok-Basli Dikkat ve Kuantum MoE Is Yuku Kosturuluyor...")
    bench_res = bench.kos(num_runs=100)

    print(f"  -> Ortalama Cikarim Gecikmesi   : {bench_res['avg_latency_ns']:.2f} ns")
    print(f"  -> Ortalama Enerji Tuketimi     : {bench_res['avg_energy_pj']:.2f} pJ")
    print(f"  -> Heterojen Verimlilik         : {bench_res['avg_tops_per_watt']:.1f} TOPS / Watt")
    print(f"  -> Klasik GPU'ya Gore Kazanc    : {bench_res['avg_energy_gain_x']:.1f}x DAHA VERIMLI")

    # 2. Profilleme
    print("\n[2/4] Heterojen Super-Hesaplama Performans Profillemesi Yapiliyor...")
    profilleyici = SupercomputerProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu FAZ 19 BUYUK FINALI Teshis Paneli Ciziliyor...")
    gorsellestirici = SupercomputerGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Özet Çıktı
    print("\n[4/4] *** FAZ 19: CIP ES-TASARIMI, FOTONIK AI & KUANTUM HIZLANDIRICILAR %100 TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
