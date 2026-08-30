"""
Day 377: Wafer-Scale Engine (WSE) 2D-Torus Network-on-Chip (NoC) & Fault Tolerance
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: Wafer-Scale 2D-Torus Kumaş Simülasyonu, Kusur Baypas ve Raporlama.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from wse_2d_torus_noc_motoru import WSEBenchmark
from wse_noc_profilleyici import WSENoCProfilleyici
from wse_noc_gorsellestirici import WSENoCGorsellestirici


def main():
    print("=" * 70)
    print(" DAY 377: WAFER-SCALE ENGINE (WSE) 2D-TORUS NoC & FAULT-TOLERANT ROUTING")
    print("=" * 70)

    # 1. Benchmark Koşumu
    bench = WSEBenchmark(width=16, height=16)
    print("\n[1/4] 16x16 Wafer-Scale 2D-Torus Kumaş Simülasyonu Koşturuluyor...")
    bench_res = bench.kos()

    h_res = bench_res["healthy"]
    f_res = bench_res["faulty"]
    print(f"  -> Kusursuz Wafer Teslimat Oranı : %{h_res['delivery_rate']:.1f} (Ort. Hop: {h_res['avg_hops']:.2f})")
    print(f"  -> %5 Kusurlu Wafer Teslimat     : %{f_res['delivery_rate']:.1f} (Ort. Hop: {f_res['avg_hops']:.2f})")
    print(f"  -> Toplam Bisection Bant Genişliği: {bench_res['bisection_bw_pbps']:.3f} PB/s")

    # 2. Profilleme
    print("\n[2/4] NoC Performans ve Kusur Toleransı Profillemesi Yapılıyor...")
    profilleyici = WSENoCProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yüksek Çözünürlüklü WSE Teşhis Paneli Çiziliyor...")
    gorsellestirici = WSENoCGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teşhis Paneli Kaydedildi: {panel_yolu}")

    # 4. Özet Çıktı
    print("\n[4/4] Wafer-Scale 2D-Torus NoC Simülasyonu Başarıyla Tamamlandı!")
    print("=" * 70)


if __name__ == "__main__":
    main()
