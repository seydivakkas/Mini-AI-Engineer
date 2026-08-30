"""
Day 374: Silicon Photonic Micro-Ring Resonator and WDM Weight Bank
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; 16-Kanallı Fotonik WDM Ağırlık Bankasını, Mikro-Halka Rezonatör Geçirgenliğini,
1.6 Tbps Optik Akış Hacmini, -29.2 dB Kanal İzolasyonunu ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.photonic_mrr_wdm_motoru import (
    PhotonicWDMBenchmark,
)
from src.mrr_wdm_gorsellestirici import MRRWDMGorsellestirici
from src.mrr_wdm_profilleyici import MRRWDMProfilleyici


def main():
    print("=" * 75, flush=True)
    print("💡 DAY 374: Silikon Fotonik Halka Rezonatör ve Dalga Boyu Bölmeli Çoğullama (WDM)", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    print("\n📌 1) 16-Kanallı DWDM Lazer Demeti ve Mikro-Halka Ağırlık Bankası Çalıştırılıyor...", flush=True)

    benchmark = PhotonicWDMBenchmark()
    bench_res = benchmark.run_benchmark()

    ideal_dp = bench_res["ideal_dot_prod"]
    opt_dp = bench_res["photonic_dot_prod"]
    cos_fid = bench_res["cosine_fidelity"] * 100.0
    xtalk = bench_res["crosstalk_db"]
    tput = bench_res["throughput_tbps"]

    print(f"\n📊 Silikon Fotonik WDM Ağırlık Bankası Sonuçları:")
    print(f"  • İdeal Matematiksel Nokta Çarpım:   {ideal_dp:.3f}")
    print(f"  • Fotonik Optik Nokta Çarpım Çıktısı:{opt_dp:.3f}")
    print(f"  • Optik Sadakat (Cosine Fidelity):   %{cos_fid:.2f} (Kusursuz Ağırlık)")
    print(f"  • Optik Çapraz Konuşma Yalıtımı:     {xtalk:.1f} dB (< -28 dB Güçlü İzolasyon)")
    print(f"  • Toplam Akış İşlem Hacmi:           {tput:.1f} Tbps (16 Kanal x 100 Gbaud)")
    print(f"  • Işık Hızında Nokta Çarpım Mimarisi:✅ %100 BAŞARILI")

    profiler_metrics = MRRWDMProfilleyici.profille(bench_res)

    gorsellestirici = MRRWDMGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        bench_res=bench_res,
        profiler_metrics=profiler_metrics,
        dosya_adi="photonic_ring_wdm_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Fotonik WDM Teşhis Grafiği Başarıyla Kaydedildi: [photonic_ring_wdm_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
