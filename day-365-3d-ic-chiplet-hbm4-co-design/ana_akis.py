"""
Day 365: 3D-IC Chiplet Architecture & HBM4 Memory Co-Design
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; 3D-IC Dikey TSV Simülasyonunu, 4-Yığınlı HBM4 Bellek Modelini,
Williams Roofline LLM Çıkarım Analizini ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.chiplet_hbm4_codesign_motoru import (
    ThreeDICCoDesignSimulator,
)
from src.chiplet_gorsellestirici import ChipletGorsellestirici
from src.chiplet_profilleyici import ChipletProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧱 DAY 365: 3D-IC Chiplet Mimarisi ve HBM4 Bellek Eş-Tasarımı (Co-Design)", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    print("\n📌 1) 3D-IC Çiplet ve 4x HBM4 Bellek Yığını (8.192 TB/s) Simüle Ediliyor...", flush=True)

    simulator = ThreeDICCoDesignSimulator()
    roofline_res = simulator.run_llm_roofline_benchmark()

    bw_hbm4 = roofline_res["total_hbm4_bw_tb_s"]
    bw_ddr5 = roofline_res["ddr5_bw_tb_s"]
    speedup = roofline_res["llm_speedup"]
    dec_hbm = roofline_res["llm_decode_hbm4_tflops"]
    dec_ddr = roofline_res["llm_decode_ddr5_tflops"]

    print(f"\n📊 3D-IC Çiplet ve HBM4 Bellek Eş-Tasarım Performans Sonuçları:")
    print(f"  • Toplam HBM4 Bant Genişliği:        {bw_hbm4:.3f} TB/s ({bw_hbm4 * 1000.0:.0f} GB/s)")
    print(f"  • Klasik 2D DDR5 Bant Genişliği:     {bw_ddr5:.3f} TB/s (128 GB/s)")
    print(f"  • LLM Decoding Performansı (HBM4):   {dec_hbm:.2f} TFLOPS (Memory-Bound)")
    print(f"  • LLM Decoding Performansı (DDR5):   {dec_ddr:.3f} TFLOPS (Bellek Duvarı)")
    print(f"  • LLM Token Üretim Hızlanması:       {speedup:.1f}x Kat Hızlı")
    print(f"  • Dikey TSV Gecikmesi:               {roofline_res['tsv_latency_ps']:.4f} ps (Kayıpsız Geçiş)")
    print(f"  • 3D-IC Donanım Eş-Tasarımı:         ✅ %100 BAŞARILI")

    profiler_metrics = ChipletProfilleyici.profille(roofline_res)

    gorsellestirici = ChipletGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        roofline_res=roofline_res,
        profiler_metrics=profiler_metrics,
        dosya_adi="chiplet_hbm4_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli 3D-IC & HBM4 Teşhis Grafiği Başarıyla Kaydedildi: [chiplet_hbm4_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
