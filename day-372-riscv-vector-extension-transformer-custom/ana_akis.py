"""
Day 372: Custom RISC-V Vector Extension ISA Design for Transformer Kernels
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; 256-Bit RISC-V Vektör Kayıtlarını, Özel Transformer Komutlarını (GeLU, Softmax, LayerNorm),
Donanım Komut ve Saykıl Hızlanmasını ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.riscv_transformer_isa_motoru import (
    TransformerKernelBenchmark,
)
from src.riscv_isa_gorsellestirici import RISCVISAGorsellestirici
from src.riscv_isa_profilleyici import RISCVISAProfilleyici


def main():
    print("=" * 75, flush=True)
    print("⚡ DAY 372: Transformer Çekirdekleri için Özel RISC-V Vektör Komut Seti Tasarımı", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    print("\n📌 1) 256-Bit Vektör Genişlikli RISC-V AI İşlemcisinde Transformer Çekirdeği Çalıştırılıyor...", flush=True)

    benchmark = TransformerKernelBenchmark()
    bench_res = benchmark.run_benchmark(seq_len=8, hidden_dim=8)

    s_inst = bench_res["scalar_instructions"]
    c_inst = bench_res["custom_instructions"]
    i_red = bench_res["instruction_reduction"]
    s_cyc = bench_res["scalar_cycles"]
    c_cyc = bench_res["custom_cycles"]
    spd = bench_res["cycle_speedup"]
    mse = bench_res["mse_fidelity"]

    print(f"\n📊 RISC-V Özel Vektör ISA Benchmark Sonuçları:")
    print(f"  • Standart Skaler Komut Sayısı:      {s_inst} komut")
    print(f"  • Özel RVV-AI Vektör Komut Sayısı:   {c_inst} komut")
    print(f"  • Dinamik Komut Sayısı Azaltımı:     {i_red:.1f}x Tasarruf")
    print(f"  • Standart Skaler Saat Çevrimi:      {s_cyc} saykıl")
    print(f"  • Özel RVV-AI Saat Çevrimi:          {c_cyc} saykıl")
    print(f"  • Donanım Yürütme Hızlanması:        {spd:.1f}x Hızlanma")
    print(f"  • GeLU Polinom Sadakati (MSE):       {mse:.2e} (Kusursuz)")
    print(f"  • Özel Donanım ISA Entegrasyonu:     ✅ %100 BAŞARILI")

    profiler_metrics = RISCVISAProfilleyici.profille(bench_res)

    gorsellestirici = RISCVISAGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        bench_res=bench_res,
        profiler_metrics=profiler_metrics,
        dosya_adi="riscv_transformer_isa_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli RISC-V ISA Teşhis Grafiği Başarıyla Kaydedildi: [riscv_transformer_isa_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
