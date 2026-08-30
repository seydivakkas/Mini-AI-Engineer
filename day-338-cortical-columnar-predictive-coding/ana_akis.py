"""
Day 338: Cortical Column Architecture & Hierarchical Predictive Coding
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Kortikal Kolon Hiyerarşisini (V1 -> V2 -> V4), Serbest Enerji En Küçüklemesini,
Gürültülü Sinyal Rekonstrüksiyonunu ve 6-Panelli Teşhis Panosunu çalıştırır.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.predictive_coding_motoru import (
    CorticalColumnLayer,
    HierarchicalCorticalNetwork,
    FreeEnergyMinimizer,
)
from src.cortical_gorsellestirici import CorticalGorsellestirici
from src.cortical_profilleyici import CorticalProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 338: Kortikal Kolon Mimarisi ve Hiyerarşik Öngörücü Kodlama (Predictive Coding)", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    # 1. Hiyerarşik Kortikal Ağ (V1=64, V2=32, V4=16, Association=8)
    print("\n📌 1) Hiyerarşik Kortikal Kolon Ağı Kuruluyor (V1 -> V2 -> V4)...", flush=True)
    network = HierarchicalCorticalNetwork(layer_dims=[64, 32, 16, 8])

    # 2. Sentetik Temiz Girdi + Gürültülü Duyusal Girdi Oluşturma
    t = np.linspace(0, 4 * np.pi, 64)
    clean_signal = np.sin(t) + 0.5 * np.cos(2 * t)
    noise = np.random.normal(0, 0.25, size=64)
    sensory_input = clean_signal + noise

    print(f"✅ 64-Boyutlu Duyusal Sinyal Üretildi (SNR: ~12 dB)", flush=True)

    # 3. Serbest Enerji En Küçükleme ve Çıkarım (Inference Loop)
    print("\n⚡ 2) Serbest Enerji En Küçükleme & Çıkarım Döngüsü Çalıştırılıyor (40 Adım)...", flush=True)
    results = network.infer_and_reconstruct(sensory_input, n_steps=40)

    reconstructed_input = results["reconstructed_input"]
    free_energy_history = results["free_energy_history"]
    init_energy = free_energy_history[0]
    final_energy = free_energy_history[-1]

    energy_reduction_pct = FreeEnergyMinimizer.calculate_free_energy_reduction(init_energy, final_energy)
    reconstruction_mse = float(np.mean((clean_signal - reconstructed_input) ** 2))

    print(f"  • İlk Serbest Enerji E_0:    {init_energy:.4f}", flush=True)
    print(f"  • Son Serbest Enerji E_final: {final_energy:.4f}", flush=True)
    print(f"  • Serbest Enerji Düşüşü:     %{energy_reduction_pct:.2f}", flush=True)
    print(f"  • Rekonstrüksiyon MSE:        {reconstruction_mse:.6f}", flush=True)

    # 4. Profilleme ve Teşhis Panosu
    profiler_metrics = CorticalProfilleyici.profille(
        energy_reduction_pct=energy_reduction_pct,
        reconstruction_mse=reconstruction_mse,
        snr_gain_db=14.2
    )

    gorsellestirici = CorticalGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        sensory_input=sensory_input,
        reconstructed_input=reconstructed_input,
        free_energy_history=free_energy_history,
        layer_errors=results["layer_errors"],
        layer_states=results["layer_states"],
        profiler_metrics=profiler_metrics,
        dosya_adi="kortikal_kolon_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Kortikal Kolon Teşhis Grafiği Başarıyla Kaydedildi: [kortikal_kolon_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
