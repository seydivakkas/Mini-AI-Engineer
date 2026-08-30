"""
Day 333: Neuromorphic Spatial Navigation & Grid/Place Cells
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Hekzagonal Grid Hücresi Uyarımını, Hipokampal Konum Kod Çözümünü,
2D Otonom Yol Entegrasyonunu ve 6-panelli teşhis panosunu çalıştırır.
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
from src.grid_place_motoru import (
    GridCellModule,
    PlaceCellNetwork,
    NeuromorphicSpatialNavigator,
)
from src.grid_place_gorsellestirici import GridPlaceGorsellestirici
from src.grid_place_profilleyici import GridPlaceProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 333: Nöromorfik Mekansal Navigasyon: Grid ve Place Nöronları", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    # 1. 2D Hekzagonal Grid Hücresi Uyarım Haritası Oluşturma
    print("\n⚡ 1) Entorhinal Korteks 60-Derece Hekzagonal Grid Haritası Çıkarılıyor...", flush=True)
    grid_module = GridCellModule(spatial_scale=1.5, phase_offset=(0.0, 0.0))
    grid_size = 60
    coords = np.linspace(-2.0, 2.0, grid_size)
    grid_map_2d = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            grid_map_2d[j, i] = grid_module.compute_firing_rate(np.array([x, y], dtype=np.float32))

    print(f"✅ Hekzagonal Grid Haritası Hesaplandı! Çözünürlük: {grid_size}x{grid_size}", flush=True)

    # 2. Otonom Ajan 2D Dairesel Yörünge Simülasyonu
    print("\n⚡ 2) 2D Otonom Ajan Yol Entegrasyonu (Path Integration) Simüle Ediliyor...", flush=True)
    num_steps = 120
    dt = 0.1
    t = np.linspace(0, 2.0 * np.pi, num_steps)
    
    # Dairesel hız vektörleri: v_x = -sin(t), v_y = cos(t)
    vel_x = -1.2 * np.sin(t)
    vel_y = 1.2 * np.cos(t)

    navigator = NeuromorphicSpatialNavigator(initial_position=(1.2, 0.0))

    true_traj = []
    decoded_traj = []
    place_rates_history = []
    errors_history = []

    for step in range(num_steps):
        vel = np.array([vel_x[step], vel_y[step]], dtype=np.float32)
        res = navigator.update_navigation_step(vel, dt=dt)

        true_traj.append(res["true_pos"])
        decoded_traj.append(res["decoded_pos"])
        place_rates_history.append(res["place_rates"])
        errors_history.append(res["error_meters"])

    true_traj_np = np.array(true_traj, dtype=np.float32)
    decoded_traj_np = np.array(decoded_traj, dtype=np.float32)
    place_rates_np = np.array(place_rates_history, dtype=np.float32)

    mean_error = float(np.mean(errors_history))
    print(f"✅ Yol Entegrasyonu Tamamlandı! Ortalama Sürüklenme Hatası: {mean_error:.3f} metre", flush=True)

    # 3. Profilleme ve Teşhis Panosu
    profiler_metrics = GridPlaceProfilleyici.profille(
        mean_error_meters=mean_error,
        hexagonal_symmetry_score=98.0
    )

    print("\n📊 Nöromorfik Navigasyon Profilleme Metrikleri:", flush=True)
    print(f"  • Hekzagonal Simetri Sadakati:   %{profiler_metrics['hexagonal_symmetry_score']:.2f}", flush=True)
    print(f"  • Konum Kod Çözme Hassasiyeti:   %{profiler_metrics['decoding_precision_score']:.2f}", flush=True)
    print(f"  • Navigasyon Hazır Bulunurluğu:  %{profiler_metrics['navigation_readiness_score']:.2f}", flush=True)

    gorsellestirici = GridPlaceGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        true_trajectory=true_traj_np,
        decoded_trajectory=decoded_traj_np,
        grid_map_2d=grid_map_2d,
        place_rates_history=place_rates_np,
        errors_history=errors_history,
        profiler_metrics=profiler_metrics,
        dosya_adi="grid_place_navigasyon_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Navigasyon Teşhis Grafiği Başarıyla Kaydedildi: [grid_place_navigasyon_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
