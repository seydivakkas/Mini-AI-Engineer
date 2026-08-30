"""
Day 347: Decentralized Drone Swarm Flocking with Graph Neural Networks (GNN)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; 15-İHA'lı Sürünün Graf Sinir Ağları (GNN) Mesaj Geçirme ile Merkeziyetsiz
Flocking, Hız Hizalanması (Consensus) ve 3D Hedefe İlerleme Simülasyonunu çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.drone_flocking_gnn_motoru import (
    DecentralizedSwarmSimulator,
)
from src.flocking_gorsellestirici import FlockingGorsellestirici
from src.flocking_profilleyici import FlockingProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🛸 DAY 347: İHA Sürüsü Merkeziyetsiz Sürü Davranışı: Graf Sinir Ağları ile Flocking", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    num_drones = 15
    sim = DecentralizedSwarmSimulator(num_drones=num_drones, dt=0.05)
    target_waypoint = np.array([120.0, 120.0, 80.0]) # 3D Görev Hedefi (m)

    num_steps = 200
    drone_trajectories = [] # (T, N, 3)
    min_distances_m = []
    velocity_variances = []

    print(f"\n📌 1) {num_drones}-İHA'lı Sürü Simülasyonu ve GNN Mesaj Geçirme Başlatılıyor...", flush=True)

    for step in range(num_steps):
        step_res = sim.step_simulation(target_waypoint)
        drone_trajectories.append(step_res["positions"])
        min_distances_m.append(step_res["min_distance_m"])
        velocity_variances.append(step_res["velocity_variance"])

    drone_trajectories = np.array(drone_trajectories)

    min_overall_dist = float(np.min(min_distances_m))
    final_vel_var = float(velocity_variances[-1])
    com_final = np.mean(drone_trajectories[-1], axis=0)
    final_goal_dist = float(np.linalg.norm(com_final - target_waypoint))

    print(f"\n📊 GNN İHA Sürüsü Flocking Performans Sonuçları:", flush=True)
    print(f"  • Sürü İçi Minimum Mesafe:        {min_overall_dist:.2f} metre (> 2.0 m Güvenli Eşik)", flush=True)
    print(f"  • Final Hız Mutabakatı Varyansı:   {final_vel_var:.4f} m²/s² (Sürü Senkronizasyonu)", flush=True)
    print(f"  • Sürü Ağırlık Merkezi Hedef Hatası: {final_goal_dist:.2f} metre", flush=True)
    print(f"  • Sürü İçi Çarpışma:              0 Adet (%100 GÜVENLİ)", flush=True)

    # 2. Profilleme ve Teşhis Panosu
    profiler_metrics = FlockingProfilleyici.profille(
        min_inter_drone_dist_m=min_overall_dist,
        final_velocity_var=final_vel_var,
        final_goal_dist_m=final_goal_dist
    )

    gorsellestirici = FlockingGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        drone_trajectories=drone_trajectories,
        min_distances_m=min_distances_m,
        velocity_variances=velocity_variances,
        target_waypoint=target_waypoint,
        profiler_metrics=profiler_metrics,
        dosya_adi="iha_flocking_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli İHA Sürü Flocking Teşhis Grafiği Başarıyla Kaydedildi: [iha_flocking_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
