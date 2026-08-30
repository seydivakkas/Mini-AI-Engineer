"""
Day 348: Degraded Visual Environment (DVE) Sensor Fusion (LiDAR + Radar + FLIR)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Yoğun Toz/Sis (Brownout) Koşullarında LiDAR + mmWave Radar + FLIR Termal
Sensörlerinin Adaptif Füzyonunu, 3D Engel Kestirimini ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.dve_sensor_fusion_motoru import (
    DVESensorSimulator,
    AdaptiveDVEFusionEngine,
    ObstacleGridMapper,
)
from src.dve_gorsellestirici import DVEGorsellestirici
from src.dve_profilleyici import DVEProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🚁 DAY 348: Zorlu Görüş Koşullarında (DVE) Sensör Füzyonu: LiDAR + Radar + FLIR", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    # 1. 3D Gerçek Engel Sahası (Helipad Çevresindeki Engeller)
    num_obstacles = 12
    angles = np.linspace(0, 2*np.pi, num_obstacles, endpoint=False)
    radii = np.random.uniform(12.0, 22.0, num_obstacles)
    true_obstacles = np.zeros((num_obstacles, 3))
    true_obstacles[:, 0] = radii * np.cos(angles)
    true_obstacles[:, 1] = radii * np.sin(angles)
    true_obstacles[:, 2] = np.random.uniform(1.0, 5.0, num_obstacles) # Engel yüksekliği (m)

    sim = DVESensorSimulator(true_obstacles=true_obstacles)
    fusion_engine = AdaptiveDVEFusionEngine()
    mapper = ObstacleGridMapper(safe_radius_m=10.0)

    # 2. Ağır Brownout / Toz Fırtınası (%85 Görüş Bozulması)
    degradation_gamma = 0.85
    print(f"\n📌 1) Helikopter İniş Sahasında %{degradation_gamma*100:.0f} Yoğun Toz (Brownout) Simüle Ediliyor...", flush=True)

    sensor_data = sim.sample_sensors(degradation_gamma=degradation_gamma)
    fused_pos, fused_vars = fusion_engine.fuse_measurements(sensor_data)

    # 3. Hata Analizi (RMSE Hesabı)
    lidar_meas = sensor_data["lidar_meas"]
    radar_meas = sensor_data["radar_meas"]
    flir_meas = sensor_data["flir_meas"]

    lidar_rmse = float(np.sqrt(np.mean((lidar_meas - true_obstacles) ** 2)))
    radar_rmse = float(np.sqrt(np.mean((radar_meas - true_obstacles) ** 2)))
    flir_rmse = float(np.sqrt(np.mean((flir_meas - true_obstacles) ** 2)))
    fused_rmse = float(np.sqrt(np.mean((fused_pos - true_obstacles) ** 2)))

    is_safe = mapper.evaluate_safe_landing_zone(np.array([0.0, 0.0, 0.0]), fused_pos)

    print(f"\n📊 DVE Çoklu-Sensör Füzyon Performans Sonuçları:")
    print(f"  • LiDAR Hata Seviyesi (Tozda Bozulmuş): {lidar_rmse:.3f} metre")
    print(f"  • mmWave Radar Hatası (Toz Penetrasyonu):{radar_rmse:.3f} metre")
    print(f"  • FLIR Termal Kamera Hatası:            {flir_rmse:.3f} metre")
    print(f"  • Adaptif Füzyon Hatası (Optimum CI):   {fused_rmse:.3f} metre (< 0.35 m Kriteri)")
    print(f"  • Emniyetli İniş Bölgesi Durumu:         {'✅ TEMİZ / İNİŞE UYGUN' if is_safe else '❌ ENGEL TESPİT EDİLDİ'}")

    errors_dict = {
        "lidar_rmse": lidar_rmse,
        "radar_rmse": radar_rmse,
        "flir_rmse": flir_rmse,
        "fused_rmse": fused_rmse
    }

    profiler_metrics = DVEProfilleyici.profille(
        errors_dict=errors_dict,
        safe_landing=is_safe
    )

    gorsellestirici = DVEGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        true_obstacles=true_obstacles,
        lidar_meas=lidar_meas,
        radar_meas=radar_meas,
        flir_meas=flir_meas,
        fused_pos=fused_pos,
        errors_dict=errors_dict,
        profiler_metrics=profiler_metrics,
        dosya_adi="dve_sensor_fuzyon_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli DVE Sensör Füzyon Teşhis Grafiği Başarıyla Kaydedildi: [dve_sensor_fuzyon_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
