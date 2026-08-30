"""
Day 386: Unit Tests for Autonomous Mining & Heavy Machinery Fleet
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mining_fleet_motoru import (
    HaulTruckState,
    GPSDeniedSubterraneanSLAM,
    DustParticulateFilter,
    ArticulatedTruckKinematics,
    MiningFleetBenchmark
)


def test_subterranean_slam_uwb_correction():
    """GPS'siz ortamda LiDAR+UWB SLAM kestiriminin konum hatasını sınırladığını test eder."""
    slam = GPSDeniedSubterraneanSLAM(uwb_beacon_interval_m=50.0)
    true_p = np.array([120.0, 15.0])
    
    est_p, err_m = slam.estimate_position(true_p, distance_traveled_m=120.0)
    assert err_m < 0.25, "SLAM hatası güvenli limit dahilinde olmalıdır."
    assert est_p.shape == (2,)


def test_dust_particulate_filter_clean():
    """Toz filtresinin düşük şiddetli toz parçacıklarını elediğini test eder."""
    flt = DustParticulateFilter(intensity_threshold=15.0)
    raw_pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    densities = np.array([5.0, 25.0, 35.0])  # İlk nokta toz

    clean_pts = flt.filter_point_cloud(raw_pts, densities)
    assert len(clean_pts) == 2, "Düşük yoğunluklu toz noktası elenmelidir."


def test_articulated_truck_kinematics_limits():
    """Belden kırma kinematiğinin maksimum kırma açısını (+-40 derece) koruduğunu test eder."""
    kinematics = ArticulatedTruckKinematics()
    truck = HaulTruckState(truck_id=0, pos=np.array([0.0, 0.0]), articulation_angle_rad=0.0)

    # Sürekli sola direksiyon kır
    for _ in range(50):
        truck = kinematics.update_kinematics(truck, target_speed=5.0, steering_rate=1.0, dt=0.1)

    max_rad = np.radians(40.0)
    assert abs(truck.articulation_angle_rad) <= max_rad + 1e-4, "Belden kırma açısı sınırını aşmamalıdır."


def test_tam_mining_fleet_benchmark():
    """Tam otonom maden filosu benchmarkını test eder."""
    bench = MiningFleetBenchmark(num_trucks=6)
    res = bench.kos(num_cycles=30)

    assert res["num_trucks"] == 6
    assert res["total_ore_extracted_tons"] > 500.0
    assert res["collision_count"] == 0, "Sıfır kaza kuralı sağlanmalıdır."
    assert res["avg_slam_positioning_error_m"] < 0.15
