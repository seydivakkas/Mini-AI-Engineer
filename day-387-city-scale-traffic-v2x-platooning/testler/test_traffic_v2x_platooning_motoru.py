"""
Day 387: Unit Tests for City-Scale Traffic Optimization & V2X Autonomous Vehicle Platooning
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from traffic_v2x_platooning_motoru import (
    VehiclePlatoonMember,
    CACCPlatoonController,
    MacroscopicTrafficModel,
    IntersectionV2XCoordinator,
    TrafficV2XBenchmark
)


def test_cacc_acceleration_following():
    """CACC kontrolcüsünün önündeki aracı takip ederken ivme ürettiğini test eder."""
    cacc = CACCPlatoonController(time_gap_s=0.5, standstill_dist_m=3.5)
    lead = VehiclePlatoonMember(vehicle_id=0, platoon_id=1, pos_m=30.0, speed_m_s=20.0, accel_m_s2=0.0, is_leader=True)
    front = VehiclePlatoonMember(vehicle_id=1, platoon_id=1, pos_m=20.0, speed_m_s=20.0, accel_m_s2=-1.0)
    current = VehiclePlatoonMember(vehicle_id=2, platoon_id=1, pos_m=5.0, speed_m_s=20.0, accel_m_s2=0.0)

    acc = cacc.compute_acceleration(current, lead, front)
    assert isinstance(acc, float)
    assert -5.0 <= acc <= 3.0


def test_string_stability_attenuation():
    """CACC kontrolcüsünün dizi kararlılığını (String Stability) sağladığını test eder."""
    bench = TrafficV2XBenchmark(platoon_size=6)
    res = bench.kos(num_steps=50)

    assert res["string_stability_ratio"] <= 1.0, "Dizi kararlılığı kuralı (||H(jw)||_inf <= 1.0) sağlanmalıdır."
    assert res["is_string_stable"] is True


def test_macroscopic_traffic_mfd_flow():
    """Makroskopik MFD trafik modelinin akım ve hız eğrisini doğru hesapladığını test eder."""
    mfd = MacroscopicTrafficModel(v_free_kmh=60.0, rho_jam_veh_km=120.0)
    flow_low, speed_low = mfd.compute_flow(20.0)
    flow_high, speed_high = mfd.compute_flow(100.0)

    assert flow_low > 0.0
    assert speed_low > speed_high, "Yoğunluk arttıkça hız düşmelidir."


def test_tam_traffic_v2x_benchmark():
    """Tam şehir ölçeği V2X trafik ve konvoy benchmarkını test eder."""
    bench = TrafficV2XBenchmark(platoon_size=8)
    res = bench.kos(num_steps=60)

    assert res["platoon_size"] == 8
    assert res["energy_saving_pct"] > 10.0, "Konvoy aerodinamik enerji tasarrufu sağlamalıdır."
    assert res["intersection_deadlock_rate"] == 0.0
    assert res["travel_time_reduction_pct"] > 20.0
