"""
Day 395: Unit Tests for Autonomous Disaster Response & Humanitarian Fleet AI
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from disaster_response_motoru import (
    DisasterZoneNode,
    HumanitarianVehicle,
    STARTTriageClassifier,
    CBBADecentralizedDispatcher,
    DisasterResponseBenchmark
)


def test_start_triage_classification():
    """START triyaj protokolünün kazazedeleri doğru sınıflandırdığını test eder."""
    triage = STARTTriageClassifier()
    # Solunum hızı > 30 -> Kırmızı Acil
    assert triage.classify_victim(respiration_rate=36.0, pulse_present=True, can_follow_commands=True) == "RED_IMMEDIATE"
    # Nabız yok ve solunum yok -> Siyah Vefat
    assert triage.classify_victim(respiration_rate=0.0, pulse_present=False, can_follow_commands=False) == "BLACK_DECEASED"
    # Normal solunum ve komut alıyor -> Sarı Gecikmeli
    assert triage.classify_victim(respiration_rate=18.0, pulse_present=True, can_follow_commands=True) == "YELLOW_DELAYED"


def test_cbba_dispatcher_roadblock_handling():
    """CBBA dağıtıcısının kapalı yollara İHA/Hava aracı atadığını test eder."""
    dispatcher = CBBADecentralizedDispatcher()
    zones = [
        DisasterZoneNode("Z1", "BlockedZone", 10.0, 10.0, 30, 10, 10, 10, 100.0, is_road_blocked=True)
    ]
    vehicles = [
        HumanitarianVehicle("AMB", "4X4_AMBULANCE", 300.0, 60.0, can_bypass_roadblock=False),
        HumanitarianVehicle("DRONE", "DRONE_VTOL", 40.0, 120.0, can_bypass_roadblock=True)
    ]

    missions = dispatcher.plan_rescue_routes(zones, vehicles)
    assert len(missions) == 1
    assert missions[0]["vehicle_type"] == "DRONE_VTOL"


def test_humanitarian_vehicle_types():
    """İnsani yardım araç özelliklerinin geçerli olduğunu test eder."""
    v = HumanitarianVehicle("HELI", "RESCUE_HELICOPTER", 500.0, 220.0, can_bypass_roadblock=True)
    assert v.speed_kmh == 220.0
    assert v.can_bypass_roadblock is True


def test_tam_disaster_response_benchmark():
    """Tam afet müdahale ve insani yardım benchmarkını test eder."""
    bench = DisasterResponseBenchmark(num_zones=10)
    res = bench.kos()

    assert res["num_zones"] == 10
    assert res["overall_survival_rate_pct"] > 90.0
    assert res["avg_response_time_min"] < 30.0
    assert len(res["missions"]) == 10
