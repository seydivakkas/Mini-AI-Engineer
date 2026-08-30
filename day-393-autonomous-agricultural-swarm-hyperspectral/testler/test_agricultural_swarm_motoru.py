"""
Day 393: Unit Tests for Autonomous Precision Agriculture Swarm
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agricultural_swarm_motoru import (
    PlantCanopyNode,
    HyperspectralSensorModel,
    SwarmVoronoiCoveragePlanner,
    RoboticSelectiveHarvester,
    AgriculturalSwarmBenchmark
)


def test_hyperspectral_indices_computation():
    """Hiperspektral NDVI ve PRI hesaplamasını test eder."""
    sensor = HyperspectralSensorModel()
    ndvi_healthy, pri_healthy = sensor.compute_spectral_indices(r_670=0.06, r_800=0.60, r_531=0.15, r_570=0.12)
    ndvi_sick, pri_sick = sensor.compute_spectral_indices(r_670=0.25, r_800=0.35, r_531=0.10, r_570=0.18)

    assert ndvi_healthy > 0.70
    assert ndvi_sick < 0.30
    assert sensor.diagnose_disease(ndvi_sick, pri_sick) is True


def test_swarm_voronoi_sector_partitioning():
    """Sürü planlayıcının tarlayı 4 eşit sektöre böldüğünü test eder."""
    planner = SwarmVoronoiCoveragePlanner(num_drones=4)
    sectors = planner.assign_field_sectors(field_width_m=400.0, field_length_m=400.0)

    assert len(sectors) == 4
    assert sectors[0] == (0.0, 100.0, 0.0, 400.0)
    assert sectors[3] == (300.0, 400.0, 0.0, 400.0)


def test_robotic_selective_harvester_gentle_grip():
    """Robotik tutucunun sadece olgun meyveyi zedelemeden kopardığını test eder."""
    harvester = RoboticSelectiveHarvester(max_grip_force_n=4.5)

    # Ham meyve (hasat edilmemeli)
    unripe_success, _, _ = harvester.harvest_fruit(ripeness_score=0.45)
    assert unripe_success is False

    # Olgun meyve
    ripe_success, force, bruised = harvester.harvest_fruit(ripeness_score=0.90)
    assert ripe_success is True
    assert force <= 4.5
    assert bruised is False


def test_tam_agricultural_swarm_benchmark():
    """Tam hassas tarım sürüsü benchmarkını test eder."""
    bench = AgriculturalSwarmBenchmark(num_plants=500)
    res = bench.kos()

    assert res["total_plants_inspected"] == 500
    assert res["pesticide_chemical_reduction_pct"] > 75.0
    assert res["ripe_fruits_harvested"] > 0
    assert res["fruit_bruising_rate_pct"] < 2.0
