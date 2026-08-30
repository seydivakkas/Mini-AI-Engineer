"""
Day 381: Unit Tests for Autonomous Mega-Factory Orchestrator
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mega_factory_orchestrator_motoru import (
    AMRState,
    RobotWorkcell,
    SpaceTimeReservations,
    MAPFPathPlanner,
    JobShopScheduler,
    FactoryDigitalTwin,
    MegaFactoryBenchmark
)


def test_uzay_zaman_rezervasyon_ve_cakisma_onleme():
    """Uzay-zaman rezervasyon tablosunun köşe ve kenar geçiş çakışmalarını doğru engellediğini test eder."""
    res = SpaceTimeReservations()
    res.reserve(amr_id=1, path=[(5, 5), (5, 6), (5, 7)], start_t=0)

    # Aynı zamanda aynı konum çakışması
    assert not res.is_vertex_available(5, 6, t=1)
    assert res.is_vertex_available(5, 6, t=2)  # t=2'de robot (5, 7)'de, (5, 6) boş

    # Karşılıklı kenar geçişi (edge swap) çakışması: Robot 1 (5, 5)->(5, 6) t=0->1 yapıyor.
    # Başka bir robot (5, 6)->(5, 5) yapamaz:
    assert not res.is_edge_available(u=(5, 6), v=(5, 5), t=0)
    assert res.is_edge_available(u=(5, 5), v=(5, 6), t=0)


def test_mapf_yol_planlama_ve_engeller():
    """MAPF Uzay-Zaman A* algoritmasının engelleri aşarak hedefe ulaştığını test eder."""
    obstacles = {(2, 1), (2, 2), (2, 3)}
    planner = MAPFPathPlanner(grid_w=10, grid_h=10, obstacles=obstacles)
    reservations = SpaceTimeReservations()

    start = (0, 2)
    goal = (4, 2)
    path = planner.plan_path(start, goal, reservations, start_t=0)

    assert path is not None
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal
    # Engellerin üzerinden geçmediğini doğrula
    for pos in path:
        assert pos not in obstacles


def test_dinamik_job_shop_is_atamasi():
    """Job Shop sıralayıcının doğru hücre tiplerine ve en az yüklü istasyona iş atadığını test eder."""
    workcells = [
        RobotWorkcell(cell_id=0, x=10, y=10, process_type="CNC_MILLING", remaining_ticks=5),
        RobotWorkcell(cell_id=1, x=20, y=20, process_type="CNC_MILLING", remaining_ticks=1),
        RobotWorkcell(cell_id=2, x=30, y=30, process_type="ROBOT_WELDING", remaining_ticks=0)
    ]
    scheduler = JobShopScheduler(workcells)
    pending = [{"job_id": 101, "process_type": "CNC_MILLING"}]

    assignments = scheduler.assign_jobs(pending)
    assert len(assignments) == 1
    # cell_id=1 seçilmeli çünkü remaining_ticks=1 < remaining_ticks=5
    assert assignments[0]["assigned_cell_id"] == 1


def test_tam_mega_fabrika_benchmark_ve_oee():
    """Mega-Fabrika dijital ikiz vardiya simülasyonunu ve OEE başarımını test eder."""
    bench = MegaFactoryBenchmark()
    res = bench.kos(num_ticks=30)

    assert res["ticks_simulated"] == 30
    assert res["collision_rate_pct"] <= 5.0  # Düşük/sıfır çakışma
    assert res["amr_fleet_utilization_pct"] > 0.0
    assert "oee_pct" in res
