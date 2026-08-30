"""
Day 400: Unit Tests for Grand Pre-Integration Layer
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from grand_pre_integration_motoru import (
    PhaseStatus,
    CrossPhaseOrchestrator,
    GrandPreIntegrationBenchmark
)


def test_cross_phase_orchestrator_dispatch():
    """Çapraz veri yolu olay iletiminin milisaniye-altı çalıştığını test eder."""
    orchestrator = CrossPhaseOrchestrator()
    latency = orchestrator.dispatch_cross_domain_event(source_phase=19, target_phase=20, payload={"msg": "READY"})

    assert latency < 1.0
    assert len(orchestrator.message_log) == 1
    assert orchestrator.message_log[0]["status"] == "DELIVERED_ACK"


def test_phase_status_dataclass():
    """Faz durum veri modelini test eder."""
    status = PhaseStatus(phase_id=20, title="Universal Autonomy", days_range="Gun 381 - Gun 401")
    assert status.phase_id == 20
    assert status.completeness_pct == 100.0


def test_all_20_phases_verified():
    """20 Fazın tamamının başlatıldığını test eder."""
    bench = GrandPreIntegrationBenchmark(total_phases=20)
    res = bench.kos()

    assert len(res["phases"]) == 20
    assert res["total_phases_verified"] == 20
    assert res["total_days_verified"] == 400


def test_tam_grand_pre_integration_benchmark():
    """Tam büyük ön-entegrasyon benchmarkını test eder."""
    bench = GrandPreIntegrationBenchmark(total_phases=20)
    res = bench.kos()

    assert res["overall_completeness_pct"] == 100.0
    assert res["system_coherence_pct"] == 100.0
    assert res["architectural_deadlocks"] == 0
    assert res["avg_bus_latency_ms"] < 1.0
