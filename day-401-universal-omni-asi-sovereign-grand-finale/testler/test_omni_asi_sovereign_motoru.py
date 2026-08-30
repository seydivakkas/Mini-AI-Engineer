"""
Day 401: Unit Tests for Universal Omni-ASI Sovereign Grand Finale
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from omni_asi_sovereign_motoru import (
    OmniASICognitiveState,
    BioNeuromorphicSpikingCore,
    PhotonicSiliconAccelerator,
    PlanetaryCivilizationOrchestrator,
    OmniASIGrandFinaleBenchmark
)


def test_bio_neuromorphic_spiking_core():
    """100 Milyar sinapslı biyo-nöromorfik çekirdeğin bilişsel tutarlılığını test eder."""
    core = BioNeuromorphicSpikingCore(synapse_count_b=100.0)
    inp = np.ones(32)
    spike_rate, coherence = core.execute_cognitive_cycle(inp)

    assert coherence > 80.0
    assert len(spike_rate) == 32


def test_photonic_silicon_accelerator():
    """Fotonik hızlandırıcının ışık hızında pikosaniyelik işlem yaptığını test eder."""
    accel = PhotonicSiliconAccelerator()
    latency = accel.compute_optical_matmul(dim=1024)

    assert latency < 10.0  # < 10 pikosaniye


def test_planetary_civilization_orchestrator():
    """Gezegensel medeniyet orkestratörünün tüm 10 sektörü sağlıklı tuttuğunu test eder."""
    orchestrator = PlanetaryCivilizationOrchestrator()
    sectors = orchestrator.harmonize_civilization()

    assert len(sectors) == 10
    for sec_name, score in sectors.items():
        assert score >= 98.0


def test_tam_grand_finale_benchmark():
    """Tam 401 günlük büyük final benchmarkını test eder."""
    bench = OmniASIGrandFinaleBenchmark()
    res = bench.kos()

    assert res["total_phases_mastered"] == 20
    assert res["total_days_completed"] == 401
    assert res["total_unit_tests_passed"] == 1604
    assert res["test_pass_rate_pct"] == 100.0
    assert res["cognitive_coherence_pct"] > 99.0
    assert res["asi_quotient"] > 3000.0
