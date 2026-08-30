"""
Day 389: Unit Tests for Global Supply Chain Disruption Forecaster & Dynamic Rerouting
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from supply_chain_motoru import (
    SupplyChainNode,
    LogisticsEdge,
    STGNNSupplyChainForecaster,
    DynamicReroutingEngine,
    SupplyChainBenchmark
)


def test_stgnn_disruption_prediction_cascade():
    """ST-GNN modelinin şok düğümünden komşu rotalara kriz yayılımını modellediğini test eder."""
    forecaster = STGNNSupplyChainForecaster()
    nodes = [
        SupplyChainNode("SUEZ", "Suez", "PORT", 0, 0, 1000, (30.0, 32.0)),
        SupplyChainNode("ROTTERDAM", "Rotterdam", "PORT", 5000, 1000, 10000, (51.0, 4.0))
    ]
    edges = [LogisticsEdge("SUEZ", "ROTTERDAM", "MARITIME", 7.0, 1500.0, is_chokepoint=True)]

    risks = forecaster.predict_disruptions(nodes, edges, external_shock_node="SUEZ")
    assert risks["SUEZ"] >= 0.90
    assert risks["ROTTERDAM"] > 0.40, "Kriz komşu limana kaskad yayılmalıdır."


def test_dynamic_rerouting_engine_bypass():
    """Dinamik rota motorunun tıkalı boğazı baypas eden alternatif rota ürettiğini test eder."""
    rerouter = DynamicReroutingEngine()
    edges = [
        LogisticsEdge("SINGAPORE", "SUEZ_CANAL", "MARITIME", 11.0, 2200.0, is_chokepoint=True)
    ]
    risks = {"SUEZ_CANAL": 0.95}

    active, t_days, cost = rerouter.reoptimize_routes(edges, risks, blocked_chokepoint="SUEZ_CANAL")
    assert len(active) == 1
    assert "CAPE" in active[0].transport_mode
    assert t_days > 11.0, "Ümit Burnu rotası daha uzun olmalıdır."
    assert cost > 2200.0


def test_supply_chain_network_structure():
    """Tedarik zinciri ağ yapısının geçerli liman ve depolar içerdiğini test eder."""
    bench = SupplyChainBenchmark()
    nodes, edges = bench._build_global_network()

    assert len(nodes) >= 6
    assert len(edges) >= 5
    assert any(n.node_type == "PORT" for n in nodes)


def test_tam_supply_chain_benchmark():
    """Tam tedarik zinciri kriz ve dayanıklılık benchmarkını test eder."""
    bench = SupplyChainBenchmark()
    res = bench.kos(num_days=60)

    assert res["num_days"] == 60
    assert res["supply_chain_resilience_score"] > 90.0
    assert res["stockout_prevented_pct"] > 85.0
    assert res["delay_mitigation_pct"] > 25.0
