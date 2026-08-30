"""
Day 388: Unit Tests for Autonomous Legal Arbitration & Compliance Sandbox
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from legal_arbitration_motoru import (
    LegalClause,
    ArbitrationCase,
    DeonticLogicEngine,
    ConflictOfLawsHarmonizer,
    AutomatedArbitrator,
    LegalArbitrationBenchmark
)


def test_deontic_logic_consistency_check():
    """Deontik mantık motorunun normatif çelişkileri (O vs F) doğru yakaladığını test eder."""
    engine = DeonticLogicEngine()
    c1 = LegalClause("C1", "Mandatory data_sharing with partners", "OBLIGATION", "GLOBAL")
    c2 = LegalClause("C2", "Strict prohibition of data_sharing with third parties", "PROHIBITION", "EU")

    is_consistent, conflicts = engine.check_normative_consistency([c1, c2])
    assert is_consistent is False
    assert len(conflicts) > 0


def test_conflict_of_laws_harmonization():
    """Kanunlar ihtilafı motorunun çoklu yargı alanını uyumladığını test eder."""
    harmonizer = ConflictOfLawsHarmonizer()
    case = ArbitrationCase(
        case_id="C_01",
        claimant="Alpha",
        respondent="Beta",
        breach_type="DATA_PRIVACY",
        claimed_damages_eur=100000.0,
        evidence_score=0.85,
        jurisdictions_involved=["EU", "US"]
    )

    res = harmonizer.harmonize_case(case)
    assert "EU" in res["applicable_law"]
    assert res["statutory_multiplier"] >= 1.20


def test_automated_arbitrator_verdict_liability():
    """Otonom hakemin güçlü delil durumunda tazminat hükmettiğini test eder."""
    arbitrator = AutomatedArbitrator()
    case = ArbitrationCase(
        case_id="C_STRONG",
        claimant="Alpha",
        respondent="Beta",
        breach_type="SLA_LATENCY",
        claimed_damages_eur=200000.0,
        evidence_score=0.90,
        jurisdictions_involved=["US"]
    )
    clauses = [LegalClause("C1", "Uptime obligation", "OBLIGATION", "US")]

    res = arbitrator.arbitrate_case(case, clauses)
    assert res["is_liable"] is True
    assert res["final_award_eur"] > 100000.0
    assert "LIABLE" in res["verdict"]


def test_tam_legal_arbitration_benchmark():
    """Tam otonom hukuki tahkim benchmarkını test eder."""
    bench = LegalArbitrationBenchmark(num_cases=50)
    res = bench.kos()

    assert res["total_cases_processed"] == 50
    assert res["avg_arbitration_latency_ms"] < 10.0
    assert res["decision_accuracy_pct"] > 90.0
    assert res["cross_border_compliance_pass"] is True
