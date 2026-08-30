"""
Day 396: Unit Tests for Autonomous Cyber Defense & Zero-Day Vaccine Synthesis
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cyber_defense_motoru import (
    VulnerabilityPayload,
    BinaryVaccine,
    SymbolicTaintAnalyzer,
    BinaryVaccineSynthesizer,
    AutonomousImmunizationBenchmark
)


def test_symbolic_taint_analyzer_overflow():
    """Leke analizcisinin bellek taşması ve RIP kontrolünü tespit ettiğini test eder."""
    analyzer = SymbolicTaintAnalyzer()
    vuln_payload = VulnerabilityPayload("EXP_01", "ROP_CHAIN", "NGINX", payload_size_bytes=1024, vulnerability_offset=512)
    is_vuln, msg = analyzer.analyze_payload(vuln_payload)

    assert is_vuln is True
    assert "CRITICAL" in msg


def test_symbolic_taint_analyzer_safe_payload():
    """Leke analizcisinin güvenli girdileri doğru tanıdığını test eder."""
    analyzer = SymbolicTaintAnalyzer()
    safe_payload = VulnerabilityPayload("SAFE_01", "CLEAN_INPUT", "NGINX", payload_size_bytes=256, vulnerability_offset=512)
    is_vuln, msg = analyzer.analyze_payload(safe_payload)

    assert is_vuln is False
    assert "SAFE" in msg


def test_binary_vaccine_synthesizer():
    """İkili aşı sentezleyicisinin doğrulanmış mikro-yama ürettiğini test eder."""
    synthesizer = BinaryVaccineSynthesizer()
    payload = VulnerabilityPayload("EXP_02", "HEAP_SPRAY", "POSTGRES", payload_size_bytes=2048, vulnerability_offset=512)
    vaccine = synthesizer.synthesize_vaccine(payload)

    assert vaccine.is_formally_verified is True
    assert vaccine.synthesis_time_ms < 50.0
    assert vaccine.bytecode_size_bytes > 0


def test_tam_cyber_defense_benchmark():
    """Tam otonom siber savunma ve aşı sentez benchmarkını test eder."""
    bench = AutonomousImmunizationBenchmark(num_exploits=100)
    res = bench.kos()

    assert res["total_exploits_tested"] == 100
    assert res["neutralization_rate_pct"] >= 99.0
    assert res["avg_synthesis_time_ms"] < 50.0
    assert res["formally_verified_pct"] == 100.0
