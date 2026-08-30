"""
Day 399: Unit Tests for Universal Polymath Scientific Researcher & Patent Drafter
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from polymath_patent_motoru import (
    ScientificHypothesis,
    PatentClaim,
    EpistemicHypothesisGenerator,
    USPTOClaimDrafter,
    UniversalPolymathBenchmark
)


def test_epistemic_hypothesis_generator():
    """Hipotez motorunun yüksek yenilik skorlu disiplinlerarası keşifler ürettiğini test eder."""
    generator = EpistemicHypothesisGenerator()
    hyps = generator.generate_breakthrough_hypotheses(count=10)

    assert len(hyps) == 10
    for h in hyps:
        assert len(h.domains) == 2
        assert h.novelty_score > 0.90
        assert h.physical_plausibility > 0.85


def test_uspto_claim_drafter_structure():
    """Patent istem taslaklayıcısının 1 bağımsız ve 9 bağımlı istem ürettiğini test eder."""
    drafter = USPTOClaimDrafter()
    hyp = ScientificHypothesis("TEST_01", "Quantum Bio Device", ["QUANTUM", "BIO"], 0.95, 0.94, r"\Psi", True)
    claims = drafter.draft_patent_claims(hyp)

    assert len(claims) == 10
    assert claims[0].claim_num == 1
    assert claims[0].is_independent is True
    assert claims[1].is_independent is False
    assert claims[0].prior_art_overlap_pct < 5.0


def test_scientific_hypothesis_dataclass():
    """Bilimsel hipotez veri modelini test eder."""
    h = ScientificHypothesis("H_01", "Hypo Title", ["PHYSICS"], 0.97, 0.96, r"E=mc^2", True)
    assert h.hypothesis_id == "H_01"
    assert h.is_validated_in_silico is True


def test_tam_polymath_benchmark():
    """Tam polimat bilimsel araştırmacı benchmarkını test eder."""
    bench = UniversalPolymathBenchmark(num_hypotheses=20)
    res = bench.kos()

    assert res["num_hypotheses"] == 20
    assert res["avg_novelty_pct"] > 90.0
    assert res["in_silico_validated_pct"] == 100.0
    assert res["drafted_claims_count"] == 10
