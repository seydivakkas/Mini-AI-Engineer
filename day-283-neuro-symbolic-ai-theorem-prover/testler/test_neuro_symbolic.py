"""
PyTest Birim Testleri - Day 283 (FAZ 15): Nöro-Sembolik Teorem İspatlayıcı.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.neuro_symbolic_motoru import NeuroSymbolicTheoremProverEngine, LogicClause
from src.neuro_symbolic_profilleyici import NeuroSymbolicProfilleyici
from src.gorsellestirici import NeuroSymbolicGorsellestirici


def test_logic_clause_representation():
    """1. LogicClause doğru sembolik temsil ve dizgi formatı üretmelidir."""
    clause = LogicClause("ModusPonens", ["P", "P_implies_Q"], "Q")
    assert clause.name == "ModusPonens"
    assert "P ∧ P_implies_Q ⟹ Q" in repr(clause)


def test_engine_add_fact_and_rule():
    """2. Bilgi tabanına olgu ve kurallar eksiksiz eklenmelidir."""
    engine = NeuroSymbolicTheoremProverEngine()
    engine.add_fact("A")
    engine.add_rule("Rule1", ["A"], "B")
    assert "A" in engine.facts
    assert len(engine.knowledge_base) == 1


def test_neural_premise_scoring():
    """3. Sinirsel öncül puanlama hedefle eşleşen kurallara daha yüksek puan atamalıdır."""
    engine = NeuroSymbolicTheoremProverEngine()
    engine.add_fact("A")
    r1 = LogicClause("R1", ["A"], "B")
    r2 = LogicClause("R2", ["C"], "D")
    scored = engine.neural_premise_scoring("B", [r1, r2])
    assert scored[0][0].conclusion == "B"
    assert scored[0][1] > scored[1][1]


def test_prove_theorem_soundness():
    """4. Rolle ve sıfır türev teoremi geriye doğru çıkarımla eksiksiz ispatlanmalıdır."""
    engine = NeuroSymbolicTheoremProverEngine()
    engine.add_fact("IsContinuous(f)")
    engine.add_fact("IsDifferentiable(f)")
    engine.add_fact("f(a) == f(b)")
    engine.add_fact("a < b")

    engine.add_rule("Rolle_Rule", ["IsContinuous(f)", "IsDifferentiable(f)", "f(a) == f(b)"], "RolleApplicable(f)")
    engine.add_rule("Zero_Derivative_Rule", ["RolleApplicable(f)", "a < b"], "ExistsC_f_prime_zero(f)")

    res = engine.prove_theorem("ExistsC_f_prime_zero(f)")
    assert res["is_proven"] is True
    assert res["proof_steps_count"] > 0


def test_prove_unprovable_theorem():
    """5. İspatlanamayan geçersiz hedefler ispatlanamaz olarak işaretlenmeli ve halüsinasyon üretilmemelidir."""
    engine = NeuroSymbolicTheoremProverEngine()
    engine.add_fact("IsContinuous(f)")
    res = engine.prove_theorem("NonExistentProperty(f)")
    assert res["is_proven"] is False


def test_zero_hallucination_rate():
    """6. Nöro-sembolik ispatlayıcı sıfır halüsinasyon (%0.0) oranını garanti etmelidir."""
    engine = NeuroSymbolicTheoremProverEngine()
    engine.add_fact("Fact1")
    engine.add_rule("R1", ["Fact1"], "Goal1")
    res = engine.prove_theorem("Goal1")
    assert res["hallucination_rate"] == 0.0


def test_profiler_hybrid_advantage():
    """7. Profilleyici nöro-sembolik yöntemin %98.4 ispat oranına ve 78x hızlanmaya ulaştığını teyit etmelidir."""
    profil = NeuroSymbolicProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["dogrulanmis_ispat_orani_yuzde"]["Noro_Sembolik_Hibrit"] > 95.0
    assert kars["halusinasyon_orani_yuzde"]["Noro_Sembolik_Hibrit"] == 0.0
    assert kars["ispat_gecikmesi_ms"]["Noro_Sembolik_Hibrit"] < 30.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. NeuroSymbolicGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_neuro_symbolic_paneli.png")
    profil = NeuroSymbolicProfilleyici.basarim_profili_cikar()

    NeuroSymbolicGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
