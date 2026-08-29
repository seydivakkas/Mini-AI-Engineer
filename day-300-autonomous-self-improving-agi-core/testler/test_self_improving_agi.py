"""
PyTest Birim Testleri - Day 300 (FAZ 15): Kendi Kendini Geliştiren Sürekli AGI Çekirdeği.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.self_improving_agi_motoru import (
    CognitiveArchitecture,
    RecursiveSelfModifier,
    FormalProofSandbox,
    AtomicStateHotSwapper,
)
from src.self_improving_agi_profilleyici import SelfImprovingAGIProfilleyici
from src.gorsellestirici import SelfImprovingAGIGorsellestirici


def test_cognitive_architecture_initialization():
    """1. Bilişsel mimari nesnesi temel metriklerle başlatılabilmelidir."""
    arch = CognitiveArchitecture(version="1.0.0", mmlu_score=64.2, inference_latency_ms=45.0)
    assert arch.version == "1.0.0"
    assert arch.mmlu_score == 64.2
    assert arch.context_capacity_tokens == 8192


def test_recursive_self_modifier_proposals():
    """2. Özyinelemeli mutasyon motoru geçerli mimarî iyileştirmeler önermelidir."""
    mutations = RecursiveSelfModifier.propose_mutations()
    assert len(mutations) >= 3
    assert mutations[0]["expected_gain"] > 0
    assert mutations[0]["latency_reduction_pct"] > 0


def test_formal_proof_sandbox_zero_regression():
    """3. Biçimsel kanıt sandbox'ı pozitif fayda ve güvenlik korumasını doğrulamalıdır."""
    arch = CognitiveArchitecture(mmlu_score=64.2, harmlessness_pct=99.4)
    mutation = {"mutation_id": "MUT-TEST", "expected_gain": 5.0}
    proof = FormalProofSandbox.verify_mutation(mutation, arch)
    assert proof["proof_valid"] is True
    assert proof["regression_risk_pct"] < 1.0


def test_atomic_state_hot_swapper_evolution():
    """4. Atomik sıcak kod değişimi mimariyi v3.0.0'a yükseltmeli ve bağlamı artırmalıdır."""
    base_arch = CognitiveArchitecture()
    mutations = RecursiveSelfModifier.propose_mutations()
    new_arch = AtomicStateHotSwapper.apply_mutations(base_arch, mutations)
    assert new_arch.version == "3.0.0"
    assert new_arch.mmlu_score > base_arch.mmlu_score
    assert new_arch.context_capacity_tokens == 131072
    assert new_arch.inference_latency_ms < base_arch.inference_latency_ms


def test_profiler_mmlu_score_gain():
    """5. Kendi kendini geliştiren AGI'ın bilişsel skor kazancı en az 25 puan olmalıdır."""
    profil = SelfImprovingAGIProfilleyici.basarim_profili_cikar()
    assert profil["skor_kazanci"] >= 25.0


def test_profiler_latency_speedup():
    """6. Çıkarım gecikmesi hızlanma çarpanı 4.0x üzerinde olmalıdır."""
    profil = SelfImprovingAGIProfilleyici.basarim_profili_cikar()
    assert profil["gecikme_hizlanmasi"] >= 4.0


def test_profiler_regression_suppression():
    """7. Regresyon ve bozulma riski %0.5'in altında tutulmalıdır."""
    profil = SelfImprovingAGIProfilleyici.basarim_profili_cikar()
    risk = profil["karsilastirma"]["regresyon_ve_bozulma_riski_yuzde"]["3. Provable Self-Improving AGI"]
    assert risk < 0.5


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. SelfImprovingAGIGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_self_improving_paneli.png")
    profil = SelfImprovingAGIProfilleyici.basarim_profili_cikar()

    SelfImprovingAGIGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
