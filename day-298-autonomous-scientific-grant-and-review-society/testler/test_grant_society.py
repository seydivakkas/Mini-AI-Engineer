"""
PyTest Birim Testleri - Day 298 (FAZ 15): Otonom Bilimsel Fonlama ve Hakemler Meclisi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grant_society_motoru import (
    GrantProposal,
    AIReviewerAgent,
    ReviewPanelSociety,
    ResourceAllocationOptimizer,
)
from src.grant_society_profilleyici import GrantSocietyProfilleyici
from src.gorsellestirici import GrantSocietyGorsellestirici


def test_grant_proposal_attributes():
    """1. Proje teklifi nesnesi bütçe, başlık ve puan alanlarını doğru taşımalıdır."""
    prop = GrantProposal("P-100", "Kuantum Batarya", "Energy", 500000.0, 9.4, 9.1)
    assert prop.proposal_id == "P-100"
    assert prop.budget_requested == 500000.0
    assert prop.novelty_score == 9.4


def test_ai_reviewer_agent_evaluation():
    """2. AI hakem uzmanlık alanına göre doğru değerlendirme puanı üretmelidir."""
    agent = AIReviewerAgent("Metodoloji Hakemi", "Metodolojik Titizlik & İstatistik", 0.25)
    prop = GrantProposal("P-101", "AGI Teori", "AI", 300000.0, 9.5, 9.2)
    eval_res = agent.evaluate_proposal(prop)
    assert eval_res["score"] == 9.2
    assert eval_res["weight"] == 0.25


def test_review_panel_consensus_scoring():
    """3. 5 Hakemli Meclis ağırlıklı konsensüs puanı ve kabul kararı üretmelidir."""
    panel = ReviewPanelSociety()
    prop = GrantProposal("P-102", "Optik Kuantum Çipi", "Hardware", 800000.0, 9.6, 9.4)
    res = panel.review_proposal(prop)
    assert len(res["reviews"]) == 5
    assert res["consensus_score"] >= 8.5
    assert "FONLANDI" in res["decision"]


def test_resource_allocation_optimizer_budget_limits():
    """4. Fon tahsis motoru toplam bütçeyi ($5M) aşmadan liyakatle dağıtmalıdır."""
    p1 = GrantProposal("P-1", "Proje 1", "AI", 2000000.0, 9.5, 9.5)
    p2 = GrantProposal("P-2", "Proje 2", "Bio", 2000000.0, 9.2, 9.2)
    p3 = GrantProposal("P-3", "Proje 3", "Energy", 2000000.0, 9.0, 9.0)

    panel = ReviewPanelSociety()
    proposals = [p1, p2, p3]
    reviews = [panel.review_proposal(p) for p in proposals]

    alloc = ResourceAllocationOptimizer.allocate_funds(proposals, reviews, total_budget=5000000.0)
    assert alloc["allocated_total"] <= 5000000.0
    assert alloc["funded_count"] == 2  # 2M + 2M = 4M fonlandı, 3. proje bütçe sınırına takıldı


def test_profiler_review_speedup():
    """5. Değerlendirme hızı geleneksel heyete göre 10,000 kattan fazla olmalıdır."""
    profil = GrantSocietyProfilleyici.basarim_profili_cikar()
    assert profil["hizlanma_orani"] >= 10000.0


def test_profiler_cost_savings():
    """6. Proje inceleme maliyeti tasarrufu 10,000 katın üzerinde olmalıdır."""
    profil = GrantSocietyProfilleyici.basarim_profili_cikar()
    assert profil["maliyet_tasarrufu"] >= 10000.0


def test_profiler_merit_alignment_superiority():
    """7. Bilimsel liyakat uyumu %90'ın üzerinde olmalıdır."""
    profil = GrantSocietyProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["liyakat_ve_adil_uyum_yuzde"]["3. AI Review Society"] >= 90.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. GrantSocietyGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_grant_paneli.png")
    profil = GrantSocietyProfilleyici.basarim_profili_cikar()

    GrantSocietyGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
