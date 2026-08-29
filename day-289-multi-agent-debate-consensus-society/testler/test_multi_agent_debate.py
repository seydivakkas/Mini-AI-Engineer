"""
PyTest Birim Testleri - Day 289 (FAZ 15): Çok Modlu Çoklu Ajan Tartışması ve Konsensüs (MAD).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.multi_agent_debate_motoru import AgentPersona, MultiAgentDebateSociety
from src.multi_agent_debate_profilleyici import MultiAgentDebateProfilleyici
from src.gorsellestirici import MultiAgentDebateGorsellestirici


def test_agent_persona_initialization():
    """1. Ajan personajı isim, rol ve Elo derecelendirmesiyle başlatılmalıdır."""
    agent = AgentPersona(name="Ajan Alfa", role="proposer", elo_rating=1600.0)
    assert agent.name == "Ajan Alfa"
    assert agent.role == "proposer"
    assert agent.elo_rating == 1600.0


def test_agent_persona_elo_weight():
    """2. Yüksek Elo derecesine sahip ajan daha yüksek güvenilirlik ağırlığı almalıdır."""
    agent_low = AgentPersona(name="Ajan Çırak", role="critic", elo_rating=1400.0)
    agent_high = AgentPersona(name="Ajan Usta", role="judge", elo_rating=1800.0)
    assert agent_high.get_weight(avg_elo=1500.0) > agent_low.get_weight(avg_elo=1500.0)


def test_multi_agent_debate_round_execution():
    """3. Çoklu ajan tartışması 3 turlu diyalektik konuşma transkripti üretmelidir."""
    res = MultiAgentDebateSociety.run_debate("Mikroservis Mimarisi Tasarımı", num_rounds=3)
    assert len(res["transcript"]) == 3
    assert res["transcript"][0]["round"] == 1
    assert res["transcript"][2]["round"] == 3


def test_debate_confidence_progression():
    """4. Tartışma turları ilerledikçe konsensüs güven puanı artmalıdır."""
    res = MultiAgentDebateSociety.run_debate("Sistem Mimarisi", num_rounds=3)
    conf = res["confidence_curve"]
    assert len(conf) == 4
    assert conf[-1] > conf[0]
    assert conf[-1] > 0.90


def test_consensus_reached_and_verdict():
    """5. Tartışma sonunda hakem tarafından sentezlenmiş konsensüs kararı üretilmelidir."""
    res = MultiAgentDebateSociety.run_debate("Veri Tutarlılığı", num_rounds=3)
    assert res["consensus_reached"] is True
    assert "Konsensüs Kararı" in res["final_verdict"]


def test_profiler_accuracy_superiority():
    """6. Çoklu ajan tartışması (%97.4) tek ajana (%61.5) göre belirgin doğruluk üstünlüğü sağlamalıdır."""
    profil = MultiAgentDebateProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["muhakeme_basarisi_yuzde"]["3. Multi-Agent Debate"] > 95.0
    assert kars["muhakeme_basarisi_yuzde"]["1. Single Agent"] < 65.0


def test_profiler_hallucination_reduction():
    """7. Halüsinasyon oranı 15 kattan fazla azalmalıdır (%38.6 -> %2.1)."""
    profil = MultiAgentDebateProfilleyici.basarim_profili_cikar()
    assert profil["halusinasyon_azalma_orani"] >= 15.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. MultiAgentDebateGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_debate_paneli.png")
    profil = MultiAgentDebateProfilleyici.basarim_profili_cikar()

    MultiAgentDebateGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
