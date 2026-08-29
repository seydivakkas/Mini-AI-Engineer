"""
PyTest Birim Testleri - Day 226: Çoklu Ajan Orkestrasyonu (Swarm) Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.swarm_motoru import (
    AgentMessage,
    SpecializedAgent,
    ResearcherAgent,
    CoderAgent,
    ReviewerAgent,
    SwarmOrchestrator,
)
from src.swarm_profilleyici import SwarmProfilleyici
from src.gorsellestirici import SwarmGorsellestirici


def test_agent_message_formatting():
    """1. AgentMessage nesnesi gönderen ve alıcıyı doğru biçimlendirmelidir."""
    msg = AgentMessage("AjanA", "AjanB", "Veri hazır")
    assert "[AjanA -> AjanB]: Veri hazır" == msg.format_metni()


def test_researcher_agent_execution():
    """2. ResearcherAgent hedefi inceleyip O(N log N) teorik spesifikasyon üretmelidir."""
    agent = ResearcherAgent()
    cikti = agent.gorev_calistir("Sıralama Algoritması")
    assert "O(N log N)" in cikti
    assert agent.rol == "Araştırmacı"


def test_coder_agent_execution():
    """3. CoderAgent geçerli ve çalıştırılabilir Python kodu üretmelidir."""
    agent = CoderAgent()
    kod = agent.gorev_calistir("Sıralama")
    assert "def hizli_sirala" in kod
    assert agent.rol == "Kodlayıcı"


def test_reviewer_agent_execution():
    """4. ReviewerAgent kodu inceleyip onay raporu vermelidir."""
    agent = ReviewerAgent()
    onay = agent.gorev_calistir("def test(): pass")
    assert "ONAYLANDI" in onay
    assert agent.rol == "Denetçi"


def test_orchestrator_initialization():
    """5. SwarmOrchestrator tüm uzman ajanları eksiksiz kaydetmelidir."""
    ork = SwarmOrchestrator()
    assert "Araştırmacı" in ork.ajanlar
    assert "Kodlayıcı" in ork.ajanlar
    assert "Denetçi" in ork.ajanlar


def test_swarm_end_to_end_collaboration():
    """6. SwarmOrchestrator uçtan uca hiyerarşik işbirliğiyle rapor üretmelidir."""
    ork = SwarmOrchestrator()
    sonuc = ork.gorev_dagit_ve_sentezle("Hızlı Sıralama")
    assert sonuc["basarili_mi"] is True
    assert sonuc["toplam_mesaj_sayisi"] == 6
    assert "SWARM PROJE RAPORU" in sonuc["nihai_cikti"]


def test_profiler_swarm_metrics():
    """7. Profilleyici Swarm mimarisinin başarı oranının %90 üstünde olduğunu göstermelidir."""
    prof = SwarmProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["karmasik_proje_basari_orani"]["Hiyerarsik_Swarm"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. SwarmGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_swarm_paneli.png")
    profil = SwarmProfilleyici.basarim_profili_cikar()

    SwarmGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
