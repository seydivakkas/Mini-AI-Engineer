"""
PyTest Birim Testleri - Day 295 (FAZ 15): Büyük Ölçekli Üretken Ajan Simülasyonu (Smallville).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generative_agent_motoru import (
    EpisodicMemory,
    MemoryStreamRetriever,
    GenerativeAgent,
    SocialTownSimulation,
)
from src.generative_agent_profilleyici import GenerativeAgentProfilleyici
from src.gorsellestirici import GenerativeAgentGorsellestirici


def test_episodic_memory_initialization():
    """1. Epizodik anı metin, zaman damgası ve önem puanıyla başlatılmalıdır."""
    mem = EpisodicMemory("Parti daveti aldım", 10, 0.9)
    assert mem.text == "Parti daveti aldım"
    assert mem.timestamp == 10
    assert mem.importance == 0.9


def test_memory_stream_retriever_scoring():
    """2. Bellek erişim puanlayıcı [0, 1] aralığında geçerli skor üretmelidir."""
    mem = EpisodicMemory("Kütüphanede ders çalıştım", 5, 0.8)
    score = MemoryStreamRetriever.calculate_score(mem, current_time=6, query="kütüphane")
    assert 0.0 <= score <= 1.0
    assert score > 0.4


def test_generative_agent_memory_addition():
    """3. Ajanın bellek akışına anı eklenebilmelidir."""
    agent = GenerativeAgent("Maria", "Organizatör")
    agent.add_memory("Yeni bir etkinlik planladım", timestamp=1, importance=0.85)
    assert len(agent.memory_stream) == 1
    assert agent.memory_stream[0].importance == 0.85


def test_generative_agent_reflection():
    """4. Refleksiyon motoru anılardan üst düzey soyut çıkarım üretmelidir."""
    agent = GenerativeAgent("Klaus", "Öğrenci")
    agent.add_memory("Maria partiye davet etti", timestamp=2, importance=0.9)
    insight = agent.reflect()
    assert "Refleksiyonu" in insight
    assert len(agent.reflections) == 1


def test_social_town_simulation_diffusion():
    """5. Kasaba sosyal simülasyonunda nihai bilgi yayılımı %90'ın üzerinde olmalıdır."""
    sim = SocialTownSimulation.simulate_information_diffusion()
    assert sim["final_reach_percentage"] > 90.0
    assert len(sim["diffusion_rates"]) == 4


def test_profiler_believability_superiority():
    """6. Üretken ajanın insan inandırıcılık puanı %95'in üzerinde olmalıdır."""
    profil = GenerativeAgentProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["inandiricilik_skoru_yuzde"]["3. Generative Agent"] > 95.0


def test_profiler_coherence_stability():
    """7. 24 saatlik davranış tutarlılığı %95'in üzerinde olmalıdır."""
    profil = GenerativeAgentProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["davranis_tutarliligi_yuzde"]["3. Generative Agent"] > 95.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. GenerativeAgentGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_agent_paneli.png")
    profil = GenerativeAgentProfilleyici.basarim_profili_cikar()

    GenerativeAgentGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
