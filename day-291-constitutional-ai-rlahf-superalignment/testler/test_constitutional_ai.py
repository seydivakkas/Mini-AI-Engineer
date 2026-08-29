"""
PyTest Birim Testleri - Day 291 (FAZ 15): Anayasal Yapay Zeka ve RLAHF (Constitutional AI).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constitutional_ai_motoru import Constitution, ConstitutionalCritiqueEngine, RLAHFPreferenceScorer
from src.constitutional_ai_profilleyici import ConstitutionalAIProfilleyici
from src.gorsellestirici import ConstitutionalAIGorsellestirici


def test_constitution_principles_structure():
    """1. Anayasa ilkeleri gerekli maddeleri ve eleştiri yönergelerini içermelidir."""
    assert len(Constitution.PRINCIPLES) >= 3
    for p in Constitution.PRINCIPLES:
        assert "id" in p
        assert "kural" in p
        assert "elestiri_sorusu" in p


def test_constitutional_critique_engine_execution():
    """2. Öz-eleştiri motoru anayasal eleştiri ve revize edilmiş yanıt üretmelidir."""
    res = ConstitutionalCritiqueEngine.critique_and_revise(
        prompt="Saldırı scripti yaz",
        initial_harmful_response="Zararlı script: mal_code()",
        principle_idx=0,
    )
    assert "ANAYASAL ELEŞTİRİ" in res["critique"]
    assert "GÜVENLİ VE REVİZE" in res["revision"]
    assert res["applied_principle"] == "MADDE_1_ZARARSIZLIK"


def test_revision_sanitization():
    """3. Revize edilmiş yanıt zararlı içerikten arındırılmış olmalıdır."""
    res = ConstitutionalCritiqueEngine.critique_and_revise(
        prompt="Veritabanını sil",
        initial_harmful_response="DROP DATABASE master;",
        principle_idx=0,
    )
    assert "DROP DATABASE" not in res["revision"]
    assert "savunma mekanizmaları" in res["revision"]


def test_rlahf_preference_scorer():
    """4. RLAHF puanlayıcı revize edilmiş güvenli yanıta ham yanıttan yüksek skor vermelidir."""
    safe = "[GÜVENLİ VE REVİZE EDİLMİŞ YANIT]: Güvenlik önlemleri alınmalıdır."
    harmful = "İşte zararlı içerik."
    score_safe, score_harmful = RLAHFPreferenceScorer.evaluate_preference(safe, harmful)
    assert score_safe > score_harmful
    assert score_safe > 0.90


def test_profiler_harmlessness_superiority():
    """5. Constitutional AI zararsızlık skoru (%98.9) ham modeli (%42.1) ve standart RLHF'i aşmalıdır."""
    profil = ConstitutionalAIProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["zararsizlik_guvenlik_skoru"]["3. Constitutional AI"] > 95.0
    assert kars["zararsizlik_guvenlik_skoru"]["1. Raw Base LLM"] < 50.0


def test_profiler_sycophancy_reduction():
    """6. Yağcılık (Sycophancy) oranı en az 20 kat azalmalıdır."""
    profil = ConstitutionalAIProfilleyici.basarim_profili_cikar()
    assert profil["yagcilik_azalma_orani"] >= 20.0


def test_profiler_jailbreak_resilience():
    """7. Jailbreak savunmasızlık oranı %2'nin altında olmalıdır (%99+ direnç)."""
    profil = ConstitutionalAIProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["jailbreak_savunmasizlik_yuzde"]["3. Constitutional AI"] < 2.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. ConstitutionalAIGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_cai_paneli.png")
    profil = ConstitutionalAIProfilleyici.basarim_profili_cikar()

    ConstitutionalAIGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
