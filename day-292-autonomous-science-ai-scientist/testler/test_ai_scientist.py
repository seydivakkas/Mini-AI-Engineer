"""
PyTest Birim Testleri - Day 292 (FAZ 15): Otonom Bilimsel Keşif (The AI Scientist).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_scientist_motoru import (
    ResearchIdea,
    LiteratureNoveltyChecker,
    ExperimentRunner,
    LaTeXPaperGenerator,
    AutonomousPeerReviewer,
)
from src.ai_scientist_profilleyici import AIScientistProfilleyici
from src.gorsellestirici import AIScientistGorsellestirici


def test_research_idea_initialization():
    """1. Araştırma fikri başlık, hipotez ve özgünlük puanıyla başlatılmalıdır."""
    idea = ResearchIdea("Adaptive Entropy Attention", "Saves compute", novelty_score=0.91)
    assert idea.title == "Adaptive Entropy Attention"
    assert idea.novelty_score == 0.91


def test_literature_novelty_checker():
    """2. Literatür tarama motoru geçerli özgünlük puanı (>0.80) döndürmelidir."""
    idea = ResearchIdea("Dynamic Entropy Gating", "Hypothesis", novelty_score=0.94)
    novelty = LiteratureNoveltyChecker.check_novelty(idea)
    assert novelty >= 0.80


def test_experiment_runner_convergence():
    """3. Deney koşturucu yakınsayan kayıp eğrileri ve doğruluk artışı üretmelidir."""
    idea = ResearchIdea("Idea", "Hyp")
    exp = ExperimentRunner.run_experiment(idea)
    assert exp["proposed_loss"][-1] < exp["proposed_loss"][0]
    assert exp["proposed_acc"] > exp["baseline_acc"]
    assert exp["compute_savings"] > 50.0


def test_latex_paper_generator_structure():
    """4. Otomatik LaTeX üreteci tüm standart akademik bölümleri içermelidir."""
    idea = ResearchIdea("Test Title", "Test Hypothesis")
    exp = ExperimentRunner.run_experiment(idea)
    tex = LaTeXPaperGenerator.generate_latex(idea, exp)
    assert "\\documentclass{article}" in tex
    assert "\\begin{abstract}" in tex
    assert "\\section{Methodology}" in tex
    assert "\\section{Experiments}" in tex


def test_autonomous_peer_reviewer_scoring():
    """5. Otonom hakem sağlamlık > 9.0 ve STRONG ACCEPT kararı vermelidir."""
    tex = "\\documentclass{article} ... \\end{document}"
    review = AutonomousPeerReviewer.review_paper(tex)
    assert review["soundness"] >= 9.0
    assert "STRONG ACCEPT" in review["decision"]


def test_profiler_cycle_latency_speedup():
    """6. Araştırma döngüsü hızlanma oranı 10,000 kattan fazla olmalıdır."""
    profil = AIScientistProfilleyici.basarim_profili_cikar()
    assert profil["hizlanma_orani"] >= 10000.0


def test_profiler_soundness_superiority():
    """7. AI Scientist metodolojik sağlamlık skoru %90'ın üzerinde olmalıdır."""
    profil = AIScientistProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["metodolojik_saglamlik_yuzde"]["3. AI Scientist"] > 90.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. AIScientistGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_science_paneli.png")
    profil = AIScientistProfilleyici.basarim_profili_cikar()

    AIScientistGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
