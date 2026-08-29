"""
PyTest Birim Testleri - Day 288 (FAZ 15): LLM Akıl Yürütme ve Test-Time Compute (MCTS & PRM).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcts_reasoning_motoru import ThoughtNode, ProcessRewardModel, MCTSReasoningEngine
from src.mcts_reasoning_profilleyici import MCTSReasoningProfilleyici
from src.gorsellestirici import MCTSReasoningGorsellestirici


def test_thought_node_initialization():
    """1. Düşünce düğümü doğru varsayılan değerlerle başlatılmalıdır."""
    node = ThoughtNode(state_text="Başlangıç", prior=0.8)
    assert node.visits == 0
    assert node.total_value == 0.0
    assert node.q_value == 0.0
    assert node.prior == 0.8


def test_thought_node_ucb_score():
    """2. UCB puanı formülü keşfedilmemiş düğümlere pozitif arama önceliği vermelidir."""
    parent = ThoughtNode(state_text="Parent")
    parent.visits = 10
    child = ThoughtNode(state_text="Child", parent=parent, prior=0.9)
    ucb = child.ucb_score(c_puct=1.414)
    assert ucb > 0.0


def test_process_reward_model_scoring():
    """3. PRM modeli doğru adımlara yüksek (>0.9), hatalı adımlara düşük (<0.1) puan vermelidir."""
    valid_score = ProcessRewardModel.evaluate_thought_step("Subtract 6 from both sides: 2x = 8")
    error_score = ProcessRewardModel.evaluate_thought_step("Divide by 6: 2x = 20")
    assert valid_score > 0.90
    assert error_score < 0.10


def test_mcts_reasoning_search_execution():
    """4. MCTS arama motoru nihai doğru çözümü içeren akıl yürütme yolu bulmalıdır."""
    res = MCTSReasoningEngine.run_mcts_reasoning("2x + 6 = 14", num_simulations=30)
    assert res["final_solution_found"] is True
    assert len(res["best_path"]) >= 2


def test_mcts_pruning_and_expansion():
    """5. MCTS döngüsü düğüm genişletmeli ve hatalı dalları budamalıdır."""
    res = MCTSReasoningEngine.run_mcts_reasoning("2x + 6 = 14", num_simulations=20)
    assert res["expanded_nodes"] >= 3
    assert res["root_visits"] == 20


def test_profiler_reasoning_accuracy_superiority():
    """6. MCTS profilleyici Test-Time Compute doğruluğunun (%96.8) CoT'u (%52.4) aştığını doğrulamalıdır."""
    profil = MCTSReasoningProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["matematik_mantik_basarisi_yuzde"]["3. MCTS + PRM Test-Time"] > 95.0
    assert kars["matematik_mantik_basarisi_yuzde"]["2. Standard CoT"] < 60.0


def test_profiler_hallucination_reduction():
    """7. Mantıksal halüsinasyon oranı 10 kattan fazla azalmalıdır."""
    profil = MCTSReasoningProfilleyici.basarim_profili_cikar()
    assert profil["halusinasyon_azalma_orani"] >= 10.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. MCTSReasoningGorsellestirici 6 panelli teşhis panosunu başarıyla oluşturmalıdır."""
    cikti = str(tmp_path / "test_mcts_paneli.png")
    profil = MCTSReasoningProfilleyici.basarim_profili_cikar()

    MCTSReasoningGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
