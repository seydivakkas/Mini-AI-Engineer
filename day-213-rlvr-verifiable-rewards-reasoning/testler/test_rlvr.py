"""
PyTest Birim Testleri - Day 213: RLVR (Reinforcement Learning with Verifiable Rewards) Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rlvr_motoru import (
    VerifiableTaskRegistry,
    GroundTruthVerifier,
    RLVRRewardCalculator,
    RLVRExplorationEngine,
    RLVRTrainer,
)
from src.rlvr_profilleyici import RLVRProfilleyici
from src.gorsellestirici import RLVRGorsellestirici


def test_task_registry_load():
    """1. Biçimsel görev havuzu görevleri eksiksiz yüklemelidir."""
    g = VerifiableTaskRegistry.gorev_getir(0)
    assert "soru" in g
    assert "hedef_cevap" in g


def test_ground_truth_verifier_correct():
    """2. GroundTruthVerifier doğru cevabı onaylamalıdır."""
    yanit = "<think>Adımlar...</think>\nSonuç: \\boxed{42}"
    assert GroundTruthVerifier.dogrula(yanit, "42") is True


def test_ground_truth_verifier_incorrect():
    """3. GroundTruthVerifier yanlış cevabı reddetmelidir."""
    yanit = "<think>Adımlar...</think>\nSonuç: \\boxed{17}"
    assert GroundTruthVerifier.dogrula(yanit, "42") is False


def test_reward_calculator_full_reward():
    """4. Doğru ve formatlı yanıtta tam ödül (1.20) üretilmelidir."""
    yanit = "<think>Mantık</think>\nSonuç: \\boxed{9}"
    odul = RLVRRewardCalculator.odul_hesapla(yanit, "9")

    assert odul["r_acc"] == 1.0
    assert odul["r_fmt"] == 0.20
    assert odul["dogru_mu"] is True


def test_reward_calculator_length_penalty():
    """5. Maksimum uzunluğu aşan yanıtlarda uzunluk cezası uygulanmalıdır."""
    uzun_yanit = "<think>" + "A" * 600 + "</think>\nSonuç: \\boxed{9}"
    odul = RLVRRewardCalculator.odul_hesapla(uzun_yanit, "9", maks_uzunluk=500, beta_uzunluk=0.01)

    assert odul["r_len"] < 0.0


def test_exploration_engine_aha_moment():
    """6. RLVRExplorationEngine düşünce izi üretmelidir."""
    dusunce = RLVRExplorationEngine.aday_cozum_uret("Soru", "5", aha_ani_olsun_mu=True)
    assert "<think>" in dusunce
    assert "\\boxed{5}" in dusunce


def test_rlvr_trainer_zero_variance():
    """7. RLVRTrainer deterministik sıfır varyans döndürmelidir."""
    adim = RLVRTrainer.egitim_adimi("Soru", "10")
    assert adim["odul_varyansi"] == 0.00
    assert "odul_raporu" in adim


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. RLVRGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_rlvr_paneli.png")
    profil = RLVRProfilleyici.karsilastirma_raporu_uret()

    RLVRGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
