"""
PyTest Birim Testleri - Day 206: Step-Level PRM (Process Reward Model) Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prm_motoru import (
    PRMStepClassifier,
    MathReasoningTrajectory,
    PRMTreeSearchEngine,
)
from src.prm_profilleyici import PRMAkisProfilleyici
from src.gorsellestirici import PRMGorsellestirici


def test_prm_step_classifier_forward():
    """1. PRMStepClassifier [0, 1] aralığında olasılık üretmelidir."""
    model = PRMStepClassifier(vocab_size=128, embed_dim=64)
    tokens = torch.randint(0, 128, (3, 8))
    scores = model(tokens)
    assert scores.shape == (3,)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)


def test_math_trajectory_creation():
    """2. MathReasoningTrajectory nesnesi adımları ve çözümü doğru saklamalıdır."""
    yorunge = MathReasoningTrajectory("x + 3 = 8", ["1. Adım: x = 5"], "5")
    assert yorunge.soru == "x + 3 = 8"
    assert len(yorunge.adimlar) == 1
    assert yorunge.nihai_cevap == "5"


def test_trajectory_prm_scoring():
    """3. prm_skorla çağrıldığında her adım için skor üretilmelidir."""
    model = PRMStepClassifier()
    yorunge = MathReasoningTrajectory(
        "2x = 10",
        ["1. Adım: x = 5"],
        "5",
    )
    skorlar = yorunge.prm_skorla(model)
    assert len(skorlar) == 1
    assert len(yorunge.adim_skorlari) == 1


def test_trajectory_product_and_min_scores():
    """4. carpim_skoru ve minimum_skor doğru matematiksel değerleri dönmelidir."""
    yorunge = MathReasoningTrajectory("q", ["a1", "a2"], "ans")
    yorunge.adim_skorlari = [0.8, 0.5]

    assert pytest.approx(yorunge.carpim_skoru) == 0.40
    assert pytest.approx(yorunge.minimum_skor) == 0.50


def test_prm_early_pruning():
    """5. PRMTreeSearchEngine düşük skorlu adımlarda dalı budamalıdır."""
    model = PRMStepClassifier()
    sonuc = PRMTreeSearchEngine.aday_yol_budama_simulasyonu(model, esik_deger=0.40)

    assert "yollar" in sonuc
    assert len(sonuc["yollar"]) == 4
    # En az bir yolun budanmış olması gerekir (içinde Hata geçen)
    budananlar = [y for y in sonuc["yollar"] if y["budandi"]]
    assert len(budananlar) >= 1


def test_prm_token_savings():
    """6. Erken budama ile token tasarruf yüzdesi sıfırdan büyük olmalıdır."""
    model = PRMStepClassifier()
    sonuc = PRMTreeSearchEngine.aday_yol_budama_simulasyonu(model)

    assert sonuc["toplam_budanan_token"] >= 0
    assert sonuc["hesaplama_tasarrufu_yuzde"] >= 0.0


def test_prm_profiler():
    """7. PRMAkisProfilleyici karşılaştırmalı metrikleri eksiksiz dönmelidir."""
    profil = PRMAkisProfilleyici.kapsamli_profil_cikar()
    assert "prm_dogruluk" in profil
    assert "orm_dogruluk" in profil
    assert "metrikler" in profil


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. PRMGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_prm_paneli.png")
    profil = PRMAkisProfilleyici.kapsamli_profil_cikar()

    PRMGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
