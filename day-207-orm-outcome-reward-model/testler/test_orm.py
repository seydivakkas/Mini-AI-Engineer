"""
PyTest Birim Testleri - Day 207: ORM (Outcome Reward Model) ve Best-of-N Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orm_motoru import (
    OutcomeRewardModel,
    ORMTrainer,
    BestOfNRanker,
)
from src.orm_profilleyici import ORMAkisProfilleyici
from src.gorsellestirici import ORMGorsellestirici


def test_orm_model_forward():
    """1. OutcomeRewardModel skalar ödül tensörü üretmelidir."""
    model = OutcomeRewardModel(vocab_size=128, embed_dim=64)
    tokens = torch.randint(0, 128, (3, 10))
    scores = model(tokens)
    assert scores.shape == (3,)


def test_orm_loss_computation():
    """2. ciftli_kayip_hesapla pozitif skalar kayıp dönmelidir."""
    trainer = ORMTrainer()
    chosen = torch.randint(0, 128, (2, 8))
    rejected = torch.randint(0, 128, (2, 8))
    loss, metrikler = trainer.ciftli_kayip_hesapla(chosen, rejected)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
    assert "reward_margin" in metrikler


def test_orm_training_step():
    """3. egitim_adimi ağırlıkları güncellemeli ve metrikleri dönmelidir."""
    trainer = ORMTrainer()
    chosen = torch.randint(0, 128, (2, 8))
    rejected = torch.randint(0, 128, (2, 8))
    metrikler = trainer.egitim_adimi(chosen, rejected)

    assert "loss" in metrikler
    assert "accuracy" in metrikler


def test_best_of_n_ranker_selection():
    """4. BestOfNRanker en yüksek skorlu adayı seçmelidir."""
    model = OutcomeRewardModel()
    adaylar = [
        {"idx": 1, "metin": "Cevap: 5", "dogru_mu": False},
        {"idx": 2, "metin": "Cevap: 10 (Doğru)", "dogru_mu": True},
    ]
    secim = BestOfNRanker.en_iyi_yaniti_sec(model, adaylar)

    assert "kazanan" in secim
    assert secim["kazanan"]["idx"] == 2
    assert secim["secilen_dogru_mu"] is True


def test_best_of_n_empty_exception():
    """5. Boş aday listesi verildiğinde ValueError fırlatılmalıdır."""
    model = OutcomeRewardModel()
    with pytest.raises(ValueError):
        BestOfNRanker.en_iyi_yaniti_sec(model, [])


def test_orm_profiler_keys():
    """6. ORMAkisProfilleyici ölçekleme metriklerini eksiksiz dönmelidir."""
    profil = ORMAkisProfilleyici.olcekleme_profilini_cikar()
    assert "n_degerleri" in profil
    assert "pass_at_1_oranlari" in profil
    assert len(profil["n_degerleri"]) == 7


def test_orm_scaling_monotonicity():
    """7. N sayısı arttıkça Pass@1 oranı monotonik olarak artmalıdır."""
    profil = ORMAkisProfilleyici.olcekleme_profilini_cikar()
    pass_at_1 = profil["pass_at_1_oranlari"]
    assert pass_at_1[-1] > pass_at_1[0]
    # Her adımda genel artış eğilimi
    for i in range(len(pass_at_1) - 1):
        assert pass_at_1[i+1] >= pass_at_1[i]


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. ORMGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_orm_paneli.png")
    profil = ORMAkisProfilleyici.olcekleme_profilini_cikar()

    ORMGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
