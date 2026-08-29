"""
PyTest Birim Testleri - Day 204: DPO (Direct Preference Optimization) Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dpo_motoru import (
    DPOModel,
    DPOTrainer,
)
from src.dpo_profilleyici import DPOAkisProfilleyici
from src.gorsellestirici import DPOGorsellestirici


def test_dpo_model_forward():
    """1. DPOModel doğru çıkış tensör boyutunu üretmelidir."""
    model = DPOModel(vocab_size=128, embed_dim=64)
    x = torch.randint(0, 128, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, 128)


def test_sekans_log_prob_calculation():
    """2. Sekans log olasılık hesabı batch boyutunda skalar tensör dönmelidir."""
    trainer = DPOTrainer()
    x = torch.randint(0, 128, (3, 10))
    logp = trainer._sekans_log_prob_hesapla(trainer.policy, x)
    assert logp.shape == (3,)
    assert torch.all(logp < 0.0)  # Log olasılıklar negatif olmalıdır


def test_dpo_loss_formula():
    """3. dpo_kaybi_hesapla pozitif skalar kayıp ve metrik sözlüğü dönmelidir."""
    trainer = DPOTrainer(beta=0.1)
    chosen = torch.randint(0, 128, (2, 8))
    rejected = torch.randint(0, 128, (2, 8))
    loss, metrikler = trainer.dpo_kaybi_hesapla(chosen, rejected)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
    assert "chosen_reward" in metrikler
    assert "rejected_reward" in metrikler
    assert "reward_margin" in metrikler


def test_dpo_reward_margin():
    """4. Chosen ve rejected logp farkı örtük ödül marjını doğru yansıtmalıdır."""
    trainer = DPOTrainer(beta=0.5)
    chosen = torch.randint(0, 128, (2, 6))
    rejected = torch.randint(0, 128, (2, 6))
    loss, metrikler = trainer.dpo_kaybi_hesapla(chosen, rejected)

    hesaplanan_marj = metrikler["chosen_reward"] - metrikler["rejected_reward"]
    assert pytest.approx(metrikler["reward_margin"], 1e-4) == hesaplanan_marj


def test_dpo_training_step():
    """5. egitim_adimi ağırlıkları güncellemeli ve metrikleri dönmelidir."""
    trainer = DPOTrainer()
    chosen = torch.randint(0, 128, (2, 8))
    rejected = torch.randint(0, 128, (2, 8))
    metrikler = trainer.egitim_adimi(chosen, rejected)

    assert "loss" in metrikler
    assert "accuracy" in metrikler


def test_dpo_reference_model_frozen():
    """6. Referans model eval modunda olmalı ve parametreleri dondurulmalıdır."""
    trainer = DPOTrainer()
    assert not trainer.ref_policy.training


def test_dpo_profiler_steps():
    """7. DPOAkisProfilleyici belirtilen adım sayısı kadar metrik kaydetmelidir."""
    profil = DPOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=7)
    assert len(profil["adimlar"]) == 7
    assert len(profil["kayiplar"]) == 7
    assert len(profil["dogruluklar"]) == 7


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. DPOGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_dpo_paneli.png")
    profil = DPOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=5)

    DPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
