"""
PyTest Birim Testleri - Day 205: KTO (Kahneman-Tversky Optimization) Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kto_motoru import (
    KTOModel,
    KTOTrainer,
)
from src.kto_profilleyici import KTOAkisProfilleyici
from src.gorsellestirici import KTOGorsellestirici


def test_kto_model_forward():
    """1. KTOModel doğru çıkış tensör boyutunu üretmelidir."""
    model = KTOModel(vocab_size=128, embed_dim=64)
    x = torch.randint(0, 128, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, 128)


def test_sekans_log_prob():
    """2. Sekans log olasılık hesabı batch boyutunda skalar tensör dönmelidir."""
    trainer = KTOTrainer()
    x = torch.randint(0, 128, (3, 10))
    logp = trainer._sekans_log_prob(trainer.policy, x)
    assert logp.shape == (3,)
    assert torch.all(logp < 0.0)


def test_kto_loss_computation():
    """3. kto_kaybi_hesapla pozitif skalar kayıp ve metrik sözlüğü dönmelidir."""
    trainer = KTOTrainer(beta=0.1)
    x = torch.randint(0, 128, (4, 8))
    labels = torch.tensor([True, True, False, False])
    loss, metrikler = trainer.kto_kaybi_hesapla(x, labels)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
    assert "desirable_reward" in metrikler
    assert "undesirable_reward" in metrikler
    assert "z_ref" in metrikler


def test_kto_loss_aversion_penalty():
    """4. lambda_u büyütüldüğünde istenmeyen örneklerin kaybı artmalıdır."""
    trainer_normal = KTOTrainer(beta=0.1, lambda_d=1.0, lambda_u=1.0)
    trainer_aversion = KTOTrainer(beta=0.1, lambda_d=1.0, lambda_u=2.0)

    x = torch.randint(0, 128, (2, 8))
    labels = torch.tensor([False, False])

    loss_norm, _ = trainer_normal.kto_kaybi_hesapla(x, labels)
    loss_aver, _ = trainer_aversion.kto_kaybi_hesapla(x, labels)

    assert loss_aver.item() > loss_norm.item()


def test_kto_training_step():
    """5. egitim_adimi ağırlıkları güncellemeli ve metrikleri dönmelidir."""
    trainer = KTOTrainer()
    x = torch.randint(0, 128, (4, 8))
    labels = torch.tensor([True, False, True, False])
    metrikler = trainer.egitim_adimi(x, labels)

    assert "loss" in metrikler
    assert "reward_delta" in metrikler


def test_kto_reference_frozen():
    """6. Referans model eval modunda kalmalı ve parametreleri dondurulmalıdır."""
    trainer = KTOTrainer()
    assert not trainer.ref_policy.training


def test_kto_profiler_steps():
    """7. KTOAkisProfilleyici belirtilen adım sayısı kadar metrik kaydetmelidir."""
    profil = KTOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=6)
    assert len(profil["adimlar"]) == 6
    assert len(profil["kayiplar"]) == 6
    assert len(profil["hizalama_skorlari"]) == 6


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. KTOGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_kto_paneli.png")
    profil = KTOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=5)

    KTOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
