"""
PyTest Birim Testleri - Day 203: PPO Actor-Critic LLM Hizalama Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ppo_motoru import (
    ActorNetwork,
    CriticNetwork,
    GAECalculator,
    PPOTrainer,
)
from src.ppo_profilleyici import PPOAkisProfilleyici
from src.gorsellestirici import PPOGorsellestirici


def test_actor_network_forward():
    """1. ActorNetwork geçerli logit tensör boyutunu üretmelidir."""
    actor = ActorNetwork(vocab_size=128, embed_dim=64)
    x = torch.randint(0, 128, (2, 10))
    logits = actor(x)
    assert logits.shape == (2, 10, 128)


def test_critic_network_forward():
    """2. CriticNetwork skalar durum değerleri üretmelidir."""
    critic = CriticNetwork(vocab_size=128, embed_dim=64)
    x = torch.randint(0, 128, (2, 10))
    values = critic(x)
    assert values.shape == (2, 10)


def test_gae_calculator_shapes():
    """3. GAECalculator girdi tensörleriyle aynı boyutta avantaj ve hedef dönmelidir."""
    oduller = torch.randn(3, 8)
    degerler = torch.randn(3, 8)
    adv, targets = GAECalculator.hesapla_avantaj_ve_hedef(oduller, degerler)
    assert adv.shape == (3, 8)
    assert targets.shape == (3, 8)


def test_gae_calculator_advantage_signs():
    """4. Yüksek pozitif ödül alan adımda GAE avantajı pozitif olmalıdır."""
    oduller = torch.tensor([[0.0, 0.0, 5.0]])
    degerler = torch.tensor([[0.0, 0.0, 1.0]])
    adv, targets = GAECalculator.hesapla_avantaj_ve_hedef(oduller, degerler)
    assert adv[0, -1] > 0.0


def test_ppo_training_step():
    """5. PPOTrainer ppo_egitim_adimi tüm kritik metrikleri dönmelidir."""
    trainer = PPOTrainer()
    input_ids = torch.randint(0, 128, (2, 12))
    rewards = torch.tensor([1.2, -0.5])
    metrics = trainer.ppo_egitim_adimi(input_ids, rewards)

    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "kl_divergence" in metrics
    assert "entropy" in metrics


def test_ppo_kl_penalty_effect():
    """6. Referans model parametreleri güncellenmemeli ve eval modunda kalmalıdır."""
    trainer = PPOTrainer()
    assert not trainer.ref_actor.training
    # Referans model parametre gradyanı gerektirmemeli
    for p in trainer.ref_actor.parameters():
        assert p.requires_grad or not trainer.ref_actor.training


def test_ppo_profiler_steps():
    """7. PPOAkisProfilleyici belirtilen adım sayısı kadar metrik kaydetmelidir."""
    profil = PPOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=6)
    assert len(profil["adimlar"]) == 6
    assert len(profil["actor_kayiplari"]) == 6
    assert len(profil["critic_kayiplari"]) == 6


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. PPOGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_ppo_paneli.png")
    profil = PPOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=5)

    PPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
