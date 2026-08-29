"""
PPO ile LLM Hizalama ve Actor-Critic Testleri (Day 109).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.ppo_matematigi import hesapla_kl_cezali_odul, hesapla_gae_avantaj, PPOClippedLoss
from src.actor_critic_modelleri import ActorPolicy, CriticValueNetwork
from src.ppo_laboratuvari import PPOLaboratuvari, SahteOdulModeli
from src.gorsellestirici import PPOGorsellestirici


def test_kl_cezali_odul_hesabi():
    """hesapla_kl_cezali_odul fonksiyonunun KL cezası ve RM ödülünü birleştirdiğini test eder."""
    act_logp = torch.tensor([[-0.5, -0.8]])
    ref_logp = torch.tensor([[-0.2, -0.4]])
    rm_score = torch.tensor([2.0])

    birlesik_odul, kl_mean = hesapla_kl_cezali_odul(act_logp, ref_logp, rm_score, kl_beta=0.1)

    assert birlesik_odul.shape == (1, 2)
    # 1. token: -0.1 * (-0.5 - (-0.2)) = -0.1 * (-0.3) = +0.03
    assert round(birlesik_odul[0, 0].item(), 3) == 0.03
    # 2. token (son): -0.1 * (-0.8 - (-0.4)) + 2.0 = +0.04 + 2.0 = 2.04
    assert round(birlesik_odul[0, 1].item(), 3) == 2.04


def test_gae_avantaj_hesabi():
    """hesapla_gae_avantaj fonksiyonunun avantaj ve getirileri doğru şekil ve ortalamayla ürettiğini test eder."""
    rewards = torch.randn(4, 8)
    values = torch.randn(4, 8)

    avantajlar, getiriler = hesapla_gae_avantaj(rewards, values, gamma=1.0, lam=0.95)

    assert avantajlar.shape == (4, 8)
    assert getiriler.shape == (4, 8)
    # Normalizasyon sonucu avantaj ortalaması ~0 olmalı
    assert abs(float(avantajlar.mean().item())) < 1e-4


def test_ppo_clipped_loss_hesaplama():
    """PPOClippedLoss fonksiyonunun kırpılmış politika ve değer kaybını test eder."""
    loss_fn = PPOClippedLoss(clip_eps=0.2, vf_coef=0.5)

    logp_new = torch.tensor([[-0.5, -0.5]])
    logp_old = torch.tensor([[-0.5, -0.5]])  # Oran = 1.0
    avantajlar = torch.tensor([[1.0, -1.0]])
    v_pred = torch.tensor([[0.5, 0.5]])
    getiriler = torch.tensor([[0.8, 0.2]])

    loss, pol_loss, val_loss, clip_frac = loss_fn(logp_new, logp_old, avantajlar, v_pred, getiriler)

    assert loss.item() > 0.0
    assert clip_frac == 0.0  # Oran 1.0 olduğu için kırpılma yok


def test_actor_policy_uret_ve_logprob():
    """ActorPolicy modülünün otoregresif üretim ve log-olasılık çıktısını test eder."""
    actor = ActorPolicy(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    prompts = torch.randint(1, 99, (2, 8))

    tam_dizi, yanit_ids, logprobs = actor.uret_ve_logprob_al(prompts, max_new_tokens=6)

    assert tam_dizi.shape == (2, 14)
    assert yanit_ids.shape == (2, 6)
    assert logprobs.shape == (2, 6)
    assert not torch.isnan(logprobs).any()


def test_actor_policy_logprob_degerlendir():
    """ActorPolicy logprob_degerlendir metodunun hedef token log-olasılıklarını doğru hesapladığını test eder."""
    actor = ActorPolicy(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    tam_dizi = torch.randint(1, 99, (2, 12))

    logprobs = actor.logprob_degerlendir(tam_dizi, yanit_baslangic_idx=8)
    assert logprobs.shape == (2, 4)
    assert not torch.isnan(logprobs).any()


def test_critic_value_network():
    """CriticValueNetwork modülünün durum değeri tahminlerini test eder."""
    critic = CriticValueNetwork(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    tam_dizi = torch.randint(1, 99, (3, 10))

    values = critic(tam_dizi, yanit_baslangic_idx=6)
    assert values.shape == (3, 4)
    assert not torch.isnan(values).any()


def test_ppo_laboratuvari_adim():
    """PPOLaboratuvari tek adımlık PPO optimizasyonunu test eder."""
    lab = PPOLaboratuvari(vocab_size=200, dim=64, num_heads=2, num_layers=2, cihaz=torch.device("cpu"))
    prompts = torch.randint(1, 50, (4, 6))

    sonuc = lab.ppo_hizalama_adimi(prompts, max_new_tokens=4)
    assert "toplam_kayip" in sonuc
    assert "ortalama_odul" in sonuc
    assert "kl_sapmasi" in sonuc


def test_ppo_gorsellestirici_pano():
    """PPOGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = PPOGorsellestirici(dpi=100)
    ornek_egitim = {
        "oduller": [-0.5, 0.2, 0.8, 1.6],
        "kl_sapmalari": [0.01, 0.04, 0.08, 0.12],
        "politika_kayiplari": [0.4, 0.2, 0.1, 0.05],
        "deger_kayiplari": [0.8, 0.4, 0.2, 0.1],
        "kirpma_oranlari": [2.0, 5.5, 8.2, 10.1],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_ppo_pano.png")
        gorsellestirici.pano_olustur(ornek_egitim, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
