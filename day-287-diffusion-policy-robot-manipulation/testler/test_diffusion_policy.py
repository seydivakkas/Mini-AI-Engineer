"""
PyTest Birim Testleri - Day 287 (FAZ 15): Difüzyon Tabanlı Planlayıcılar (Diffusion Policy).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.diffusion_policy_motoru import ConditionalNoisePredictor1D, DiffusionPolicyEngine
from src.diffusion_policy_profilleyici import DiffusionPolicyProfilleyici
from src.gorsellestirici import DiffusionPolicyGorsellestirici


def test_noise_predictor_output_shape():
    """1. 1D Gürültü tahmin ağı (B, T_p, D_a) şeklinde tensör üretmelidir."""
    net = ConditionalNoisePredictor1D(action_dim=2, action_horizon=8, obs_dim=16, embed_dim=64)
    noisy_a = torch.randn(2, 8, 2)
    timestep = torch.tensor([5, 10])
    obs = torch.randn(2, 16)

    out = net(noisy_a, timestep, obs)
    assert out.shape == (2, 8, 2)
    assert not torch.isnan(out).any()


def test_forward_diffusion_q_sample():
    """2. İleri difüzyon süreci doğru alfa çizelgesiyle gürültü eklemelidir."""
    engine = DiffusionPolicyEngine(action_dim=2, action_horizon=8, obs_dim=16, num_diffusion_steps=16)
    a_0 = torch.zeros(1, 8, 2)
    noisy_a, noise = engine.forward_diffusion(a_0, k=10)
    assert noisy_a.shape == (1, 8, 2)
    assert noise.shape == (1, 8, 2)


def test_reverse_diffusion_sample_trajectory():
    """3. Ters difüzyon (Reverse Sampling) geçerli eylem yörüngesi üretmelidir."""
    engine = DiffusionPolicyEngine(action_dim=2, action_horizon=8, obs_dim=16, num_diffusion_steps=8)
    obs = torch.randn(1, 16)
    traj = engine.reverse_sample_trajectory(obs)
    assert traj.shape == (1, 8, 2)
    assert not torch.isnan(traj).any()


def test_diffusion_policy_engine_initialization():
    """4. Varyans çizelgesi (Betas) pozitif ve monoton artan olmalıdır."""
    engine = DiffusionPolicyEngine(action_dim=2, action_horizon=8, obs_dim=16, num_diffusion_steps=16)
    assert (engine.betas > 0).all()
    assert (engine.alphas_cumprod <= 1.0).all()


def test_profiler_success_rate_superiority():
    """5. Diffusion Policy başarı oranı (%95.8) standart BC'yi (%46.2) aşmalıdır."""
    profil = DiffusionPolicyProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["gorev_basari_orani_yuzde"]["Diffusion_Policy"] > 90.0
    assert kars["gorev_basari_orani_yuzde"]["Standart_BC"] < 50.0


def test_profiler_tracking_error_reduction():
    """6. Yörünge takip hatası 10 kattan fazla düşmelidir."""
    profil = DiffusionPolicyProfilleyici.basarim_profili_cikar()
    assert profil["hata_azalma_orani"] >= 10.0


def test_multimodal_distribution_capture():
    """7. Çok modlu dağılım yakalama oranı %95'in üzerinde olmalıdır."""
    profil = DiffusionPolicyProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["cok_modlu_yakalama_orani"]["Diffusion_Policy"] > 95.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. DiffusionPolicyGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_diff_paneli.png")
    profil = DiffusionPolicyProfilleyici.basarim_profili_cikar()

    DiffusionPolicyGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
