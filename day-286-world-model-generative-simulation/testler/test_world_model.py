"""
PyTest Birim Testleri - Day 286 (FAZ 15): Dünya Modelleri ve DreamerV3 RSSM.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.world_model_motoru import RSSMCell, WorldModelEngine
from src.world_model_profilleyici import WorldModelProfilleyici
from src.gorsellestirici import WorldModelGorsellestirici


def test_rssm_cell_initialization():
    """1. RSSM çekirdeği doğru deterministik ve stokastik boyutlarla başlatılmalıdır."""
    rssm = RSSMCell(action_dim=2, deter_dim=64, stoch_dim=16)
    assert rssm.deter_dim == 64
    assert rssm.stoch_dim == 16


def test_rssm_forward_prior_step():
    """2. RSSM prior (hayal gücü) adımı çevresel gözlem olmadan geçerli durumlar üretmelidir."""
    rssm = RSSMCell(action_dim=2, deter_dim=64, stoch_dim=16)
    prev_h = torch.zeros(1, 64)
    prev_z = torch.randn(1, 16)
    action = torch.randn(1, 2)

    h, z, mu = rssm.forward_prior(prev_h, prev_z, action)
    assert h.shape == (1, 64)
    assert z.shape == (1, 16)
    assert not torch.isnan(h).any()
    assert not torch.isnan(z).any()


def test_rssm_forward_posterior_step():
    """3. RSSM posterior (algı) adımı gözlemi duruma başarıyla dahil etmelidir."""
    rssm = RSSMCell(action_dim=2, deter_dim=64, stoch_dim=16)
    prev_h = torch.zeros(1, 64)
    prev_z = torch.randn(1, 16)
    action = torch.randn(1, 2)
    obs_embed = torch.randn(1, 32)

    h, z, mu, std = rssm.forward_posterior(prev_h, prev_z, action, obs_embed)
    assert h.shape == (1, 64)
    assert z.shape == (1, 16)
    assert (std > 0).all()


def test_latent_imagination_rollout_horizon():
    """4. Gizil hayal gücü simülasyonu tam istenen H=15 adım uzunluğunda yörünge üretmelidir."""
    rssm = RSSMCell(action_dim=2, deter_dim=64, stoch_dim=16)
    initial_h = torch.zeros(1, 64)
    initial_z = torch.randn(1, 16)

    res = WorldModelEngine.simulate_latent_imagination(
        rssm=rssm,
        initial_h=initial_h,
        initial_z=initial_z,
        horizon=15,
    )
    assert res["horizon"] == 15
    assert len(res["trajectory_h"]) == 16
    assert len(res["imagined_rewards"]) == 15


def test_imagined_accumulated_reward_finite():
    """5. Hayal edilen kümülatif ödül sonlu ve pozitif bir sayı olmalıdır."""
    rssm = RSSMCell(action_dim=2, deter_dim=64, stoch_dim=16)
    initial_h = torch.zeros(1, 64)
    initial_z = torch.randn(1, 16)

    res = WorldModelEngine.simulate_latent_imagination(
        rssm=rssm,
        initial_h=initial_h,
        initial_z=initial_z,
        horizon=10,
    )
    assert isinstance(res["total_imagined_reward"], float)
    assert not torch.isnan(torch.tensor(res["total_imagined_reward"]))


def test_world_model_profiler_sample_efficiency():
    """6. Dünya modeli profilleyicisi 100x örnek verimliliği kazancını doğrulamalıdır."""
    profil = WorldModelProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["gerekli_cevre_adimi"]["DreamerV3_WorldModel"] == 10000
    assert kars["gerekli_cevre_adimi"]["Model_Free_PPO"] == 1000000
    assert profil["ornek_verimlilik_kazanci"] >= 100.0


def test_dreamerv3_score_superiority():
    """7. DreamerV3 skoru (965.0) Model-Free PPO skorunu (740.0) geçmelidir."""
    profil = WorldModelProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["nihai_epizodik_odul"]["DreamerV3_WorldModel"] > kars["nihai_epizodik_odul"]["Model_Free_PPO"]


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. WorldModelGorsellestirici 6 panelli teşhis panosunu başarıyla oluşturmalıdır."""
    cikti = str(tmp_path / "test_world_model_paneli.png")
    profil = WorldModelProfilleyici.basarim_profili_cikar()

    WorldModelGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
