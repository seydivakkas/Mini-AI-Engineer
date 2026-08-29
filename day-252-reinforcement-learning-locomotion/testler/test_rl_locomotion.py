"""
PyTest Birim Testleri - Day 252: Pekiştirmeli Öğrenme ile Robotik Yürüme (RL Locomotion - PPO).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl_locomotion_motoru import (
    RewardShaper,
    PPOActorCritic,
    LocomotionEnvironment,
)
from src.rl_locomotion_profilleyici import RLLocomotionProfilleyici
from src.gorsellestirici import RLLocomotionGorsellestirici


def test_reward_shaper_tracking_reward():
    """1. RewardShaper hedef hıza ulaşıldığında pozitif yüksek ödül dönmelidir."""
    odul = RewardShaper.compute_step_reward(
        lin_vel=np.array([1.0, 0.0, 0.0]),
        target_lin_vel=np.array([1.0, 0.0, 0.0]),
        ang_vel=0.0,
        target_ang_vel=0.0,
        joint_torques=np.zeros(12),
        joint_acc=np.zeros(12),
        base_z=0.35,
        target_z=0.35,
    )
    assert odul["toplam_odul"] > 2.0
    assert odul["r_lin_tracking"] == 1.0


def test_reward_shaper_cost_of_transport():
    """2. compute_cost_of_transport geçerli pozitif COT değeri üretmelidir."""
    torques = np.ones(12) * 5.0
    q_dot = np.ones(12) * 2.0
    cot = RewardShaper.compute_cost_of_transport(torques, q_dot, mass_kg=12.0, lin_vel_mag=1.0)
    assert cot > 0.0
    assert isinstance(cot, float)


def test_ppo_actor_critic_shapes():
    """3. PPOActorCritic aktör (B, 12) ve kritik (B, 1) tensör üretmelidir."""
    net = PPOActorCritic(obs_dim=48, act_dim=12)
    obs = torch.randn(4, 48)
    mean, std = net.forward_actor(obs)
    val = net.forward_critic(obs)

    assert mean.shape == (4, 12)
    assert std.shape == (12,)
    assert val.shape == (4, 1)


def test_ppo_actor_critic_get_action():
    """4. get_action tekil numpy gözlem için (12,) eylem vektörü dönmelidir."""
    net = PPOActorCritic(obs_dim=48, act_dim=12)
    obs_np = np.random.randn(48)
    act = net.get_action(obs_np)
    assert isinstance(act, np.ndarray)
    assert act.shape == (12,)


def test_locomotion_env_init():
    """5. LocomotionEnvironment 12 eklem ve başlangıç gövde konumunu doğru kurmalıdır."""
    env = LocomotionEnvironment()
    assert env.dof == 12
    assert len(env.q) == 12
    assert env.base_pos[2] == 0.35


def test_locomotion_env_step():
    """6. env.step torkları hesaplayıp gövdeyi ilerletmelidir."""
    env = LocomotionEnvironment()
    act = np.zeros(12)
    res = env.step(act)

    assert len(res["torques"]) == 12
    assert len(res["q"]) == 12
    assert res["base_pos"][0] >= 0.0


def test_rl_locomotion_profiler_output():
    """7. RLLocomotionProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = RLLocomotionProfilleyici.basarim_profili_cikar()
    assert "Curriculum_PPO_WBC" in profil["karsilastirma"]["arazi_gecis_basarisi_yuzde"]
    assert profil["karsilastirma"]["arazi_gecis_basarisi_yuzde"]["Curriculum_PPO_WBC"] == 98.8
    assert profil["karsilastirma"]["tasima_maliyeti_COT"]["Curriculum_PPO_WBC"] == 0.85


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. RLLocomotionGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_rl_locomotion_paneli.png")
    profil = RLLocomotionProfilleyici.basarim_profili_cikar()

    RLLocomotionGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
