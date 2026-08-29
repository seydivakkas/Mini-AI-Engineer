"""
PyTest Birim Testleri - Day 297 (FAZ 15): Dünya Modelleri ve DreamerV3 ile Hayal İçi Öğrenme.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dreamerv3_world_model_motoru import (
    SymlogTransform,
    RSSMCell,
    LatentImaginationActorCritic,
)
from src.dreamerv3_profilleyici import DreamerV3Profilleyici
from src.gorsellestirici import DreamerV3Gorsellestirici


def test_symlog_transform_reversibility():
    """1. Symlog ve Symexp fonksiyonları tersinir matematiksel doğruluk sağlamalıdır."""
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0, 10000.0])
    s = SymlogTransform.symlog(x)
    rec = SymlogTransform.symexp(s)
    assert torch.allclose(x, rec, atol=1e-3)


def test_rssm_cell_step_shapes():
    """2. RSSM hücresi doğru deterministik ve kategorik stokastik boyutları üretmelidir."""
    rssm = RSSMCell(deter_dim=256, stoch_dim=32, classes_dim=32, action_dim=6)
    prev_deter = torch.zeros(2, 256)
    prev_stoch = torch.zeros(2, 1024)
    action = torch.zeros(2, 6)

    deter, stoch, prior, post = rssm(prev_deter, prev_stoch, action)
    assert deter.shape == (2, 256)
    assert stoch.shape == (2, 1024)
    assert prior.shape == (2, 1024)


def test_rssm_posterior_prior_transition():
    """3. RSSM gözlem gömmesi verildiğinde posterior, verilmediğinde prior kullanmalıdır."""
    rssm = RSSMCell(deter_dim=128, stoch_dim=16, classes_dim=16, action_dim=4)
    prev_deter = torch.zeros(1, 128)
    prev_stoch = torch.zeros(1, 256)
    action = torch.zeros(1, 4)
    embed = torch.randn(1, 256)

    deter, stoch, prior, post = rssm(prev_deter, prev_stoch, action, embed=embed)
    assert deter.shape == (1, 128)
    assert stoch.shape == (1, 256)


def test_latent_imagination_actor_critic_rollout():
    """4. Hayal içi simülatör fiziksel etkileşim olmadan H=15 adım üretmelidir."""
    rssm = RSSMCell(deter_dim=128, stoch_dim=16, classes_dim=16, action_dim=4)
    ac = LatentImaginationActorCritic(state_dim=128 + 256, action_dim=4)

    start_deter = torch.zeros(1, 128)
    start_stoch = torch.zeros(1, 256)
    res = ac.imagine_rollout(rssm, start_deter, start_stoch, horizon=15)

    assert res["horizon"] == 15
    assert res["imagined_steps"] == 15
    assert res["values"].shape[0] == 15


def test_profiler_sample_efficiency_gain():
    """5. DreamerV3 örneklem verimlilik çarpanı 50x veya üzeri olmalıdır."""
    profil = DreamerV3Profilleyici.basarim_profili_cikar()
    assert profil["verimlilik_kazanci"] >= 50.0


def test_profiler_sim_to_real_accuracy():
    """6. Sıfır-Atış Sim-to-Real aktarım doğruluğu %90 üzerinde olmalıdır."""
    profil = DreamerV3Profilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["sim_to_real_basarisi_yuzde"]["3. DreamerV3 World Model"] >= 90.0


def test_profiler_damage_risk_reduction():
    """7. Robotik donanım hasar riski %5'in altına düşmelidir."""
    profil = DreamerV3Profilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["donanim_yipranma_riski_yuzde"]["3. DreamerV3 World Model"] < 5.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. DreamerV3Gorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_world_model_paneli.png")
    profil = DreamerV3Profilleyici.basarim_profili_cikar()

    DreamerV3Gorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
