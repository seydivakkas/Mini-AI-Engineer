"""
PyTest Birim Testleri - Day 216: Reward Hacking ve Goodhart Yasası Önleme Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reward_hacking_motoru import (
    AdaptiveKLController,
    RewardSquasher,
    EnsembleRewardModel,
    RewardHackingDetector,
    RobustRLTrainer,
)
from src.reward_hacking_profilleyici import RewardHackingProfilleyici
from src.gorsellestirici import RewardHackingGorsellestirici


def test_adaptive_kl_controller_increase():
    """1. Adaptif KL Denetleyicisi yüksek sapmada cezayı (beta) artırmalıdır."""
    ctrl = AdaptiveKLController(kl_hedef=0.05, beta_baslangic=0.10)
    yeni_beta = ctrl.guncelle(olculen_kl=0.15)
    assert yeni_beta > 0.10


def test_adaptive_kl_controller_decrease():
    """2. Adaptif KL Denetleyicisi düşük sapmada cezayı (beta) gevşetmelidir."""
    ctrl = AdaptiveKLController(kl_hedef=0.05, beta_baslangic=0.10)
    yeni_beta = ctrl.guncelle(olculen_kl=0.01)
    assert yeni_beta < 0.10


def test_reward_squasher_tanh():
    """3. Tanh squashing çok büyük puanları maks değere yaklaştırmalıdır."""
    r = RewardSquasher.tanh_kirp(100.0, maks_odul=5.0)
    assert r <= 5.0
    assert r > 4.90


def test_reward_squasher_clip():
    """4. Sert kırpma aralık dışı değerleri sınırlamalıdır."""
    assert RewardSquasher.sert_kirp(15.0, -5.0, 5.0) == 5.0
    assert RewardSquasher.sert_kirp(-15.0, -5.0, 5.0) == -5.0


def test_ensemble_reward_model_lcb():
    """5. Ensemble LCB uyuşmazlık durumunda ortalamadan düşük değer (LCB) vermelidir."""
    ens = EnsembleRewardModel.degerlendir([10.0, 2.0], lambda_lcb=1.5)
    assert ens["lcb_odul"] < ens["ortalama_odul"]


def test_reward_hacking_detector_sycophancy():
    """6. RewardHackingDetector dalkavukluk ifadelerini yakalamalıdır."""
    metin = "Siz mükemmel bir uzmansınız, kesinlikle haklısınız efendim."
    analiz = RewardHackingDetector.denetle(metin, ham_odul=5.0, perplexity=15.0)
    assert analiz["dalkavukluk_var_mi"] is True
    assert analiz["hacking_suphesi"] is True


def test_robust_rl_trainer_step():
    """7. RobustRLTrainer tam güvenli adım raporu üretmelidir."""
    ctrl = AdaptiveKLController()
    adim = RobustRLTrainer.guvenli_odul_adimi(
        model_yaniti="Güvenli cevap",
        topluluk_puanlari=[3.0, 3.2, 3.1],
        olculen_kl=0.04,
        kl_controller=ctrl,
        perplexity=12.0,
    )
    assert "nihai_saglam_odul" in adim
    assert "hacking_raporu" in adim


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. RewardHackingGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_reward_hacking_paneli.png")
    profil = RewardHackingProfilleyici.basarim_profili_cikar()

    RewardHackingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
