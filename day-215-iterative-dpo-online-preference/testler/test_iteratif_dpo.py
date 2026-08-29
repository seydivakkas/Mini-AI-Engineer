"""
PyTest Birim Testleri - Day 215: İteratif ve Çevrimiçi DPO Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.iteratif_dpo_motoru import (
    OnlinePreferenceBuffer,
    OnlineRolloutSampler,
    ReferencePolicyUpdater,
    IterativeDPOTrainer,
)
from src.iteratif_dpo_profilleyici import IterativeDPOProfilleyici
from src.gorsellestirici import IterativeDPOGorsellestirici


def test_preference_buffer_add():
    """1. OnlinePreferenceBuffer verileri kaydetmeli ve kapasiteyi aşmamalıdır."""
    buf = OnlinePreferenceBuffer(kapasite=3)
    buf.ekle("p1", "c1", "r1", 1)
    buf.ekle("p2", "c2", "r2", 1)
    buf.ekle("p3", "c3", "r3", 2)
    buf.ekle("p4", "c4", "r4", 2)
    assert len(buf.havuz) == 3
    assert buf.havuz[-1]["prompt"] == "p4"


def test_preference_buffer_sample():
    """2. OnlinePreferenceBuffer rastgele mini-batch örneklemelidir."""
    buf = OnlinePreferenceBuffer(kapasite=10)
    buf.ekle("p1", "c1", "r1", 1)
    buf.ekle("p2", "c2", "r2", 1)
    sample = buf.orneklem_al(batch_boyutu=2)
    assert len(sample) == 2


def test_online_rollout_sampler():
    """3. OnlineRolloutSampler chosen ve rejected metinleri üretmelidir."""
    c, r = OnlineRolloutSampler.cift_yanit_uret_ve_etiketle("Test", tur_no=2)
    assert "Derin düşünce" in c
    assert "Basit Sonuç" in r


def test_reference_policy_updater():
    """4. ReferencePolicyUpdater derin kopya referansı oluşturmalıdır."""
    agirliklar = {"w1": 1.5, "w2": 2.5}
    ref = ReferencePolicyUpdater.referansi_guncelle(agirliklar)
    agirliklar["w1"] = 9.9
    assert ref["w1"] == 1.5


def test_online_dpo_loss():
    """5. Online DPO kaybı geçerli skaler tensör üretmelidir."""
    loss, delta = IterativeDPOTrainer.online_dpo_kaybi(
        logp_pi_w=torch.tensor(-1.0),
        logp_ref_w=torch.tensor(-1.5),
        logp_pi_l=torch.tensor(-2.5),
        logp_ref_l=torch.tensor(-1.8),
        beta=0.1,
    )
    assert loss.item() > 0.0
    assert delta > 0.0


def test_implicit_reward_margin_positive():
    """6. Seçilen yanıtın log-olasılığı yüksek olduğunda örtük marjin pozitif olmalıdır."""
    _, delta = IterativeDPOTrainer.online_dpo_kaybi(
        logp_pi_w=torch.tensor(-0.5),
        logp_ref_w=torch.tensor(-1.5),
        logp_pi_l=torch.tensor(-3.0),
        logp_ref_l=torch.tensor(-1.5),
        beta=0.1,
    )
    assert delta > 0.0


def test_iterative_round_execution():
    """7. IterativeDPOTrainer iterasyon turunu eksiksiz yürütmelidir."""
    buf = OnlinePreferenceBuffer()
    sonuc = IterativeDPOTrainer.iteratif_tur_yurut("Test", tur_no=1, buffer=buf)
    assert "kayip" in sonuc
    assert "ortuk_odul_marjini" in sonuc
    assert sonuc["buffer_boyutu"] == 1


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. IterativeDPOGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_iteratif_dpo_paneli.png")
    profil = IterativeDPOProfilleyici.basarim_profili_cikar()

    IterativeDPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
