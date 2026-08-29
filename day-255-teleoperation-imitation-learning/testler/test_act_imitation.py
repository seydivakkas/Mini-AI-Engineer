"""
PyTest Birim Testleri - Day 255: Teleoperasyon ve Taklit Öğrenmesi (ACT & Behavior Cloning).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.act_imitation_motoru import (
    TeleoperationDataBuffer,
    ACTCVAEModel,
    TemporalEnsembler,
)
from src.act_imitation_profilleyici import ACTImitationProfilleyici
from src.gorsellestirici import ACTImitationGorsellestirici


def test_teleoperation_data_buffer_add():
    """1. TeleoperationDataBuffer yörüngeleri K boyutlu eylem parçalarına bölmelidir."""
    buffer = TeleoperationDataBuffer(chunk_size=10)
    states = np.zeros((25, 14))
    actions = np.zeros((25, 7))
    buffer.add_demonstration(states, actions)
    assert len(buffer.samples) == 25
    assert buffer.samples[0]["action_chunk"].shape == (10, 7)


def test_teleoperation_data_buffer_batch():
    """2. get_batch uygun tensör boyutlarında mini-batch üretmelidir."""
    buffer = TeleoperationDataBuffer(chunk_size=10)
    buffer.add_demonstration(np.zeros((15, 14)), np.zeros((15, 7)))
    s_batch, a_batch = buffer.get_batch(batch_size=4)
    assert s_batch.shape == (4, 14)
    assert a_batch.shape == (4, 10, 7)


def test_act_cvae_model_forward_train():
    """3. ACTCVAEModel eğitim modunda (B, K, 7) tahmin ve CVAE parametreleri dönmelidir."""
    model = ACTCVAEModel(state_dim=14, action_dim=7, chunk_size=10, latent_dim=16)
    s = torch.randn(4, 14)
    a = torch.randn(4, 10, 7)
    pred_chunk, mu, log_std = model(s, a)
    assert pred_chunk.shape == (4, 10, 7)
    assert mu.shape == (4, 16)
    assert log_std.shape == (4, 16)


def test_act_cvae_model_forward_eval():
    """4. ACTCVAEModel çıkarım modunda z=0 kullanarak geçerli eylem yığını üretmelidir."""
    model = ACTCVAEModel(state_dim=14, action_dim=7, chunk_size=10, latent_dim=16)
    s = torch.randn(2, 14)
    pred_chunk, _, _ = model(s)
    assert pred_chunk.shape == (2, 10, 7)


def test_act_cvae_model_loss():
    """5. compute_loss L1 ve KL kayıplarını hesaplamalıdır."""
    model = ACTCVAEModel(state_dim=14, action_dim=7, chunk_size=10, latent_dim=16)
    pred = torch.randn(4, 10, 7)
    target = torch.randn(4, 10, 7)
    mu = torch.zeros(4, 16)
    log_std = torch.zeros(4, 16)
    losses = model.compute_loss(pred, target, mu, log_std)
    assert "total_loss" in losses
    assert losses["total_loss"].item() > 0.0


def test_temporal_ensembler_weighted_action():
    """6. TemporalEnsembler çakışan tahminlerden tekil (7,) eylem üretmelidir."""
    ensembler = TemporalEnsembler(chunk_size=10, m_decay=0.05)
    for _ in range(3):
        pred_k = np.random.randn(10, 7)
        ensembler.add_prediction(pred_k)
    act = ensembler.get_ensembled_action()
    assert act.shape == (7,)


def test_act_imitation_profiler_output():
    """7. ACTImitationProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = ACTImitationProfilleyici.basarim_profili_cikar()
    assert "ACT_Temporal_Ensemble" in profil["karsilastirma"]["cok_asamali_gorev_basarisi_yuzde"]
    assert profil["karsilastirma"]["cok_asamali_gorev_basarisi_yuzde"]["ACT_Temporal_Ensemble"] == 97.8


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. ACTImitationGorsellestirici 6 panelli teşhis panosunu üretmelidir."""
    cikti = str(tmp_path / "test_act_teleoperation_paneli.png")
    profil = ACTImitationProfilleyici.basarim_profili_cikar()

    ACTImitationGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
