"""
PyTest Birim Testleri - Day 241: OpenVLA Robotik Mimari Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.openvla_motoru import (
    OpenVLAActionTokenizer,
    OpenVLAModel,
    OpenVLAController,
)
from src.openvla_profilleyici import OpenVLAProfilleyici
from src.gorsellestirici import OpenVLAGorsellestirici


def test_action_tokenizer_bounds():
    """1. OpenVLAActionTokenizer sınır değerleri doğru belirteçlemelidir."""
    tokenizer = OpenVLAActionTokenizer(kova_sayisi=256, eylem_boyutu=3)
    actions = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    tokens = tokenizer.tokenize_action(actions)
    assert tokens[0] == 0
    assert tokens[1] in [127, 128]
    assert tokens[2] == 255


def test_action_tokenizer_reconstruction_error():
    """2. Ayrıklaştırma ve geri çözme hatası 0.02'nin altında olmalıdır."""
    tokenizer = OpenVLAActionTokenizer(kova_sayisi=256, eylem_boyutu=7)
    actions = np.array([0.15, -0.42, 0.88, -0.12, 0.05, 0.99, -0.99], dtype=np.float32)
    tokens = tokenizer.tokenize_action(actions)
    recovered = tokenizer.detokenize_action(tokens)
    error = np.max(np.abs(actions - recovered))
    assert error < 0.02


def test_openvla_model_initialization():
    """3. OpenVLAModel forward çıktısı [B, 7, 256] boyutunda olmalıdır."""
    model = OpenVLAModel(viz_dim=64, text_dim=64, gizli_boyut=128, eylem_sayisi=7, kova_sayisi=256)
    img = torch.randn(2, 64)
    txt = torch.randn(2, 64)
    out = model(img, txt)
    assert out.shape == (2, 7, 256)


def test_openvla_model_action_prediction():
    """4. predict_action metodu 7 elemanlı sürekli eylem vektörü üretmelidir."""
    model = OpenVLAModel(viz_dim=64, text_dim=64, gizli_boyut=128)
    img = torch.randn(1, 64)
    txt = torch.randn(1, 64)
    action = model.predict_action(img, txt)
    assert len(action) == 7
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_openvla_controller_trajectory_step():
    """5. OpenVLAController her adımda robotun uç nokta konumunu güncellemelidir."""
    model = OpenVLAModel(viz_dim=64, text_dim=64, gizli_boyut=128)
    controller = OpenVLAController(model)
    img = torch.randn(1, 64)
    txt = torch.randn(1, 64)
    delta, pos = controller.adim_yurut(img, txt)
    assert len(delta) == 7
    assert len(pos) == 7


def test_profiler_openvla_metrics():
    """6. Profilleyici OpenVLA başarısının %80 üstünde olduğunu doğrulamalıdır."""
    prof = OpenVLAProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["gorev_basari_orani"]["OpenVLA_Model"]
    assert skor > 80.0
    assert prof["karsilastirma"]["eylem_tahmin_hatasi_mse"]["OpenVLA_Model"] < 0.05


def test_action_tokenizer_clipping():
    """7. Aralık dışı eylemler [-1.0, 1.0] aralığına kırpılmalıdır."""
    tokenizer = OpenVLAActionTokenizer(kova_sayisi=256, eylem_boyutu=2)
    extreme = np.array([-5.0, 10.0], dtype=np.float32)
    tokens = tokenizer.tokenize_action(extreme)
    assert tokens[0] == 0
    assert tokens[1] == 255


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. OpenVLAGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_openvla_paneli.png")
    profil = OpenVLAProfilleyici.basarim_profili_cikar()

    OpenVLAGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
