"""
PyTest Birim Testleri - Day 290 (FAZ 15): Mekanistik Yorumlanabilirlik ve Seyrek Otokodlayıcılar (SAE).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sparse_autoencoder_motoru import SparseAutoencoder, ActivationSteeringEngine
from src.sparse_autoencoder_profilleyici import SAEProfilleyici
from src.gorsellestirici import SAEGorsellestirici


def test_sae_module_initialization():
    """1. SAE modülü doğru giriş ve aşırı tamamlanmış sözlük boyutlarıyla başlatılmalıdır."""
    sae = SparseAutoencoder(d_in=32, d_sae=128)
    assert sae.W_enc.shape == (32, 128)
    assert sae.W_dec.shape == (128, 32)


def test_sae_decoder_normalization():
    """2. Kod çözücü (Decoder) ağırlık satırları birim L2 normuna normalize edilmelidir."""
    sae = SparseAutoencoder(d_in=32, d_sae=128)
    norms = torch.norm(sae.W_dec, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_sae_forward_pass_loss():
    """3. İleri geçiş geçerli yeniden inşa ve L1 seyreklik kaybı üretmelidir."""
    sae = SparseAutoencoder(d_in=16, d_sae=64)
    x = torch.randn(8, 16)
    x_hat, f, l2_loss, total_loss = sae(x)
    assert x_hat.shape == x.shape
    assert f.shape == (8, 64)
    assert l2_loss.item() >= 0.0
    assert total_loss.item() >= 0.0


def test_sae_sparsity_encoding():
    """4. ReLU aktivasyonu nedeniyle tüm gizli öznitelikler negatif olmamalıdır (f >= 0)."""
    sae = SparseAutoencoder(d_in=16, d_sae=64)
    x = torch.randn(4, 16)
    f = sae.encode(x)
    assert (f >= 0).all()


def test_activation_steering_modifies_vector():
    """5. Aktivasyon yönlendirme residual akımı hedef öznitelik yönünde değiştirmelidir."""
    sae = SparseAutoencoder(d_in=16, d_sae=64)
    x = torch.randn(1, 16)
    x_steered = ActivationSteeringEngine.steer_activation(x, sae, feature_idx=5, alpha=2.0)
    assert not torch.allclose(x, x_steered)
    assert x_steered.shape == x.shape


def test_profiler_monosemanticity_superiority():
    """6. SAE tek anlamlılık saflığı (%97.8) ham nöronları (%24.5) belirgin şekilde aşmalıdır."""
    profil = SAEProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["tek_anlamlilik_safligi_yuzde"]["3. Sparse Autoencoder"] > 95.0
    assert kars["tek_anlamlilik_safligi_yuzde"]["1. Ham Nöronlar"] < 30.0


def test_profiler_l0_sparsity_reduction():
    """7. SAE token başına L0 aktif öznitelik sayısı aşırı seyrek olmalıdır (<= 15.0)."""
    profil = SAEProfilleyici.basarim_profili_cikar()
    assert profil["karsilastirma"]["l0_aktiflik_sayisi"]["3. Sparse Autoencoder"] <= 15.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. SAEGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_sae_paneli.png")
    profil = SAEProfilleyici.basarim_profili_cikar()

    SAEGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
