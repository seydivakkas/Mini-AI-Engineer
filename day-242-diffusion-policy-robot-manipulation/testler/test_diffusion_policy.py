"""
PyTest Birim Testleri - Day 242: Diffusion Policy Robotik Manipülasyon Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.diffusion_policy_motoru import (
    DiffusionPolicyScheduler,
    DiffusionUNet1D,
    DiffusionPolicyController,
)
from src.diffusion_policy_profilleyici import DiffusionPolicyProfilleyici
from src.gorsellestirici import DiffusionPolicyGorsellestirici


def test_diffusion_scheduler_initialization():
    """1. DiffusionPolicyScheduler katsayı dizilerini doğru oluşturmalıdır."""
    scheduler = DiffusionPolicyScheduler(adim_sayisi=10)
    assert len(scheduler.betalar) == 10
    assert len(scheduler.kumulatif_alfalar) == 10
    assert scheduler.kumulatif_alfalar[0] > scheduler.kumulatif_alfalar[-1]


def test_diffusion_scheduler_noise_addition():
    """2. gurultu_ekle ileri difüzyon işlemini doğru boyutla tamamlamalıdır."""
    scheduler = DiffusionPolicyScheduler(adim_sayisi=10)
    x0 = torch.zeros(2, 8, 7)
    noise = torch.randn(2, 8, 7)
    t = torch.tensor([5, 5])
    noisy = scheduler.gurultu_ekle(x0, noise, t)
    assert noisy.shape == (2, 8, 7)


def test_diffusion_unet1d_forward_shape():
    """3. DiffusionUNet1D çıktısı [B, Ta, Da] formatında olmalıdır."""
    model = DiffusionUNet1D(eylem_boyutu=7, eylem_ufku=8, kosul_boyutu=32, gizli_boyut=64)
    x = torch.randn(2, 8, 7)
    t = torch.tensor([3, 3])
    c = torch.randn(2, 32)
    out = model(x, t, c)
    assert out.shape == (2, 8, 7)


def test_diffusion_policy_sample_action_chunk():
    """4. Controller eylem_bloku_uret ile [Ta, Da] boyutunda blok üretmelidir."""
    model = DiffusionUNet1D(eylem_boyutu=7, eylem_ufku=8, kosul_boyutu=32, gizli_boyut=64)
    scheduler = DiffusionPolicyScheduler(adim_sayisi=5)
    controller = DiffusionPolicyController(model, scheduler)
    c = torch.randn(1, 32)
    chunk = controller.eylem_bloku_uret(c)
    assert chunk.shape == (8, 7)


def test_diffusion_policy_receding_horizon():
    """5. kayan_ufuk_icra_et belirtilen Te adım kadar eylemi döndürmelidir."""
    model = DiffusionUNet1D(eylem_boyutu=7, eylem_ufku=8, kosul_boyutu=32, gizli_boyut=64)
    scheduler = DiffusionPolicyScheduler(adim_sayisi=5)
    controller = DiffusionPolicyController(model, scheduler)
    c = torch.randn(1, 32)
    steps = controller.kayan_ufuk_icra_et(c, icra_adimi=3)
    assert steps.shape == (3, 7)


def test_profiler_diffusion_metrics():
    """6. Profilleyici Diffusion Policy başarısının %85 üstünde olduğunu doğrulamalıdır."""
    prof = DiffusionPolicyProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["gorev_basari_orani"]["Diffusion_Policy"]
    assert skor > 85.0
    assert prof["karsilastirma"]["yorunge_sarsinti_indeksi_jerk"]["Diffusion_Policy"] < 10.0


def test_denoise_step_reduction():
    """7. gurultuden_arindir_adimi ters adımı tensör boyutunu korumalıdır."""
    scheduler = DiffusionPolicyScheduler(adim_sayisi=10)
    xk = torch.randn(1, 8, 7)
    noise_pred = torch.randn(1, 8, 7)
    prev = scheduler.gurultuden_arindir_adimi(noise_pred, 5, xk)
    assert prev.shape == (1, 8, 7)


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. DiffusionPolicyGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_diffusion_paneli.png")
    profil = DiffusionPolicyProfilleyici.basarim_profili_cikar()

    DiffusionPolicyGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
