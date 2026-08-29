"""
GÜN 172: Latent Diffusion Modelleri (LDM / Stable Diffusion) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.gurultu_zaman_cizelgesi import GurultuZamanCizelgesi
from src.denoising_unet import DenoisingUNet, SinuzoidalZamanGomusu
from src.latent_diffusion_motoru import LatentDiffusionMotoru
from src.gorsellestirici import LDMGorsellestirici


def test_linear_gurultu_zaman_cizelgesi():
    """Linear gürültü zaman çizelgesinin artan beta ve azalan alpha_bar ürettiğini test eder."""
    schedule = GurultuZamanCizelgesi(num_timesteps=100, schedule_type="linear")
    assert len(schedule.betas) == 100
    assert schedule.alphas_cumprod[0] > schedule.alphas_cumprod[-1]
    assert schedule.sqrt_alphas_cumprod.shape == (100,)


def test_cosine_gurultu_zaman_cizelgesi():
    """Cosine gürültü zaman çizelgesinin sınır değerlerini test eder."""
    schedule = GurultuZamanCizelgesi(num_timesteps=100, schedule_type="cosine")
    assert len(schedule.betas) == 100
    assert torch.all(schedule.betas > 0)
    assert torch.all(schedule.betas < 1.0)


def test_ileri_difuzyon_formulasyonu():
    """q(z_t | z_0) tek adımlı kapalı formülünün doğru tensör boyutu ürettiğini test eder."""
    schedule = GurultuZamanCizelgesi(num_timesteps=100)
    z_0 = torch.randn(2, 4, 16, 16)
    t = torch.tensor([10, 50])

    z_t, gurultu = schedule.ileri_difuzyon(z_0, t)

    assert z_t.shape == (2, 4, 16, 16)
    assert gurultu.shape == (2, 4, 16, 16)


def test_sinuzoidal_zaman_gomusu():
    """Sinüzoidal zaman gömme katmanının doğru boyutta embedding ürettiğini test eder."""
    emb_layer = SinuzoidalZamanGomusu(dim=64)
    t = torch.tensor([0, 100, 500])
    emb = emb_layer(t)

    assert emb.shape == (3, 64)


def test_denoising_unet_ileri_besleme():
    """Denoising UNet'in [B, 4, H, W] gizli uzay tensöründen gürültü kestirimi yaptığını test eder."""
    unet = DenoisingUNet(in_channels=4, out_channels=4, base_channels=32)
    z_t = torch.randn(2, 4, 16, 16)
    t = torch.tensor([10, 50])

    eps_pred = unet(z_t, t)

    assert eps_pred.shape == (2, 4, 16, 16)


def test_latent_diffusion_motoru_kayip_ve_gradyan():
    """LDM motorunun MSE gürültü kestirim kaybı hesaplayıp gradyan akıttığını test eder."""
    motor = LatentDiffusionMotoru(
        unet=DenoisingUNet(in_channels=4, out_channels=4, base_channels=16),
        schedule=GurultuZamanCizelgesi(num_timesteps=50),
    )
    z_0 = torch.randn(2, 4, 16, 16)
    kayip, z_t, eps_pred = motor.kayip_hesapla(z_0)

    assert kayip.item() >= 0.0
    kayip.backward()
    assert motor.unet.conv_in.weight.grad is not None


def test_difuzyon_ornek_rapor():
    """Örnek LDM raporunun doğru metrikler içerdiğini test eder."""
    rapor = LatentDiffusionMotoru.ornek_difuzyon_senaryolarini_getir()
    assert rapor["num_timesteps"] == 1000
    assert len(rapor["adimlar"]) == 5
    assert rapor["ortalama_gurultu_kestirim_mse"] < 0.05


def test_gorsellestirici_pano_uretme():
    """6 panelli LDM teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = LatentDiffusionMotoru.ornek_difuzyon_senaryolarini_getir()
    gorsellestirici = LDMGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_ldm_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
