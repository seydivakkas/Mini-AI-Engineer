"""
GÜN 177: Diffusion Transformers (DiT - Sora & Flux Omurgası) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from src.adaln_zero import AdaLNZero, modulate
from src.dit_blok import DiTBlock
from src.dit_modeli import DiffusionTransformer
from src.gorsellestirici import DiTGorsellestirici


def test_adaln_zero_baslangic_sifirlama():
    """adaLN-Zero son katmanının ağırlıklarının sıfır başlatıldığını test eder."""
    adaln = AdaLNZero(hidden_size=64, cond_size=32)
    c = torch.randn(2, 32)
    gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln(c)

    assert torch.allclose(alpha1, torch.zeros_like(alpha1))
    assert torch.allclose(alpha2, torch.zeros_like(alpha2))


def test_modulate_islevi():
    """modulate fonksiyonunun doğru formülle çalıştığını test eder: x * (1 + scale) + shift."""
    x = torch.ones(2, 4, 8)
    shift = torch.ones(2, 8) * 2.0
    scale = torch.ones(2, 8) * 3.0

    out = modulate(x, shift, scale)
    # x * (1 + 3) + 2 = 1*4 + 2 = 6
    assert torch.allclose(out, torch.full_like(out, 6.0))


def test_dit_block_ileri_besleme_ve_sekil():
    """DiTBlock'un girdi tensör boyutunu koruduğunu test eder."""
    block = DiTBlock(hidden_size=64, num_heads=4, cond_size=32)
    x = torch.randn(2, 16, 64)
    c = torch.randn(2, 32)

    out = block(x, c)
    assert out.shape == (2, 16, 64)


def test_dit_modeli_patchify_unpatchify_dongusu():
    """DiT modelinin [B, C, H, W] girdisini alıp aynı uzaysal boyutta çıktı ürettiğini test eder."""
    model = DiffusionTransformer(
        input_size=16,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        depth=2,
        num_heads=2,
        cond_size=64,
    )
    x = torch.randn(2, 4, 16, 16)
    t = torch.tensor([100, 500])

    out = model(x, t)
    assert out.shape == (2, 4, 16, 16)


def test_dit_gradyan_akisi():
    """Gürültü kestirimi MSE kaybından tüm DiT parametrelerine gradyan aktığını test eder."""
    model = DiffusionTransformer(
        input_size=16,
        patch_size=4,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=2,
        cond_size=32,
    )
    x = torch.randn(1, 4, 16, 16)
    t = torch.tensor([250])
    target_noise = torch.randn(1, 4, 16, 16)

    pred_noise = model(x, t)
    loss = nn.functional.mse_loss(pred_noise, target_noise)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parametre {name} gradyan almadı!"


def test_dit_farkli_patch_boyutlari():
    """Farklı yama boyutları (p=2 ve p=4) için modelin doğru sayıda yama ürettiğini test eder."""
    model_p2 = DiffusionTransformer(input_size=16, patch_size=2, in_channels=4, hidden_size=32)
    model_p4 = DiffusionTransformer(input_size=16, patch_size=4, in_channels=4, hidden_size=32)

    assert model_p2.num_patches == (16 // 2) ** 2 == 64
    assert model_p4.num_patches == (16 // 4) ** 2 == 16


def test_dit_ornek_rapor():
    """Örnek DiT raporunun 4 model varyantını ve ölçeklenme yasasını içerdiğini test eder."""
    rapor = DiffusionTransformer.ornek_dit_karsilastirma_raporu()
    assert len(rapor["model_varyantlari"]) == 4
    assert len(rapor["patch_boyut_analizi"]) == 3


def test_gorsellestirici_pano_uretme():
    """6 panelli DiT teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = DiffusionTransformer.ornek_dit_karsilastirma_raporu()
    gorsellestirici = DiTGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_dit_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
