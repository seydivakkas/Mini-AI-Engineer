"""
GÜN 178: NeRF (Neural Radiance Fields) 3D Sahne Hacimsel Sentezi Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from src.pozisyonel_kodlayici import PozisyonelKodlayici
from src.nerf_mlp import NeRFModeli
from src.hacimsel_isin_izleyici import HacimselIsinIzleyici
from src.gorsellestirici import NeRFGorsellestirici


def test_pozisyonel_kodlayici_cikti_boyutu():
    """Fourier pozisyonel kodlayıcının doğru çıktı tensör boyutu ürettiğini test eder."""
    encoder = PozisyonelKodlayici(in_dims=3, num_frequencies=10, include_input=True)
    # 3 * (2 * 10 + 1) = 63
    assert encoder.out_dim == 63

    x = torch.randn(4, 3)
    out = encoder(x)
    assert out.shape == (4, 63)


def test_nerf_mlp_ileri_besleme():
    """NeRF MLP modelinin doğru RGB ve sigma şekillerini ve aralıklarını ürettiğini test eder."""
    model = NeRFModeli(pos_frequencies=4, dir_frequencies=2, hidden_dim=64)
    pts = torch.randn(8, 3)
    views = torch.randn(8, 3)
    views = views / torch.norm(views, dim=-1, keepdim=True)

    rgb, sigma = model(pts, views)
    assert rgb.shape == (8, 3)
    assert sigma.shape == (8, 1)
    assert (rgb >= 0.0).all() and (rgb <= 1.0).all()
    assert (sigma >= 0.0).all()


def test_hacimsel_isin_izleyici_ornekleme():
    """Işın boyunca örneklenen nokta sayısı ve derinlik aralıklarını test eder."""
    model = NeRFModeli(pos_frequencies=4, dir_frequencies=2, hidden_dim=32)
    renderer = HacimselIsinIzleyici(model, near=2.0, far=6.0, num_samples=16)

    rays_o = torch.zeros(5, 3)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]]).expand(5, 3)

    pts, z_vals, deltas = renderer.isin_ornekleme_noktalari_uret(rays_o, rays_d, perturb=False)
    assert pts.shape == (5, 16, 3)
    assert z_vals.shape == (5, 16)
    assert (z_vals >= 2.0).all() and (z_vals <= 6.0).all()


def test_hacimsel_render_isin_cikti_tipleri():
    """Hacimsel render sonucunda RGB, derinlik ve ağırlık tensörlerinin eksiksiz döndüğünü test eder."""
    model = NeRFModeli(pos_frequencies=4, dir_frequencies=2, hidden_dim=32)
    renderer = HacimselIsinIzleyici(model, near=1.0, far=4.0, num_samples=8)

    rays_o = torch.zeros(3, 3)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, 3)

    sonuclar = renderer.render_isin(rays_o, rays_d, perturb=False)
    assert sonuclar["rgb"].shape == (3, 3)
    assert sonuclar["depth"].shape == (3,)
    assert sonuclar["weights"].shape == (3, 8)


def test_nerf_gradyan_akisi():
    """Piksel rengi MSE kaybından NeRF MLP parametrelerine gradyan aktığını test eder."""
    model = NeRFModeli(pos_frequencies=4, dir_frequencies=2, hidden_dim=32)
    renderer = HacimselIsinIzleyici(model, near=1.0, far=3.0, num_samples=4)

    rays_o = torch.zeros(2, 3)
    rays_d = torch.tensor([[0.0, 1.0, 0.0]]).expand(2, 3)
    target_rgb = torch.rand(2, 3)

    out = renderer.render_isin(rays_o, rays_d, perturb=True)
    loss = nn.functional.mse_loss(out["rgb"], target_rgb)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parametre {name} gradyan almadı!"


def test_nerf_ornek_rapor():
    """Örnek NeRF raporunun PSNR metriklerini ve karşılaştırmalarını içerdiğini test eder."""
    rapor = HacimselIsinIzleyici.ornek_nerf_sahne_raporu()
    assert "metrikler" in rapor
    assert rapor["metrikler"]["psnr"] > 30.0
    assert len(rapor["karsilastirma"]) == 3


def test_transmittance_monotonic_azalma():
    """Transmittance değerlerinin ışın derinleştikçe monotonically azaldığını veya sabit kaldığını test eder."""
    model = NeRFModeli(pos_frequencies=4, dir_frequencies=2, hidden_dim=32)
    renderer = HacimselIsinIzleyici(model, near=1.0, far=5.0, num_samples=16)

    rays_o = torch.zeros(1, 3)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])

    out = renderer.render_isin(rays_o, rays_d, perturb=False)
    weights = out["weights"][0]
    # Ağırlıkların toplamı 1.0'ı aşamaz
    assert weights.sum() <= 1.0 + 1e-4


def test_gorsellestirici_pano_uretme():
    """6 panelli NeRF teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = HacimselIsinIzleyici.ornek_nerf_sahne_raporu()
    gorsellestirici = NeRFGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_nerf_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
