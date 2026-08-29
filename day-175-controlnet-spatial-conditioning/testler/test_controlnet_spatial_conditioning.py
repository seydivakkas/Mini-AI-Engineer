"""
GÜN 175: ControlNet: Mekansal Koşullu Görüntü Üretimi Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.zero_convolution import ZeroConv2d
from src.controlnet_modeli import ControlNetModeli
from src.mekansal_kontrol_degerlendirici import MekansalKontrolDegerlendirici
from src.gorsellestirici import ControlNetGorsellestirici


def test_zero_convolution_baslangic_sifir():
    """Sıfır-konvolüsyonun başlangıçta sıfır tensörü ürettiğini test eder."""
    zero_conv = ZeroConv2d(in_channels=32, out_channels=64)
    x = torch.randn(2, 32, 16, 16)
    out = zero_conv(x)

    assert out.shape == (2, 64, 16, 16)
    assert torch.allclose(out, torch.zeros_like(out))


def test_zero_convolution_gradyan_akisi():
    """Sıfır-konvolüsyonun sıfırla başlamasına rağmen gradyan aldığını test eder."""
    zero_conv = ZeroConv2d(in_channels=16, out_channels=16)
    x = torch.randn(1, 16, 8, 8)
    out = zero_conv(x)
    loss = (out + x).sum()
    loss.backward()

    assert zero_conv.conv.weight.grad is not None


def test_controlnet_modeli_ileri_besleme():
    """ControlNet modelinin doğru sayıda ve boyutta rezidüel tensör ürettiğini test eder."""
    model = ControlNetModeli(in_channels=4, hint_channels=3, base_channels=32)
    z_t = torch.randn(2, 4, 16, 16)
    hint = torch.randn(2, 3, 16, 16)

    residuals = model(z_t, hint, control_weight=1.0)

    assert len(residuals) == 3
    assert residuals[0].shape == (2, 64, 8, 8)
    assert residuals[1].shape == (2, 128, 4, 4)
    assert residuals[2].shape == (2, 128, 4, 4)


def test_controlnet_control_weight_sifir():
    """control_weight=0.0 olduğunda tüm rezidüellerin sıfır olduğunu test eder."""
    model = ControlNetModeli(in_channels=4, hint_channels=3, base_channels=16)
    z_t = torch.randn(1, 4, 8, 8)
    hint = torch.randn(1, 3, 8, 8)

    residuals = model(z_t, hint, control_weight=0.0)

    for res in residuals:
        assert torch.allclose(res, torch.zeros_like(res))


def test_controlnet_gradyan_akisi():
    """ControlNet'in geriye yayılımda gradyanları sorunsuz aldığını test eder."""
    model = ControlNetModeli(in_channels=4, hint_channels=3, base_channels=16)
    z_t = torch.randn(1, 4, 8, 8)
    hint = torch.randn(1, 3, 8, 8)

    residuals = model(z_t, hint)
    loss = sum(r.sum() for r in residuals)
    loss.backward()

    assert model.down1.weight.grad is not None


def test_mekansal_kontrol_raporu():
    """Örnek raporun Canny, Depth ve OpenPose koşullarını ve %96 uyum içerdiğini test eder."""
    rapor = MekansalKontrolDegerlendirici.ornek_kontrol_raporunu_getir()
    assert len(rapor["kosul_tipleri"]) == 3
    assert rapor["ortalama_mekansal_uyum"] > 0.90


def test_controlnet_farkli_batch_boyutlari():
    """Farklı batch boyutlarında modelin sorunsuz çalıştığını test eder."""
    model = ControlNetModeli(in_channels=4, hint_channels=3, base_channels=16)
    for b in [1, 3]:
        z_t = torch.randn(b, 4, 8, 8)
        hint = torch.randn(b, 3, 8, 8)
        residuals = model(z_t, hint)
        assert residuals[0].shape[0] == b


def test_gorsellestirici_pano_uretme():
    """6 panelli ControlNet teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = MekansalKontrolDegerlendirici.ornek_kontrol_raporunu_getir()
    gorsellestirici = ControlNetGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_controlnet_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
