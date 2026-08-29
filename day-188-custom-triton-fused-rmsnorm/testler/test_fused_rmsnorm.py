"""
PyTest Birim Testleri - Day 188: Özel Triton Fused RMSNorm & Residual.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fused_rmsnorm_motoru import (
    PyTorchUnfusedRMSNormResidual,
    FusedRMSNormResidual,
)
from src.profilleyici import RMSNormBellekProfilleyici
from src.gorsellestirici import RMSNormGorsellestirici


@pytest.fixture
def test_verileri():
    """Test tensörleri (B=2, S=32, D=256)."""
    torch.manual_seed(42)
    b, s, d = 2, 32, 256
    x = torch.randn(b, s, d, requires_grad=True)
    res = torch.randn(b, s, d, requires_grad=True)
    return x, res, d


def test_fused_rmsnorm_ileri_gecis_eslesme(test_verileri):
    """1. Fused RMSNorm ileri geçiş çıktısı standart PyTorch ile sayısal olarak eşleşmelidir."""
    x, res, d = test_verileri
    fused_norm = FusedRMSNormResidual(hidden_dim=d)
    torch_norm = PyTorchUnfusedRMSNormResidual(hidden_dim=d)
    with torch.no_grad():
        torch_norm.weight.copy_(fused_norm.weight)

    out_fused, _ = fused_norm(x, res)
    out_torch, _ = torch_norm(x, res)
    assert torch.allclose(out_fused, out_torch, atol=1e-6)


def test_fused_rmsnorm_residual_donusu(test_verileri):
    """2. Fused RMSNorm residual tensörünü (X + Residual) doğru döndürmelidir."""
    x, res, d = test_verileri
    fused_norm = FusedRMSNormResidual(hidden_dim=d)
    _, x_res_fused = fused_norm(x, res)
    assert torch.allclose(x_res_fused, x + res, atol=1e-6)


def test_fused_rmsnorm_residualsiz_calisma():
    """3. Residual None verildiğinde sadece normalizasyon çalışmalıdır."""
    x = torch.randn(2, 16, 128)
    fused_norm = FusedRMSNormResidual(hidden_dim=128)
    out, x_res = fused_norm(x, residual=None)
    assert out.shape == x.shape
    assert torch.allclose(x_res, x, atol=1e-6)


def test_fused_autograd_dx_ve_dres_gradyan(test_verileri):
    """4. Geri geçişte dX ve dResidual gradyanları standart PyTorch autograd ile eşleşmelidir."""
    x, res, d = test_verileri
    fused_norm = FusedRMSNormResidual(hidden_dim=d)

    out_fused, _ = fused_norm(x, res)
    loss_f = out_fused.sum()
    loss_f.backward()

    x_ref = x.detach().clone().requires_grad_(True)
    res_ref = res.detach().clone().requires_grad_(True)
    torch_norm = PyTorchUnfusedRMSNormResidual(hidden_dim=d)
    with torch.no_grad():
        torch_norm.weight.copy_(fused_norm.weight)

    out_torch, _ = torch_norm(x_ref, res_ref)
    loss_t = out_torch.sum()
    loss_t.backward()

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5)
    assert torch.allclose(res.grad, res_ref.grad, atol=1e-5)


def test_fused_autograd_dweight_gradyan(test_verileri):
    """5. Geri geçişte dWeight gradyanı standart PyTorch autograd ile eşleşmelidir."""
    x, res, d = test_verileri
    fused_norm = FusedRMSNormResidual(hidden_dim=d)

    out_fused, _ = fused_norm(x, res)
    out_fused.sum().backward()

    x_ref = x.detach().clone().requires_grad_(True)
    res_ref = res.detach().clone().requires_grad_(True)
    torch_norm = PyTorchUnfusedRMSNormResidual(hidden_dim=d)
    with torch.no_grad():
        torch_norm.weight.copy_(fused_norm.weight)

    out_torch, _ = torch_norm(x_ref, res_ref)
    out_torch.sum().backward()

    assert torch.allclose(fused_norm.weight.grad, torch_norm.weight.grad, atol=1e-5)


def test_bellek_profilleyici_hbm_gecis_orani():
    """6. Bellek profilleyici HBM geçiş sayısının 13'ten 5'e düştüğünü (%61.5 kazanç) doğrulamalıdır."""
    analiz = RMSNormBellekProfilleyici.bellek_ve_gecis_analizi(batch_size=2, seq_len=1024, hidden_dim=4096)
    assert analiz["tasarruf_orani"] == pytest.approx(2.6, abs=1e-2)
    assert analiz["triton_ara_bellek_mb"] == 0.0


def test_model_olcegi_tasarruf_raporu():
    """7. Model ölçeği raporu Llama-3-8B, Gemma-2-27B ve Llama-3-70B'yi içermelidir."""
    rapor = RMSNormBellekProfilleyici.model_olcegi_tasarruf_raporu()
    assert len(rapor) == 3
    adlar = [r["model_adi"] for r in rapor]
    assert "Llama-3-70B" in adlar
    assert "Llama-3-8B" in adlar


def test_gorsellestirme_cikti_dosyasi(tmp_path):
    """8. RMSNormGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_rmsnorm_paneli.png")
    katman_analizi = RMSNormBellekProfilleyici.bellek_ve_gecis_analizi(batch_size=2, seq_len=512, hidden_dim=2048)
    model_raporu = RMSNormBellekProfilleyici.model_olcegi_tasarruf_raporu()

    RMSNormGorsellestirici.teshis_paneli_olustur(
        katman_analizi=katman_analizi,
        model_raporu=model_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
