"""
PyTest Birim Testleri - Day 189: Özel Triton Fused SwiGLU.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fused_swiglu_motoru import (
    PyTorchUnfusedSwiGLU,
    FusedSwiGLU,
    SwiGLUMLP,
)
from src.swiglu_profilleyici import SwiGLUBellekProfilleyici
from src.gorsellestirici import SwiGLUGorsellestirici


@pytest.fixture
def test_tensörleri():
    """Test tensörleri (B=2, S=16, D_ffn=128)."""
    torch.manual_seed(42)
    b, s, d_ffn = 2, 16, 128
    gate = torch.randn(b, s, d_ffn, requires_grad=True)
    up = torch.randn(b, s, d_ffn, requires_grad=True)
    return gate, up


def test_fused_swiglu_ileri_gecis_eslesme(test_tensörleri):
    """1. Fused SwiGLU ileri geçişi standart PyTorch SiLU(Gate)*Up ile eşleşmelidir."""
    gate, up = test_tensörleri
    fused_swiglu = FusedSwiGLU()
    torch_swiglu = PyTorchUnfusedSwiGLU()

    out_fused = fused_swiglu(gate, up)
    out_torch = torch_swiglu(gate, up)
    assert torch.allclose(out_fused, out_torch, atol=1e-6)


def test_fused_swiglu_negatif_ve_pozitif_degerler():
    """2. Fused SwiGLU aşırı negatif ve pozitif değerlerde sayısal kararlılık göstermelidir."""
    gate = torch.tensor([[-50.0, 0.0, 50.0]], requires_grad=True)
    up = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    fused = FusedSwiGLU()(gate, up)
    ref = PyTorchUnfusedSwiGLU()(gate, up)
    assert torch.allclose(fused, ref, atol=1e-5)


def test_fused_autograd_dgate_gradyan(test_tensörleri):
    """3. Geri geçişte dGate gradyanı PyTorch autograd ile birebir eşleşmelidir."""
    gate, up = test_tensörleri
    out_fused = FusedSwiGLU()(gate, up)
    out_fused.sum().backward()

    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    out_ref = PyTorchUnfusedSwiGLU()(gate_ref, up_ref)
    out_ref.sum().backward()

    assert torch.allclose(gate.grad, gate_ref.grad, atol=1e-6)


def test_fused_autograd_dup_gradyan(test_tensörleri):
    """4. Geri geçişte dUp gradyanı PyTorch autograd ile birebir eşleşmelidir."""
    gate, up = test_tensörleri
    out_fused = FusedSwiGLU()(gate, up)
    out_fused.sum().backward()

    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    out_ref = PyTorchUnfusedSwiGLU()(gate_ref, up_ref)
    out_ref.sum().backward()

    assert torch.allclose(up.grad, up_ref.grad, atol=1e-6)


def test_swiglu_mlp_ileri_ve_geri_gecis():
    """5. SwiGLUMLP bloğu uçtan uca ileri ve geri geçişi hatasız tamamlamalıdır."""
    mlp = SwiGLUMLP(hidden_dim=64, intermediate_dim=128)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = mlp(x)
    assert out.shape == (2, 8, 64)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 8, 64)


def test_swiglu_bellek_profilleyici_tasarruf_orani():
    """6. Bellek profilleyici HBM geçiş sayısının 8'den 3'e düştüğünü (%62.5 kazanç) doğrulamalıdır."""
    analiz = SwiGLUBellekProfilleyici.katman_bazli_hbm_analizi(batch_size=2, seq_len=1024, intermediate_dim=4096)
    assert analiz["tasarruf_orani"] == pytest.approx(2.67, abs=1e-2)
    assert analiz["triton_ara_bellek_mb"] == 0.0


def test_tam_model_swiglu_raporu():
    """7. Tam model raporu Mistral-7B, Llama-3-8B ve Llama-3-70B'yi içermelidir."""
    rapor = SwiGLUBellekProfilleyici.tam_model_swiglu_raporu()
    assert len(rapor) == 3
    adlar = [r["model_adi"] for r in rapor]
    assert "Llama-3-70B" in adlar
    assert "Mistral-7B" in adlar


def test_gorsellestirme_cikti_dosyasi(tmp_path):
    """8. SwiGLUGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_swiglu_paneli.png")
    katman_analizi = SwiGLUBellekProfilleyici.katman_bazli_hbm_analizi(batch_size=2, seq_len=512, intermediate_dim=2048)
    model_raporu = SwiGLUBellekProfilleyici.tam_model_swiglu_raporu()

    SwiGLUGorsellestirici.teshis_paneli_olustur(
        katman_analizi=katman_analizi,
        model_raporu=model_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
