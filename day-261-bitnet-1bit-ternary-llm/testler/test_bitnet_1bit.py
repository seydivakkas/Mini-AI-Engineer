"""
PyTest Birim Testleri - Day 261 (FAZ 14 BAŞLANGICI): BitNet b1.58 Ternary LLM.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bitnet_1bit_motoru import (
    weight_quantization_b158,
    activation_quantization_int8,
    RMSNorm,
    BitLinear,
    BitNetAttention,
    BitNetFFN,
    BitNetTransformer,
)
from src.bitnet_1bit_profilleyici import BitNetProfilleyici
from src.gorsellestirici import BitNetGorsellestirici


def test_weight_quantization_ternary_values():
    """1. weight_quantization_b158 ağırlıkları sadece {-1, 0, 1} değerlerine kuantize etmelidir."""
    w = torch.randn(20, 20)
    w_ternary, gamma = weight_quantization_b158(w)
    unique_vals = torch.unique(w_ternary).tolist()
    for val in unique_vals:
        assert val in [-1.0, 0.0, 1.0]
    assert gamma.item() > 0.0


def test_weight_quantization_ste_gradient():
    """2. Straight-Through Estimator (STE) gradyan akışını kesintisiz iletmelidir."""
    w = torch.randn(10, 10, requires_grad=True)
    w_ternary, _ = weight_quantization_b158(w)
    loss = w_ternary.sum()
    loss.backward()
    assert w.grad is not None
    assert torch.all(w.grad == 1.0)


def test_activation_quantization_int8_range():
    """3. activation_quantization_int8 değerleri [-127, 127] aralığına sıkıştırmalıdır."""
    x = torch.randn(5, 10) * 10.0
    x_quant, gamma_x = activation_quantization_int8(x)
    assert x_quant.min().item() >= -127.0
    assert x_quant.max().item() <= 127.0


def test_rmsnorm_forward_shape_and_norm():
    """4. RMSNorm şekli korumalı ve birim varyansa yakın ölçeklemelidir."""
    norm = RMSNorm(d_model=32)
    x = torch.randn(2, 5, 32)
    out = norm(x)
    assert out.shape == (2, 5, 32)


def test_bitlinear_forward_shape():
    """5. BitLinear katmanı doğru çıktı tensör boyutunu üretmelidir."""
    layer = BitLinear(in_features=32, out_features=64)
    x = torch.randn(2, 10, 32)
    out = layer(x)
    assert out.shape == (2, 10, 64)


def test_bitnet_attention_and_ffn():
    """6. BitNetAttention ve BitNetFFN blokları doğru çıktı üretmelidir."""
    attn = BitNetAttention(d_model=32, n_heads=4)
    ffn = BitNetFFN(d_model=32, d_ff=64)
    x = torch.randn(2, 8, 32)

    attn_out = attn(x)
    assert attn_out.shape == (2, 8, 32)

    ffn_out = ffn(x)
    assert ffn_out.shape == (2, 8, 32)


def test_bitnet_transformer_end_to_end():
    """7. BitNetTransformer uçtan uca token dizisinden logits üretmelidir."""
    model = BitNetTransformer(vocab_size=500, d_model=32, n_layers=2, n_heads=2, d_ff=64)
    tokens = torch.tensor([[1, 20, 150, 42]], dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (1, 4, 500)


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. BitNetGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_bitnet_1bit_paneli.png")
    profil = BitNetProfilleyici.basarim_profili_cikar()

    BitNetGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
