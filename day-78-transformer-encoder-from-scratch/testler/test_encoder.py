"""
Sıfırdan Transformer Encoder Bloğu Birim Testleri
-------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import torch

from src.pozisyonel_kodlama import SinusoidalPozisyonelKodlama, OgrenilebilirPozisyonelKodlama
from src.layer_norm import OzelLayerNorm
from src.feed_forward import BeslemeliIleriAg
from src.encoder_blogu import TransformerEncoderBlogu, TransformerEncoderGovdesi


def test_sinusoidal_pozisyonel_kodlama_boyut_ve_degerler():
    pe = SinusoidalPozisyonelKodlama(model_boyutu=32, maksimum_uzunluk=64)
    x = torch.zeros(2, 10, 32)
    out = pe(x)
    
    assert out.shape == (2, 10, 32)
    # Sinüs ve kosinüs değerleri [-1.0, 1.0] aralığında olmalı
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_ogrenilebilir_pozisyonel_kodlama_gradyan():
    pe = OgrenilebilirPozisyonelKodlama(model_boyutu=32, maksimum_uzunluk=64)
    x = torch.randn(2, 10, 32)
    out = pe(x)
    loss = out.sum()
    loss.backward()

    assert pe.pos_embed.grad is not None
    assert pe.pos_embed.grad.norm().item() > 0.0


def test_ozel_layer_norm_sifir_ortalama_birim_varyans():
    ln = OzelLayerNorm(normalize_edilecek_boyut=32)
    x = torch.randn(4, 16, 32) * 5.0 + 10.0 # Yüksek ortalama ve varyans
    out = ln(x)

    # Son boyut boyunca ortalama ~0 ve varyans ~1 olmalı
    ort = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)

    assert torch.allclose(ort, torch.zeros_like(ort), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_feed_forward_gelu_genisleme_boyut():
    ffn = BeslemeliIleriAg(model_boyutu=32, genisleme_faktoru=4, aktivasyon="gelu")
    x = torch.randn(2, 8, 32)
    out = ffn(x)

    assert out.shape == (2, 8, 32)
    assert ffn.fc1.out_features == 128
    assert ffn.fc2.out_features == 32


def test_transformer_encoder_blogu_pre_ln_ileri_gecis():
    blok = TransformerEncoderBlogu(model_boyutu=32, kafa_sayisi=2, norm_tipi="pre_ln")
    x = torch.randn(2, 8, 32)
    out, att = blok(x)

    assert out.shape == (2, 8, 32)
    assert att.shape == (2, 2, 8, 8)


def test_transformer_encoder_blogu_post_ln_ileri_gecis():
    blok = TransformerEncoderBlogu(model_boyutu=32, kafa_sayisi=2, norm_tipi="post_ln")
    x = torch.randn(2, 8, 32)
    out, att = blok(x)

    assert out.shape == (2, 8, 32)
    assert att.shape == (2, 2, 8, 8)


def test_transformer_encoder_govdesi_cok_katman():
    govde = TransformerEncoderGovdesi(
        katman_sayisi=3,
        model_boyutu=32,
        kafa_sayisi=2,
        norm_tipi="pre_ln"
    )
    x = torch.randn(2, 8, 32)
    out, atts, layers = govde(x, tum_katmanlari_don=True)

    assert out.shape == (2, 8, 32)
    assert len(atts) == 3
    assert len(layers) == 3
    for att in atts:
        assert att.shape == (2, 2, 8, 8)


def test_pre_ln_gradyan_akis_guvencesi():
    govde = TransformerEncoderGovdesi(
        katman_sayisi=4,
        model_boyutu=32,
        kafa_sayisi=2,
        norm_tipi="pre_ln"
    )
    x = torch.randn(2, 8, 32, requires_grad=True)
    out, _, _ = govde(x)
    hedef = torch.randn_like(out)
    loss = torch.nn.functional.mse_loss(out, hedef)
    loss.backward()

    assert x.grad is not None
    assert x.grad.norm().item() > 0.0
    for idx, blok in enumerate(govde.bloklar):
        assert blok.ffn.fc1.weight.grad is not None
        assert blok.ffn.fc1.weight.grad.norm().item() > 0.0, f"Katman {idx+1} gradyan almalıdır!"
