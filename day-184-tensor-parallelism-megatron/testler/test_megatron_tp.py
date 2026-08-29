"""
PyTest Birim Testleri - Day 184: Megatron-LM Tensor Parallelism (TP).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.megatron_tp_motoru import (
    ColumnParallelLinear,
    RowParallelLinear,
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from src.megatron_transformer_blok import (
    MegatronMLP,
    MegatronSelfAttention,
    MegatronTransformerKatmani,
    TPDogrulayici,
)
from src.gorsellestirici import TPGorsellestirici


def test_column_parallel_linear_boyutlari():
    """1. ColumnParallelLinear katmanı çıkış özelliklerini tam olarak K'ya bölmelidir."""
    in_feat = 256
    out_feat = 1024
    tp_world_size = 4

    col_layer = ColumnParallelLinear(
        in_features=in_feat,
        out_features=out_feat,
        tp_world_size=tp_world_size,
        tp_rank=0,
    )

    assert col_layer.weight.shape == (out_feat // tp_world_size, in_feat)
    assert col_layer.bias.shape == (out_feat // tp_world_size,)

    x = torch.randn(8, in_feat)
    out = col_layer(x)
    assert out.shape == (8, out_feat // tp_world_size)


def test_row_parallel_linear_boyutlari():
    """2. RowParallelLinear katmanı giriş özelliklerini K'ya bölmeli ve tam boyutta çıktı üretmelidir."""
    in_feat = 1024
    out_feat = 256
    tp_world_size = 4

    row_layer = RowParallelLinear(
        in_features=in_feat,
        out_features=out_feat,
        tp_world_size=tp_world_size,
        tp_rank=0,
    )

    assert row_layer.weight.shape == (out_feat, in_feat // tp_world_size)
    assert row_layer.bias.shape == (out_feat,)

    x_part = torch.randn(8, in_feat // tp_world_size)
    out = row_layer(x_part)
    assert out.shape == (8, out_feat)


def test_f_ve_g_operator_fonksiyonlari():
    """3. f (copy) ve g (reduce) autograd operatörleri doğru çalışmalıdır."""
    t = torch.randn(4, 16, requires_grad=True)
    out_f = copy_to_tensor_model_parallel_region(t, world_size=2)
    assert torch.allclose(out_f, t)

    out_g = reduce_from_tensor_model_parallel_region(t, world_size=2)
    assert torch.allclose(out_g, t)


def test_megatron_mlp_ileri_gecis():
    """4. MegatronMLP ileri geçişi doğru boyutta tensör üretmelidir."""
    mlp = MegatronMLP(hidden_size=128, ffn_hidden_size=512, tp_world_size=2, tp_rank=0)
    x = torch.randn(4, 128)
    out = mlp(x)
    assert out.shape == (4, 128)


def test_megatron_self_attention_baslik_bolunmesi():
    """5. MegatronSelfAttention dikkat başlıklarını tam olarak H/K şeklinde bölmelidir."""
    attn = MegatronSelfAttention(hidden_size=256, num_heads=8, tp_world_size=4, tp_rank=0)
    assert attn.num_heads_per_partition == 2
    assert attn.head_dim == 32

    x = torch.randn(2, 16, 256)
    out = attn(x)
    assert out.shape == (2, 16, 256)


def test_megatron_transformer_katmani_cikti_sekli():
    """6. MegatronTransformerKatmani uçtan uca Transformer bloğu doğru çalışmalıdır."""
    katman = MegatronTransformerKatmani(
        hidden_size=128,
        num_heads=4,
        ffn_hidden_size=512,
        tp_world_size=2,
        tp_rank=0,
    )
    x = torch.randn(2, 8, 128)
    out = katman(x)
    assert out.shape == (2, 8, 128)


def test_tp_dogrulayici_matematiksel_eslesme():
    """7. Standart tek GPU MLP ile Megatron TP matematiksel olarak birebir eşleşmelidir."""
    for k in [2, 4]:
        res = TPDogrulayici.mlp_esdegerlik_dogrula(hidden_size=128, ffn_hidden_size=512, tp_world_size=k)
        assert res["matematiksel_olarak_eslesiyor"] is True
        assert res["maksimum_mutlak_hata"] < 1e-3


def test_gorsellestirme_cikti_dosyasi(tmp_path):
    """8. TPGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_tp_paneli.png")
    sonuclar = [
        TPDogrulayici.mlp_esdegerlik_dogrula(hidden_size=128, ffn_hidden_size=512, tp_world_size=2),
        TPDogrulayici.mlp_esdegerlik_dogrula(hidden_size=128, ffn_hidden_size=512, tp_world_size=4),
    ]

    TPGorsellestirici.tp_teshis_paneli_olustur(
        dogrulama_sonuclari=sonuclar,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
