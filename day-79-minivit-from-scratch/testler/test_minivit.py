"""
Sıfırdan Mini Vision Transformer (MiniViT) Birim Testleri
---------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import torch
import numpy as np

from src.patch_gocume import YamaGomulmeKatmani
from src.minivit_modeli import MiniVisionTransformer
from src.dikkat_haritasi import ViTDikkatCikarici


def test_patch_embedding_boyut_ve_bolunebilirlik():
    patch_layer = YamaGomulmeKatmani(gorsel_boyutu=32, yama_boyutu=4, giris_kanali=3, gomulme_boyutu=64)
    x = torch.randn(2, 3, 32, 32)
    out = patch_layer(x)

    # N = (32/4) * (32/4) = 64 yama
    assert out.shape == (2, 64, 64)
    assert patch_layer.toplam_yama_sayisi == 64


def test_patch_embedding_uyumsuz_boyut_hatasi():
    with pytest.raises(AssertionError):
        # 33, 4'e tam bölünmez
        _ = YamaGomulmeKatmani(gorsel_boyutu=33, yama_boyutu=4)


def test_cls_token_ve_pos_embed_sekli():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        giris_kanali=3,
        sinif_sayisi=10,
        gomulme_boyutu=64,
        derinlik=2,
        kafa_sayisi=4
    )

    # Toplam token sayısı: 64 yama + 1 [CLS] = 65
    assert model.cls_token.shape == (1, 1, 64)
    assert model.pos_embed.shape == (1, 65, 64)


def test_minivit_ileri_gecis_logitler():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        giris_kanali=3,
        sinif_sayisi=10,
        gomulme_boyutu=64,
        derinlik=2,
        kafa_sayisi=4
    )
    x = torch.randn(3, 3, 32, 32)
    logits = model(x)

    assert logits.shape == (3, 10)


def test_minivit_dikkat_haritalari_listesi():
    derinlik = 3
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        giris_kanali=3,
        sinif_sayisi=5,
        gomulme_boyutu=64,
        derinlik=derinlik,
        kafa_sayisi=4
    )
    x = torch.randn(2, 3, 32, 32)
    logits, dikkatler = model(x, dikkat_haritalarini_don=True)

    assert logits.shape == (2, 5)
    assert len(dikkatler) == derinlik
    for att in dikkatler:
        assert att.shape == (2, 4, 65, 65)


def test_minivit_gradyan_akisi():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        giris_kanali=3,
        sinif_sayisi=10,
        gomulme_boyutu=32,
        derinlik=2,
        kafa_sayisi=2
    )
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    logits = model(x)
    targets = torch.tensor([1, 4])
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    assert x.grad is not None
    assert model.cls_token.grad is not None
    assert model.pos_embed.grad is not None
    assert model.patch_embed.projeksiyon.weight.grad is not None
    assert model.head.weight.grad is not None


def test_attention_rollout_hesaplama():
    b, h, n = 2, 4, 16
    # 2 katmanlı rastgele dikkat matrisi
    att1 = torch.softmax(torch.randn(b, h, n, n), dim=-1)
    att2 = torch.softmax(torch.randn(b, h, n, n), dim=-1)

    rollout = ViTDikkatCikarici.hesapla_attention_rollout([att1, att2])
    assert rollout.shape == (b, n, n)
    # Satır toplamları 1.0 olmalı
    satir_toplami = rollout.sum(dim=-1)
    assert torch.allclose(satir_toplami, torch.ones_like(satir_toplami), atol=1e-4)


def test_cls_dikkat_haritasi_2d_boyut():
    b, h, n = 2, 4, 65 # 1 CLS + 64 yama (8x8)
    att = torch.softmax(torch.randn(b, h, n, n), dim=-1)
    
    heatmaps = ViTDikkatCikarici.cls_dikkat_haritasi_2d(
        [att],
        grid_boyutu=(8, 8),
        orijinal_boyut=(32, 32)
    )

    assert heatmaps.shape == (2, 32, 32)
    assert np.all(heatmaps >= 0.0) and np.all(heatmaps <= 1.0)
