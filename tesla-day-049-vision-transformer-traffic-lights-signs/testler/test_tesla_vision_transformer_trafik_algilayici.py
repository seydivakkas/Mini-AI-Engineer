"""
Tesla Vision Transformer (ViT) Birim Testleri (PyTest)
=======================================================
Bu test paketi; ViT yama gömmeyi, ölçekli noktasal çarpım öz-dikkatini,
trafik ışığı durumu ve geri sayım süresi çıkarımını test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_vision_transformer_trafik_algilayici import TeslaVisionTransformerTrafficDetector


def test_vit_patch_embedding_boyutlari():
    """64x64 görüntünün 8x8 yamalara bölündüğünde 64 yama tokeni ürettiği test edilir."""
    vit = TeslaVisionTransformerTrafficDetector(img_size=64, patch_size=8, embed_dim=32)
    img = np.zeros((64, 64, 3), dtype=np.float32)

    tokens = vit.extract_patch_embeddings(img)

    assert tokens.shape == (64, 32)


def test_self_attention_softmax_ozelligi():
    """Öz-dikkat matrisinin her satırının toplamının 1.0 (olasılık dağılımı) olduğu test edilir."""
    vit = TeslaVisionTransformerTrafficDetector(embed_dim=32, num_heads=4)
    tokens = np.random.normal(0, 1, (16, 32)).astype(np.float32)

    context, attn = vit.compute_self_attention(tokens)

    assert attn.shape == (16, 16)
    satir_toplamlari = np.sum(attn, axis=-1)
    assert np.allclose(satir_toplamlari, 1.0)


def test_trafik_isigi_ve_levha_cikarimi():
    """ViT çıkarımının geçerli durum, güven skoru ve geri sayım ürettiği test edilir."""
    vit = TeslaVisionTransformerTrafficDetector()
    img = np.ones((64, 64, 3), dtype=np.float32) * 200.0

    res = vit.forward_vit_traffic_detector(img)

    assert res["traffic_light_state"] in vit.traffic_light_classes
    assert res["traffic_light_confidence"] > 0.8
    assert res["countdown_seconds"] >= 0.0
    assert res["traffic_sign"] in vit.traffic_sign_classes
    assert res["traffic_sign_confidence"] > 0.8
