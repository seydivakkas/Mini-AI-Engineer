"""
GÜN 162: Görüntü Token Sıkıştırma Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.qformer_sikistirici import QFormerSikistirici
from src.c_abstractor_sikistirici import CAbstractorSikistirici
from src.spatial_pooling_sikistirici import SpatialPoolingSikistirici
from src.sikistirma_karsilastirici import SikistirmaKarsilastirici
from src.gorsellestirici import TokenSikistirmaGorsellestirici


def test_qformer_sikistirici_cikti_boyutu():
    """Q-Former'ın 256 tokenı 32 query tokenına (512d) indirdiğini test eder."""
    qformer = QFormerSikistirici(num_query_tokens=32, d_vision=768, d_model=512, katman_sayisi=1)
    dummy_visual = torch.randn(2, 256, 768)
    out = qformer(dummy_visual)

    assert out.shape == (2, 32, 512)


def test_c_abstractor_sikistirici_cikti_boyutu():
    """C-Abstractor'ın stride=2 ile 256 tokenı 64 tokena indirdiğini test eder."""
    abstractor = CAbstractorSikistirici(d_vision=768, d_model=512, stride=2)
    dummy_visual = torch.randn(2, 256, 768)
    out = abstractor(dummy_visual)

    assert out.shape == (2, 64, 512)


def test_spatial_pooling_2x2_cikti_boyutu():
    """Spatial Pooling 2x2'nin 256 tokenı 64 tokena indirdiğini test eder."""
    pooler = SpatialPoolingSikistirici(d_vision=768, d_model=512, pool_boyutu=2)
    dummy_visual = torch.randn(2, 256, 768)
    out = pooler(dummy_visual)

    assert out.shape == (2, 64, 512)


def test_spatial_pooling_4x4_cikti_boyutu():
    """Spatial Pooling 4x4'ün 256 tokenı 16 tokena indirdiğini test eder."""
    pooler = SpatialPoolingSikistirici(d_vision=768, d_model=512, pool_boyutu=4)
    dummy_visual = torch.randn(1, 256, 768)
    out = pooler(dummy_visual)

    # 16/4 = 4x4 = 16 token
    assert out.shape == (1, 16, 512)


def test_qformer_gradyan_akisi():
    """Q-Former öğrenilebilir query tokenlarının gradyan aldığını test eder."""
    qformer = QFormerSikistirici(num_query_tokens=16, d_vision=256, d_model=128, katman_sayisi=1)
    dummy_in = torch.randn(2, 64, 256, requires_grad=True)
    out = qformer(dummy_in)
    loss = out.sum()
    loss.backward()

    assert qformer.query_tokens.grad is not None
    assert dummy_in.grad is not None


def test_c_abstractor_gradyan_akisi():
    """C-Abstractor konvolüsyonel katmanlarının gradyan aldığını test eder."""
    abstractor = CAbstractorSikistirici(d_vision=256, d_model=128, stride=2)
    dummy_in = torch.randn(2, 64, 256, requires_grad=True)
    out = abstractor(dummy_in)
    loss = out.sum()
    loss.backward()

    assert dummy_in.grad is not None


def test_sikistirma_karsilastirici_analizi():
    """Karşılaştırıcı modülün 4 yöntemi eksiksiz kıyasladığını test eder."""
    rapor = SikistirmaKarsilastirici.yontemleri_karsilastir(batch_size=2)

    assert len(rapor) == 4
    assert rapor["1. Ham ViT (Sıkıştırmasız)"]["token_sayisi"] == 256
    assert rapor["4. BLIP-2 Q-Former (32 Query)"]["token_sayisi"] == 32
    assert rapor["4. BLIP-2 Q-Former (32 Query)"]["sikistirma_orani"] == 87.5


def test_gorsellestirici_pano_uretme():
    """6 panelli token sıkıştırma teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = SikistirmaKarsilastirici.yontemleri_karsilastir(batch_size=2)
    gorsellestirici = TokenSikistirmaGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_compression_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
