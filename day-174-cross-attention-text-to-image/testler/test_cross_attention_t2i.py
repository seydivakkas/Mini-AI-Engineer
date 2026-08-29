"""
GÜN 174: Metinden Görüntüye (Text-to-Image) Cross-Attention Mekanizması Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.spatial_cross_attention import SpatialCrossAttention
from src.text_conditioned_unet_dit import TextConditionedDiffusionBlock
from src.dikkat_haritasi_analizoru import DikkatHaritasiAnalizoru
from src.gorsellestirici import CrossAttentionGorsellestirici


def test_spatial_cross_attention_ileri_besleme():
    """Çapraz dikkat katmanının doğru boyutta tensör ve dikkat haritası ürettiğini test eder."""
    attn = SpatialCrossAttention(query_dim=64, context_dim=128, heads=4, dim_head=16)
    x = torch.randn(2, 64, 16, 16)  # [B, C, H, W]
    context = torch.randn(2, 10, 128)  # [B, S_text, context_dim]

    out, attn_map = attn(x, context)

    assert out.shape == (2, 64, 16, 16)
    assert attn_map.shape == (2, 256, 10)  # [B, H*W=256, S_text=10]


def test_cross_attention_softmax_toplami_1():
    """Mekansal dikkat ağırlıklarının metin boyutu boyunca toplamının 1.0 olduğunu test eder."""
    attn = SpatialCrossAttention(query_dim=32, context_dim=64, heads=2, dim_head=16)
    x = torch.randn(1, 32, 8, 8)
    context = torch.randn(1, 5, 64)

    _, attn_map = attn(x, context)

    toplam = attn_map.sum(dim=-1)  # [1, 64]
    assert torch.allclose(toplam, torch.ones_like(toplam), atol=1e-4)


def test_text_conditioned_diffusion_block():
    """Residual Conv + Self-Attn + Cross-Attn içeren hibrit bloğun doğru tensör döndürdüğünü test eder."""
    block = TextConditionedDiffusionBlock(channels=64, context_dim=128, heads=2)
    x = torch.randn(2, 64, 8, 8)
    context = torch.randn(2, 6, 128)

    out, attn_map = block(x, context)

    assert out.shape == (2, 64, 8, 8)
    assert attn_map.shape == (2, 64, 6)


def test_text_conditioned_block_gradyan_akisi():
    """Modelin geriye yayılımda gradyanları sorunsuz hesapladığını test eder."""
    block = TextConditionedDiffusionBlock(channels=32, context_dim=64, heads=2)
    x = torch.randn(1, 32, 4, 4)
    context = torch.randn(1, 4, 64)

    out, _ = block(x, context)
    loss = out.sum()
    loss.backward()

    assert block.conv1.weight.grad is not None
    assert block.cross_attn.to_k.weight.grad is not None


def test_kelime_odak_skorlarini_hesapla():
    """Kelime bazlı dikkat odak skorlarının doğru ayrıştırıldığını test eder."""
    attn_map = torch.rand(1, 64, 4)  # [B=1, HW=64, S=4]
    kelimeler = ["kedi", "kask", "galaksi", "uzay"]

    sonuclar = DikkatHaritasiAnalizoru.kelime_odak_skorlarini_hesapla(attn_map, kelimeler, H=8, W=8)

    assert len(sonuclar) == 4
    assert sonuclar[0]["kelime"] == "kedi"
    assert "tepe_konum" in sonuclar[0]


def test_ornek_cross_attention_raporu():
    """Örnek raporun beklenen prompt ve semantik hizalama metriklerini içerdiğini test eder."""
    rapor = DikkatHaritasiAnalizoru.ornek_cross_attention_raporu()
    assert len(rapor["kelime_skorlari"]) == 5
    assert rapor["metin_piksel_hizalama_dogrulugu"] > 0.90


def test_cross_attention_bos_kelime_listesi():
    """Kelime listesi boş olduğunda analizörün boş liste döndürdüğünü test eder."""
    attn_map = torch.rand(1, 64, 4)
    sonuclar = DikkatHaritasiAnalizoru.kelime_odak_skorlarini_hesapla(attn_map, [], H=8, W=8)
    assert len(sonuclar) == 0


def test_gorsellestirici_pano_uretme():
    """6 panelli Cross-Attention teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = DikkatHaritasiAnalizoru.ornek_cross_attention_raporu()
    gorsellestirici = CrossAttentionGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_cross_attn_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
