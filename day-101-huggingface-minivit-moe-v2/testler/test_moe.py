"""
MiniViT-MoE v2 Birim ve Entegrasyon Testleri (Day 101).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.konfigurasyon import MiniViTMoEConfig
from src.moe_katmanlari import (
    RMSNorm,
    ModernDikkatSDPA,
    SwiGLUUzmani,
    TopKRouter,
    MoEKatmani,
    MoETransformerBlok,
)
from src.model import MiniViTMoEForImageClassification
from src.hub_yoneticisi import MoEHubYoneticisi
from src.gorsellestirici import MoEBuyukFinalGorsellestirici


def test_topk_router_ileri_gecis():
    """Top-K router çıktılarının tensör boyutlarını ve ağırlık toplamlarını test eder."""
    router = TopKRouter(dim=64, uzman_sayisi=4, aktif_uzman_sayisi=2)
    x = torch.randn(2, 16, 64)
    topk_idx, topk_weights, aux_loss = router(x)

    assert topk_idx.shape == (32, 2)
    assert topk_weights.shape == (32, 2)
    # Ağırlıkların toplamının 1.0 olduğunu doğrula
    toplam = topk_weights.sum(dim=-1)
    assert torch.allclose(toplam, torch.ones_like(toplam), atol=1e-3)
    assert aux_loss.item() >= 0.0


def test_swiglu_uzman_ileri_gecis():
    """Bireysel SwiGLU uzman bloğunun ileri geçişini test eder."""
    uzman = SwiGLUUzmani(in_features=64, hidden_features=128)
    x = torch.randn(8, 64)
    out = uzman(x)
    assert out.shape == (8, 64)


def test_moe_katmani_ileri_gecis():
    """Sparse MoE katmanının ileri geçişini ve aux loss üretimini test eder."""
    cfg = MiniViTMoEConfig(gizli_boyut=64, uzman_sayisi=4, aktif_uzman_sayisi=2)
    moe = MoEKatmani(cfg)
    x = torch.randn(2, 16, 64)
    out, aux_loss = moe(x)
    assert out.shape == (2, 16, 64)
    assert aux_loss is not None
    assert aux_loss.item() >= 0.0


def test_moe_transformer_blok():
    """MoETransformerBlok ileri geçişini test eder."""
    cfg = MiniViTMoEConfig(gizli_boyut=64, dikkat_baslik_sayisi=4, uzman_sayisi=4, aktif_uzman_sayisi=2)
    blok = MoETransformerBlok(cfg)
    x = torch.randn(2, 16, 64)
    out, aux_loss = blok(x)
    assert out.shape == (2, 16, 64)


def test_minivit_moe_ileri_gecis_ve_kayip():
    """Modelin uçtan uca logits ve loss hesaplamasını test eder."""
    cfg = MiniViTMoEConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        gizli_boyut=64,
        katman_sayisi=2,
        dikkat_baslik_sayisi=2,
        uzman_sayisi=4,
        aktif_uzman_sayisi=2,
        sinif_sayisi=10,
    )
    model = MiniViTMoEForImageClassification(cfg).train()
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([1, 4])
    cikis = model(pixel_values=x, labels=y)

    assert cikis.logits.shape == (2, 10)
    assert cikis.loss is not None
    assert cikis.loss.item() > 0.0


def test_aktif_parametre_hesaplama():
    """Toplam vs aktif parametre sayısını ve tasarruf oranını test eder."""
    cfg = MiniViTMoEConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        gizli_boyut=64,
        katman_sayisi=2,
        uzman_sayisi=4,
        aktif_uzman_sayisi=2,
    )
    model = MiniViTMoEForImageClassification(cfg)
    istatistik = model.aktif_parametre_hesapla()

    assert istatistik["toplam_parametre"] > istatistik["aktif_parametre"]
    assert istatistik["tasarruf_orani_yuzde"] > 0.0


def test_hub_yoneticisi_yerel_paket():
    """MoEHubYoneticisi'nin safetensors ve config dosyalarını ürettiğini test eder."""
    cfg = MiniViTMoEConfig(goruntu_boyutu=32, yama_boyutu=4, gizli_boyut=32, katman_sayisi=1)
    model = MiniViTMoEForImageClassification(cfg)
    yonetici = MoEHubYoneticisi(repo_adi="test-user/minivit-moe-test")

    with tempfile.TemporaryDirectory() as tmp_dir:
        hedef = os.path.join(tmp_dir, "moe_paket")
        yonetici.yerel_paket_olustur(model, hedef_dizin=hedef)
        assert os.path.exists(os.path.join(hedef, "model.safetensors"))
        assert os.path.exists(os.path.join(hedef, "config.json"))
        assert os.path.exists(os.path.join(hedef, "preprocessor_config.json"))
        assert os.path.exists(os.path.join(hedef, "README.md"))
        assert os.path.exists(os.path.join(hedef, "app.py"))


def test_gorsellestirici_pano_uretme():
    """6 panelli büyük final teşhis panosunun oluşturulduğunu test eder."""
    gorsellestirici = MoEBuyukFinalGorsellestirici(dpi=100)
    ornek_veriler = {
        "uzman_yukleri": [24.0, 26.0, 25.0, 25.0],
        "toplam_parametre": 1580000,
        "aktif_parametre": 803000,
        "dense_parametre": 805000,
        "p50_gecikme_ms": 12.3,
        "throughput_fps": 1300,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit_yolu = os.path.join(tmp_dir, "test_buyuk_final_paneli.png")
        gorsellestirici.pano_olustur(ornek_veriler, kayit_yolu=kayit_yolu)
        assert os.path.exists(kayit_yolu)
        assert os.path.getsize(kayit_yolu) > 1000
