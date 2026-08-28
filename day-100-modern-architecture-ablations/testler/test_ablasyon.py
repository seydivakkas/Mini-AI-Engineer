"""
Modern Mimari Ablasyon Birim ve Entegrasyon Testleri (Day 100).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.konfigurasyon import ModernMiniViTConfig
from src.modern_katmanlar import (
    RMSNorm,
    SwiGLU,
    GELUFFN,
    ModernDikkatSDPA,
    ModernTransformerBlok,
)
from src.model import ModernMiniViTForImageClassification
from src.ablasyon_motoru import AblasyonMotoru
from src.gorsellestirici import AblasyonGorsellestirici


def test_rmsnorm_ileri_gecis():
    """RMSNorm katmanının ileri geçişini ve normalizasyon çıktısını test eder."""
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 16, 64) * 5.0
    out = norm(x)
    assert out.shape == (2, 16, 64)
    # RMS değerinin yaklaşık 1.0 olduğunu doğrula
    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_swiglu_ileri_gecis():
    """SwiGLU kapılı aktivasyon katmanının ileri geçişini test eder."""
    swiglu = SwiGLU(in_features=64, hidden_features=128)
    x = torch.randn(2, 16, 64)
    out = swiglu(x)
    assert out.shape == (2, 16, 64)


def test_sdpa_dikkat_ileri_gecis():
    """ModernDikkatSDPA (Scaled Dot-Product Attention) katmanını test eder."""
    dikkat = ModernDikkatSDPA(dim=64, num_heads=4)
    x = torch.randn(2, 16, 64)
    out = dikkat(x)
    assert out.shape == (2, 16, 64)


def test_modern_transformer_blok_varyasyonlari():
    """ModernTransformerBlok'un farklı norm ve ffn varyasyonlarını test eder."""
    # 1. RMSNorm + SwiGLU + SDPA
    cfg1 = ModernMiniViTConfig(gizli_boyut=64, dikkat_baslik_sayisi=4, norm_turu="rmsnorm", ffn_turu="swiglu", dikkat_turu="sdpa")
    blok1 = ModernTransformerBlok(cfg1)
    x = torch.randn(2, 16, 64)
    assert blok1(x).shape == (2, 16, 64)

    # 2. LayerNorm + GELU + Standard
    cfg2 = ModernMiniViTConfig(gizli_boyut=64, dikkat_baslik_sayisi=4, norm_turu="layernorm", ffn_turu="gelu", dikkat_turu="standard")
    blok2 = ModernTransformerBlok(cfg2)
    assert blok2(x).shape == (2, 16, 64)


def test_modern_minivit_ileri_gecis():
    """ModernMiniViTForImageClassification modelinin ileri geçiş tensör boyutlarını test eder."""
    cfg = ModernMiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        gizli_boyut=64,
        katman_sayisi=2,
        dikkat_baslik_sayisi=2,
        sinif_sayisi=10,
        norm_turu="rmsnorm",
        ffn_turu="swiglu",
        dikkat_turu="sdpa",
    )
    model = ModernMiniViTForImageClassification(cfg).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        cikis = model(x)
    assert cikis.logits.shape == (2, 10)


def test_ablasyon_motoru_varyant_olusturma():
    """AblasyonMotoru'nun 4 farklı mimari varyant ürettiğini test eder."""
    motor = AblasyonMotoru(cihaz=torch.device("cpu"))
    varyantlar = motor.varyantlari_olustur()
    assert len(varyantlar) == 4
    for isim, m in varyantlar.items():
        assert isinstance(m, ModernMiniViTForImageClassification)


def test_ablasyon_motoru_olcum():
    """AblasyonMotoru'nun tekil varyant ölçümünü test eder."""
    motor = AblasyonMotoru(cihaz=torch.device("cpu"))
    cfg = ModernMiniViTConfig(gizli_boyut=32, katman_sayisi=1, dikkat_baslik_sayisi=2)
    model = ModernMiniViTForImageClassification(cfg)
    girdi = torch.randn(2, 3, 32, 32)

    sonuc = motor.varyanti_olc(model, girdi, iterasyon=3)
    assert "p50_gecikme_ms" in sonuc
    assert "parametre_sayisi" in sonuc
    assert "throughput_fps" in sonuc
    assert sonuc["parametre_sayisi"] > 0


def test_gorsellestirici_pano_uretme():
    """6 panelli ablasyon teşhis panosunun oluşturulduğunu test eder."""
    gorsellestirici = AblasyonGorsellestirici(dpi=100)
    ornek_sonuclar = {
        "01_MiniViT_Base": {"p50_gecikme_ms": 3.2, "throughput_fps": 310, "tepe_bellek_mb": 15.0, "parametre_sayisi": 540000},
        "02_+RMSNorm": {"p50_gecikme_ms": 2.9, "throughput_fps": 340, "tepe_bellek_mb": 14.8, "parametre_sayisi": 540000},
        "03_+SwiGLU": {"p50_gecikme_ms": 2.8, "throughput_fps": 355, "tepe_bellek_mb": 14.5, "parametre_sayisi": 560000},
        "04_Modern_MiniViT_v2": {"p50_gecikme_ms": 2.3, "throughput_fps": 430, "tepe_bellek_mb": 12.0, "parametre_sayisi": 560000},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit_yolu = os.path.join(tmp_dir, "test_ablasyon_paneli.png")
        gorsellestirici.pano_olustur(ornek_sonuclar, kayit_yolu=kayit_yolu)
        assert os.path.exists(kayit_yolu)
        assert os.path.getsize(kayit_yolu) > 1000
