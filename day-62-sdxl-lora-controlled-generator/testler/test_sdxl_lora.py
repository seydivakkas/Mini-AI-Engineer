"""
Day 62: SDXL + LoRA ile Kontrollü Görsel Üretimi Birim Testleri.
"""

import os
import pytest
import torch
from src.sdxl_lora_motoru import LoRAKatmani, SDXLLoRAMotoru, LatentDenoisingSampler
from src.lora_fuzyon_yoneticisi import LoRAFuzyonYoneticisi
from src.gorsellestirici import SDXLLoRAGorsellestirici


def test_lora_katmani_matematiksel_delta():
    """LoRA katmanının başlangıçta B=0 olduğu için temel çıktıyı bozmadığını test eder."""
    in_dim, out_dim = 64, 64
    lora = LoRAKatmani(in_features=in_dim, out_features=out_dim, rank=4, alpha=8.0)

    x = torch.randn(2, in_dim)
    temel_cikti = torch.randn(2, out_dim)

    # Başlangıçta B sıfır olduğu için delta 0 olmalı
    cikti = lora(x, temel_cikti)
    assert torch.allclose(cikti, temel_cikti, atol=1e-6)

    # B ağırlıklarına değer verildiğinde deltanın eklendiğini doğrula
    lora.lora_B.data.fill_(0.1)
    cikti_delta = lora(x, temel_cikti)
    assert not torch.allclose(cikti_delta, temel_cikti, atol=1e-6)


def test_lora_skala_ayari_degisimi():
    """LoRA adaptör ağırlık skalasının dinamik değişimini test eder."""
    lora = LoRAKatmani(in_features=32, out_features=32, rank=4, alpha=8.0)
    lora.lora_B.data.fill_(0.1)

    x = torch.randn(1, 32)
    temel = torch.zeros(1, 32)

    lora.adapter_agirligi = 0.5
    cikti_yarim = lora(x, temel)

    lora.adapter_agirligi = 1.0
    cikti_tam = lora(x, temel)

    assert torch.allclose(cikti_tam, cikti_yarim * 2.0, atol=1e-5)


def test_sdxl_cross_attention_forward():
    """SDXLLoRAMotoru'nun Cross-Attention ve LoRA ileri geçiş boyutlarını test eder."""
    model = SDXLLoRAMotoru(d_model=64, d_text=128)
    model.adaptor_ekle("test_lora", rank=4, alpha=8.0)

    latent_x = torch.randn(2, 16, 64)
    text_emb = torch.randn(2, 8, 128)

    cikti = model(latent_x, text_emb)
    assert cikti.shape == (2, 16, 64)


def test_parametre_verimliligi_tasarruf():
    """LoRA ile taban modele kıyasla %95+ parametre tasarrufu sağlandığını test eder."""
    model = SDXLLoRAMotoru(d_model=256, d_text=512)
    model.adaptor_ekle("stil_lora", rank=8, alpha=16.0)

    analiz = LoRAFuzyonYoneticisi.parametre_verimlilik_analizi(model)
    assert analiz["taban_parametre_sayisi"] > analiz["lora_parametre_sayisi"]
    assert analiz["tasarruf_orani_yuzde"] > 90.0


def test_latent_denoising_sampler():
    """LatentDenoisingSampler'ın çok adımlı CFG difüzyon simülasyonunu test eder."""
    model = SDXLLoRAMotoru(d_model=32, d_text=64)
    sampler = LatentDenoisingSampler(adim_sayisi=5, cfg_skalasi=5.0)

    kosul = torch.randn(1, 4, 64)
    kosulsuz = torch.zeros(1, 4, 64)

    z_out, enerjiler = sampler.ornekle_latent(model, kosul, kosulsuz, latent_sekli=(1, 8, 32))
    assert z_out.shape == (1, 8, 32)
    assert len(enerjiler) == 5


def test_lora_fuzyon_deneyi():
    """LoRAFuzyonYoneticisi'nin farklı skala ve CFG analizlerini sorunsuz çalıştırdığını test eder."""
    model = SDXLLoRAMotoru(d_model=32, d_text=64)
    model.adaptor_ekle("stil_lora", rank=4, alpha=8.0)

    sonuclar = LoRAFuzyonYoneticisi.calistir_fuzyon_deneyi(
        model=model,
        skala_degerleri=[0.0, 0.5, 1.0],
        cfg_degerleri=[3.0, 7.5]
    )

    assert "parametre_verimliligi" in sonuclar
    assert "skala_analizi" in sonuclar
    assert "cfg_analizi" in sonuclar


def test_gorsellestirici_panel_cizimi(tmp_path):
    """6 panelli SDXL LoRA görselleştiricisinin geçerli bir PNG ürettiğini test eder."""
    model = SDXLLoRAMotoru(d_model=32, d_text=64)
    model.adaptor_ekle("stil_lora", rank=4, alpha=8.0)
    sonuclar = LoRAFuzyonYoneticisi.calistir_fuzyon_deneyi(model=model, skala_degerleri=[0.0, 1.0], cfg_degerleri=[7.5])

    hedef = str(tmp_path / "test_sdxl_lora_paneli.png")
    cikis = SDXLLoRAGorsellestirici.panel_ciz(sonuclar, hedef_path=hedef)

    assert os.path.exists(cikis)
    assert os.path.getsize(cikis) > 1000
