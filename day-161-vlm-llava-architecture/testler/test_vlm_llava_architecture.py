"""
GÜN 161: LLaVA VLM Mimarisi Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.vit_goruntu_kodlayici import ViTGoruntuKodlayici
from src.mlp_projektor import MLPProjektor
from src.llava_vlm_modeli import LLaVAVLMModeli, BasitLLMKodlayici
from src.gorsellestirici import VLMGorsellestirici


def test_vit_goruntu_kodlayici_cikti_boyutu():
    """ViT modelinin 224x224 görüntüyü 256 patch tokenına (768d) dönüştürdüğünü test eder."""
    vit = ViTGoruntuKodlayici(goruntu_boyutu=224, patch_boyutu=14, d_vision=768, katman_sayisi=2)
    dummy_img = torch.randn(2, 3, 224, 224)
    tokens = vit(dummy_img)

    assert tokens.shape == (2, 256, 768)


def test_mlp_projektor_hizalama():
    """MLP Projektörün 768d görsel tokenları 512d LLM uzayına yansıttığını test eder."""
    mlp = MLPProjektor(d_vision=768, d_text=512)
    dummy_visual = torch.randn(2, 256, 768)
    projected = mlp(dummy_visual)

    assert projected.shape == (2, 256, 512)


def test_basit_llm_ileri_gecis():
    """Decoder LLM'in füzyon embedding'lerinden doğru logit boyutu ürettiğini test eder."""
    llm = BasitLLMKodlayici(vocab_size=500, d_text=512, katman_sayisi=2)
    dummy_fused = torch.randn(2, 266, 512)
    logits = llm(dummy_fused)

    assert logits.shape == (2, 266, 500)


def test_llava_vlm_uc_uca_ileri_gecis():
    """LLaVA VLM modelinin görüntü ve metin tokenlarını birleştirip uçtan uca logit ürettiğini test eder."""
    vlm = LLaVAVLMModeli(goruntu_boyutu=224, patch_boyutu=14, d_vision=768, d_text=512, vocab_size=500)
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_text_ids = torch.randint(0, 500, (1, 10))

    logits = vlm(dummy_img, dummy_text_ids)

    # Toplam dizi: 256 patch + 10 text = 266 token
    assert logits.shape == (1, 266, 500)


def test_llava_vlm_farkli_batch_boyutu():
    """Modelin değişken batch boyutlarında (Batch=3) sorunsuz çalıştığını test eder."""
    vlm = LLaVAVLMModeli(goruntu_boyutu=224, patch_boyutu=14, d_vision=768, d_text=512, vocab_size=500)
    dummy_img = torch.randn(3, 3, 224, 224)
    dummy_text_ids = torch.randint(0, 500, (3, 15))

    logits = vlm(dummy_img, dummy_text_ids)
    assert logits.shape == (3, 271, 500)


def test_vit_patch_sayisi_hesabi():
    """Patch boyutu değiştirildiğinde (örn: 28x28) patch sayısının doğru hesaplandığını test eder."""
    vit = ViTGoruntuKodlayici(goruntu_boyutu=224, patch_boyutu=28, d_vision=512, katman_sayisi=1)
    dummy_img = torch.randn(1, 3, 224, 224)
    tokens = vit(dummy_img)

    # (224/28)^2 = 8^2 = 64 patch
    assert tokens.shape == (1, 64, 512)


def test_mlp_projektor_gradyan_akisi():
    """MLP projektörün parametrelerinin eğitilebilir ve gradyan alır olduğunu test eder."""
    mlp = MLPProjektor(d_vision=128, d_text=64)
    dummy_in = torch.randn(2, 10, 128, requires_grad=True)
    out = mlp(dummy_in)
    loss = out.sum()
    loss.backward()

    assert dummy_in.grad is not None


def test_gorsellestirici_pano_uretme():
    """6 panelli LLaVA VLM teşhis panosunun PNG olarak kaydedildiğini test eder."""
    bilgi = {
        "visual_token_sayisi": 256,
        "text_token_sayisi": 10,
        "d_text": 512,
        "goruntu_aciklamasi": "Sentetik Kırmızı Elma ve Yeşil Masa",
        "kullanici_sorusu": "Masanın üzerindeki meyve nedir?",
        "model_yaniti": "Masanın üzerinde taze kırmızı bir elma bulunmaktadır.",
    }
    gorsellestirici = VLMGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_vlm_pano.png")
        gorsellestirici.pano_olustur(bilgi, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
