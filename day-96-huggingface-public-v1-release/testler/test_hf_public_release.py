"""
Day 96: MiniViT v1.0 Hugging Face Canlı Dağıtım ve Demo Test Paketi (PyTest).
8 adet kapsamlı birim ve entegrasyon testi içerir.
"""

import os
import shutil
import tempfile
import pytest
import torch
import numpy as np
from PIL import Image
import gradio as gr

from src.konfigurasyon import MiniViTConfig
from src.model import MiniViTForImageClassification
from src.dagitim_yoneticisi import HfDagitimYoneticisi, MiniViTPipeline
from src.canli_demo import GradioDemoOlusturucu
from src.gorsellestirici import PublicReleaseGorsellestirici


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def minivit_model():
    torch.manual_seed(42)
    config = MiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        kanal_sayisi=3,
        gizli_boyut=64,
        katman_sayisi=2,
        dikkat_baslik_sayisi=2,
        ileri_besleme_boyutu=128,
        sinif_sayisi=10,
    )
    model = MiniViTForImageClassification(config)
    model.eval()
    return model


def test_minivit_model_olusturma_ve_cikti_boyutu(minivit_model):
    """MiniViT modelinin başlatılması ve çıktı tensör boyutunun (Batch, Sınıf) doğrulanması."""
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = minivit_model(x)

    assert out.logits.shape == (2, 10)
    assert not torch.isnan(out.logits).any()


def test_dagitim_yoneticisi_model_kaydetme(minivit_model, temp_dir):
    """Modelin diske SafeTensors formatında ve Hub dosyalarıyla kaydedilmesi."""
    dagitimci = HfDagitimYoneticisi()
    bilgi = dagitimci.modeli_hazirla_ve_kaydet(
        model=minivit_model,
        kayit_dizini=temp_dir,
        repo_id="test/minivit-v1",
    )

    assert os.path.exists(os.path.join(temp_dir, "model.safetensors"))
    assert os.path.exists(os.path.join(temp_dir, "config.json"))
    assert os.path.exists(os.path.join(temp_dir, "preprocessor_config.json"))
    assert os.path.exists(os.path.join(temp_dir, "README.md"))
    assert bilgi["toplam_boyut_kb"] > 0


def test_autoclass_ile_yukleme_ve_dogrulama(minivit_model, temp_dir):
    """Kaydedilen paketin AutoConfig ve AutoModelForImageClassification ile geri yüklenmesi."""
    dagitimci = HfDagitimYoneticisi()
    dagitimci.modeli_hazirla_ve_kaydet(
        model=minivit_model,
        kayit_dizini=temp_dir,
        repo_id="test/minivit-v1",
    )

    pipeline = dagitimci.yukle_ve_pipeline_kur(temp_dir)
    assert isinstance(pipeline, MiniViTPipeline)
    assert pipeline.config.model_type == "minivit"


def test_pipeline_pil_girdi_on_isleme(minivit_model):
    """PIL, NumPy ve Tensor girdilerinin doğru tensör formatına dönüştürülmesi."""
    pipe = MiniViTPipeline(minivit_model, minivit_model.config)

    # 1. PIL Image
    pil_img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    t_pil = pipe.goruntu_on_isle(pil_img)
    assert t_pil.shape == (1, 3, 32, 32)

    # 2. NumPy Array
    np_img = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
    t_np = pipe.goruntu_on_isle(np_img)
    assert t_np.shape == (1, 3, 32, 32)

    # 3. Direct Tensor
    t_in = torch.randn(1, 3, 32, 32)
    t_out = pipe.goruntu_on_isle(t_in)
    assert t_out.shape == (1, 3, 32, 32)


def test_pipeline_siniflandirma_ve_topk(minivit_model):
    """Pipeline'ın Top-K olasılıkları ve etiketleri doğru döndürmesi."""
    pipe = MiniViTPipeline(minivit_model, minivit_model.config)
    img = Image.new("RGB", (32, 32), color="red")

    sonuclar = pipe(img, top_k=3)
    assert len(sonuclar) == 3
    assert "label" in sonuclar[0]
    assert "score" in sonuclar[0]
    assert "gecikme_ms" in sonuclar[0]
    assert sonuclar[0]["score"] >= sonuclar[1]["score"] >= sonuclar[2]["score"]


def test_gradio_demo_siniflandir_fonksiyonu(minivit_model):
    """Gradio demo sınıflandır fonksiyonunun sözlük ve gecikme metni üretmesi."""
    pipe = MiniViTPipeline(minivit_model, minivit_model.config)
    demo_olusturucu = GradioDemoOlusturucu(pipe)

    img = Image.new("RGB", (32, 32), color="blue")
    olasiliklar, gecikme_str = demo_olusturucu.siniflandir(img)

    assert isinstance(olasiliklar, dict)
    assert len(olasiliklar) == 5
    assert "Çıkarım Gecikmesi" in gecikme_str


def test_gradio_arayuz_olusturma(minivit_model):
    """Gradio demo arayüz blok nesnesinin hatasız inşa edilmesi."""
    pipe = MiniViTPipeline(minivit_model, minivit_model.config)
    demo_olusturucu = GradioDemoOlusturucu(pipe)
    arayuz = demo_olusturucu.arayuz_olustur()

    assert isinstance(arayuz, gr.Blocks)


def test_gorsellestirici_pano_uretme(temp_dir):
    """6-panelli teşhis panosunun oluşturulması ve dosya boyutunun doğrulanması."""
    gorsellestirici = PublicReleaseGorsellestirici(dpi=100)
    pano_yolu = os.path.join(temp_dir, "test_pano.png")

    dagitim_bilgisi = {
        "repo_id": "seydivakkas/minivit-cifar10-v1",
        "toplam_boyut_kb": 2140.0,
    }
    ornek_tahminler = [
        {"label": "uçak", "score": 0.85},
        {"label": "kuş", "score": 0.08},
        {"label": "gemi", "score": 0.04},
        {"label": "otomobil", "score": 0.02},
        {"label": "kedi", "score": 0.01},
    ]
    gecikme_istatistikleri = {
        "ham_gecikme_ms": 2.1,
        "pipeline_gecikme_ms": 2.8,
    }

    gorsellestirici.pano_olustur(
        dagitim_bilgisi=dagitim_bilgisi,
        ornek_tahminler=ornek_tahminler,
        gecikme_istatistikleri=gecikme_istatistikleri,
        kayit_yolu=pano_yolu,
    )

    assert os.path.exists(pano_yolu)
    assert os.path.getsize(pano_yolu) > 1000
