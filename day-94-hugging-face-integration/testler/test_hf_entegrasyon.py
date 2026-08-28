"""
Day 94: Hugging Face Entegrasyonu ve Model Paketleme Birim Testleri
------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import tempfile
import pytest
import torch
from transformers import AutoConfig, AutoModelForImageClassification

from src.konfigurasyon import MiniViTConfig
from src.model import MiniViTForImageClassification
from src.hub_yoneticisi import HubPaketleyici
from src.gorsellestirici import HubGorsellestirici


def test_minivit_config_olusturma_ve_serilestirme():
    """MiniViTConfig'in parametrelerini ve JSON serileştirmesini test eder."""
    config = MiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        gizli_boyut=64,
        katman_sayisi=2,
        dikkat_baslik_sayisi=2,
        sinif_sayisi=5,
    )
    assert config.model_type == "minivit"
    assert config.goruntu_boyutu == 32
    assert config.sinif_sayisi == 5

    c_dict = config.to_dict()
    assert c_dict["model_type"] == "minivit"
    assert c_dict["gizli_boyut"] == 64

    yeniden_config = MiniViTConfig.from_dict(c_dict)
    assert yeniden_config.gizli_boyut == 64


def test_minivit_model_ileri_gecis():
    """MiniViTForImageClassification modelinin ileri geçişini ve logit boyutlarını test eder."""
    config = MiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        gizli_boyut=64,
        katman_sayisi=2,
        dikkat_baslik_sayisi=2,
        sinif_sayisi=10,
    )
    model = MiniViTForImageClassification(config)
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        cikti = model(pixel_values=x)

    assert cikti.logits.shape == (2, 10)
    assert cikti.loss is None


def test_minivit_loss_hesaplama():
    """Modelin etiketler sağlandığında CrossEntropy kaybı hesapladığını test eder."""
    config = MiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        gizli_boyut=64,
        katman_sayisi=2,
        dikkat_baslik_sayisi=2,
        sinif_sayisi=4,
    )
    model = MiniViTForImageClassification(config)

    x = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])

    cikti = model(pixel_values=x, labels=labels)
    assert cikti.loss is not None
    assert cikti.loss.item() > 0.0


def test_safetensors_kaydetme_ve_yukleme():
    """SafeTensors formatında diske kaydetme ve from_pretrained ile yüklemeyi test eder."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = MiniViTConfig(
            goruntu_boyutu=32,
            yama_boyutu=4,
            gizli_boyut=32,
            katman_sayisi=2,
            dikkat_baslik_sayisi=2,
            sinif_sayisi=3,
        )
        model = MiniViTForImageClassification(config)
        model.save_pretrained(tmp_dir, safe_serialization=True)

        safetensor_dosyasi = os.path.join(tmp_dir, "model.safetensors")
        config_dosyasi = os.path.join(tmp_dir, "config.json")
        assert os.path.exists(safetensor_dosyasi)
        assert os.path.exists(config_dosyasi)

        yuklenen_model = MiniViTForImageClassification.from_pretrained(tmp_dir)
        assert isinstance(yuklenen_model, MiniViTForImageClassification)


def test_autoclass_entegrasyonu():
    """AutoConfig ve AutoModelForImageClassification ile model yüklemeyi test eder."""
    paketleyici = HubPaketleyici()

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = MiniViTConfig(
            goruntu_boyutu=32,
            yama_boyutu=4,
            gizli_boyut=32,
            katman_sayisi=2,
            dikkat_baslik_sayisi=2,
            sinif_sayisi=3,
        )
        model = MiniViTForImageClassification(config)
        model.save_pretrained(tmp_dir, safe_serialization=True)

        auto_config = AutoConfig.from_pretrained(tmp_dir)
        assert auto_config.model_type == "minivit"

        auto_model = AutoModelForImageClassification.from_pretrained(tmp_dir)
        assert isinstance(auto_model, MiniViTForImageClassification)


def test_sayisal_ve_tensor_birebirligi():
    """Orijinal model ile SafeTensors'tan yüklenen modelin çıktılarının birebir (1e-5) eşitliğini test eder."""
    paketleyici = HubPaketleyici()

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = MiniViTConfig(
            goruntu_boyutu=32,
            yama_boyutu=4,
            gizli_boyut=48,
            katman_sayisi=2,
            dikkat_baslik_sayisi=3,
            sinif_sayisi=5,
        )
        model = MiniViTForImageClassification(config)
        # Paketi diske kaydet ve doğrula
        paketleyici.modeli_paketle_ve_kaydet(model, tmp_dir)
        uyumlu, maks_fark = paketleyici.sayisal_uyumluluk_dogrula(model, tmp_dir)

        assert uyumlu is True
        assert maks_fark < 1e-5


def test_hub_paketleyici_tam_is_akisi():
    """HubPaketleyici'nin tüm yardımcı dosyaları ve ModelPaketBilgisi'ni eksiksiz ürettiğini test eder."""
    paketleyici = HubPaketleyici()

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = MiniViTConfig(
            goruntu_boyutu=32,
            yama_boyutu=4,
            gizli_boyut=32,
            katman_sayisi=2,
            dikkat_baslik_sayisi=2,
            sinif_sayisi=4,
        )
        model = MiniViTForImageClassification(config)

        bilgi = paketleyici.modeli_paketle_ve_kaydet(model, tmp_dir, repo_adi="test/minivit-v1")

        assert os.path.exists(bilgi.config_dosyasi)
        assert os.path.exists(bilgi.safetensors_dosyasi)
        assert os.path.exists(bilgi.preprocessor_dosyasi)
        assert os.path.exists(bilgi.hub_model_card_dosyasi)
        assert bilgi.sayisal_uyumluluk_dogrulandi is True
        assert bilgi.toplam_parametre > 0


def test_gorsellestirici_pano_uretme():
    """6 panelli teşhis panosunun hatasız oluşturulup diske kaydedildiğini test eder."""
    gorsellestirici = HubGorsellestirici(cizim_boyutu=(12, 8), dpi=100)
    paketleyici = HubPaketleyici()

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = MiniViTConfig(
            goruntu_boyutu=32,
            yama_boyutu=4,
            gizli_boyut=32,
            katman_sayisi=2,
            dikkat_baslik_sayisi=2,
            sinif_sayisi=4,
        )
        model = MiniViTForImageClassification(config)
        bilgi = paketleyici.modeli_paketle_ve_kaydet(model, tmp_dir)

        cikti_png = os.path.join(tmp_dir, "test_pano.png")
        gorsellestirici.olustur_hf_entegrasyon_paneli(
            paket_bilgisi=bilgi,
            model=model,
            cikarim_sureleri=[1.2, 1.5, 1.1, 1.3],
            kayit_yolu=cikti_png,
        )

        assert os.path.exists(cikti_png)
        assert os.path.getsize(cikti_png) > 1000
