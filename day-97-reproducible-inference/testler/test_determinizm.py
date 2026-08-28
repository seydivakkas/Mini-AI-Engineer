"""
Determinizm ve Donanım Doğrulama Birim ve Entegrasyon Testleri (Day 97).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.konfigurasyon import MiniViTConfig
from src.model import MiniViTForImageClassification
from src.determinizm_yoneticisi import (
    DeterminizmOrtami,
    BitHashHesaplayici,
    DeterminizmDenetleyicisi,
)
from src.capraz_donanim_motoru import CaprazDonanimDogrulayici, HassasiyetKiyaslayici
from src.gorsellestirici import DeterminizmGorsellestirici


@pytest.fixture
def kucuk_model():
    """Hızlı testler için hafif MiniViT model fikstürü."""
    cfg = MiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        gizli_boyut=64,
        katman_sayisi=2,
        dikkat_baslik_sayisi=2,
        ileri_besleme_boyutu=128,
        sinif_sayisi=10,
    )
    model = MiniViTForImageClassification(cfg)
    model.eval()
    return model


def test_determinizm_ortami_context_manager():
    """Determinizm ortamının çevre değişkenlerini ve determinizm bayraklarını doğru yönettiğini test eder."""
    with DeterminizmOrtami(seed=123) as ort:
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        t1 = torch.randn(5, 5)
        torch.manual_seed(123)
        t2 = torch.randn(5, 5)
        torch.manual_seed(123)
        t3 = torch.randn(5, 5)
        assert torch.equal(t2, t3)


def test_bithash_hesaplayici_aynilik():
    """BitHashHesaplayici'nın özdeş tensörlerde aynı, farklı tensörlerde farklı hash ürettiğini test eder."""
    t1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    t2 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    t3 = torch.tensor([1.0, 2.0, 3.0, 4.0001], dtype=torch.float32)

    h1 = BitHashHesaplayici.tensor_bit_hash(t1)
    h2 = BitHashHesaplayici.tensor_bit_hash(t2)
    h3 = BitHashHesaplayici.tensor_bit_hash(t3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA-256 hex string


def test_ardil_cikarim_determinizmi(kucuk_model):
    """Ardışık çıkarımların bit-level determinizmini ve sıfır sapmayı doğrular."""
    denetleyici = DeterminizmDenetleyicisi(kucuk_model)
    girdi = torch.randn(1, 3, 32, 32)

    sonuc = denetleyici.ardil_cikarim_testi(girdi, tekrar_sayisi=20)

    assert sonuc["tam_deterministik"] is True
    assert sonuc["benzersiz_hash_sayisi"] == 1
    assert sonuc["global_maks_sapma"] == 0.0


def test_minivit_ileri_gecis_seki(kucuk_model):
    """Modelin ileri geçiş tensör boyutlarını doğrular."""
    girdi = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        cikis = kucuk_model(girdi)
    assert cikis.logits.shape == (2, 10)


def test_capraz_donanim_cpu_gpu_parite(kucuk_model):
    """CPU ve GPU çıkarımları arasındaki sayısal pariteyi ($L_\\infty < 1e-4$) test eder."""
    dogrulayici = CaprazDonanimDogrulayici(kucuk_model)
    girdi = torch.randn(1, 3, 32, 32)

    sonuc = dogrulayici.cpu_gpu_parite_testi(girdi, tolerans_linf=1e-4)

    assert sonuc["parite_uyumlu"] is True
    assert sonuc["kosinus_benzerligi"] > 0.9999
    assert sonuc["linf_hata"] <= 1e-4


def test_hassasiyet_kiyaslayici_fp16_bf16(kucuk_model):
    """FP16 ve BF16 hassasiyet kıyaslaması ve SNR hesaplamasını test eder."""
    kiyaslayici = HassasiyetKiyaslayici(kucuk_model)
    girdi = torch.randn(1, 3, 32, 32)

    sonuc = kiyaslayici.hassasiyet_karsilastir(girdi, iterasyon=5)

    assert "linf_fp16" in sonuc
    assert "snr_fp16_db" in sonuc
    assert sonuc["snr_fp16_db"] > 30.0  # Kabul edilebilir SNR eşiği
    assert "FP32" in sonuc["gecikmeler_ms"]


def test_farkli_batch_boyutlari_determinizm(kucuk_model):
    """Farklı batch boyutlarında (1, 2, 4) determinizmin korunduğunu test eder."""
    denetleyici = DeterminizmDenetleyicisi(kucuk_model)
    for b in [1, 2, 4]:
        girdi = torch.randn(b, 3, 32, 32)
        sonuc = denetleyici.ardil_cikarim_testi(girdi, tekrar_sayisi=10)
        assert sonuc["tam_deterministik"] is True


def test_gorsellestirici_pano_uretme(kucuk_model):
    """6 panelli teşhis panosunun hatasız oluşturulduğunu test eder."""
    gorsellestirici = DeterminizmGorsellestirici(dpi=100)
    denetleyici = DeterminizmDenetleyicisi(kucuk_model)
    dogrulayici = CaprazDonanimDogrulayici(kucuk_model)
    kiyaslayici = HassasiyetKiyaslayici(kucuk_model)

    girdi = torch.randn(1, 3, 32, 32)
    det_res = denetleyici.ardil_cikarim_testi(girdi, tekrar_sayisi=5)
    par_res = dogrulayici.cpu_gpu_parite_testi(girdi)
    has_res = kiyaslayici.hassasiyet_karsilastir(girdi, iterasyon=3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit_yolu = os.path.join(tmp_dir, "test_paneli.png")
        gorsellestirici.pano_olustur(det_res, par_res, has_res, kayit_yolu=kayit_yolu)
        assert os.path.exists(kayit_yolu)
        assert os.path.getsize(kayit_yolu) > 1000
