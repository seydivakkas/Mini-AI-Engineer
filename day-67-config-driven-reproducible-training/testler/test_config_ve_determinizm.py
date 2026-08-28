"""
Day 67: Konfigürasyon Yönetimi ve Deterministik Eğitim Birim Test Paketi
=======================================================================
Pydantic v2 şema doğrulaması, YAML override mekanizması, rastgelelik tohumlama,
model mimarisi ve bit-for-bit tekrarlanabilirlik motorunun test edilmesi.
"""

import os
import tempfile
import pytest
import numpy as np
import torch
import yaml

from src.konfigurasyon_semasi import KokKonfigurasyon, VeriKonfigurasyonu, ModelKonfigurasyonu
from src.konfigurasyon_yoneticisi import KonfigurasyonYoneticisi
from src.determinizm_motoru import DeterminizmYoneticisi
from src.model_mimari import ModulerVisionNet
from src.egitim_motoru import TekrarlanabilirEgitici
from src.deney_dogrulayici import DeterminizmDogrulayici
from src.gorsellestirici import DeterminizmGorsellestirici


@pytest.fixture
def ornek_yaml_yolu():
    proje_kok = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(proje_kok, "konfigurasyonlar", "varsayilan_egitim.yaml")


@pytest.fixture
def gecici_dizin():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_varsayilan_yaml_yukleme_ve_dogrulama(ornek_yaml_yolu: str):
    """Varsayılan YAML dosyasının Pydantic şemasıyla başarıyla yüklendiğini test eder."""
    config = KonfigurasyonYoneticisi.yaml_yukle(ornek_yaml_yolu)
    assert isinstance(config, KokKonfigurasyon)
    assert config.model.mimari_adi == "ModulerVisionNet"
    assert config.veri.batch_size > 0
    assert config.egitim.tohum == 42


def test_override_mekanizmasi(ornek_yaml_yolu: str):
    """Noktasal parametre override mekanizmasının alt sözlükleri doğru güncellediğini test eder."""
    overrides = [
        "egitim.tohum=999",
        "model.taban_kanal=64",
        "optimizer.lr=0.005",
        "egitim.deterministik_mod=false"
    ]
    config = KonfigurasyonYoneticisi.yaml_yukle(ornek_yaml_yolu, override_listesi=overrides)
    assert config.egitim.tohum == 999
    assert config.model.taban_kanal == 64
    assert config.optimizer.lr == 0.005
    assert config.egitim.deterministik_mod is False


def test_gecersiz_konfigurasyon_hatalari():
    """Geçersiz kanal uyumsuzluğu veya negatif batch_size'da Pydantic'in hata fırlattığını test eder."""
    with pytest.raises(ValueError):
        # Kanal uyumsuzluğu: Model 1 kanal beklerken Veri 3 kanal veriyor
        KokKonfigurasyon(
            model=ModelKonfigurasyonu(girdi_kanali=1),
            veri=VeriKonfigurasyonu(girdi_boyutu=[3, 32, 32])
        )

    with pytest.raises(ValueError):
        VeriKonfigurasyonu(batch_size=-10)


def test_determinizm_tohum_sabitleme():
    """Tohum kilitlendiğinde NumPy ve PyTorch'un birebir aynı rastgele sayı dizisini ürettiğini test eder."""
    DeterminizmYoneticisi.tohum_sabitle(42)
    np_1 = np.random.randn(5)
    t_1 = torch.randn(5)

    DeterminizmYoneticisi.tohum_sabitle(42)
    np_2 = np.random.randn(5)
    t_2 = torch.randn(5)

    assert np.allclose(np_1, np_2)
    assert torch.equal(t_1, t_2)


def test_moduler_vision_net_ileri_yayilim():
    """Konfigürasyondan türetilen modelin doğru tensör boyutları ürettiğini test eder."""
    cfg = ModelKonfigurasyonu(girdi_kanali=3, sinif_sayisi=4, taban_kanal=16)
    model = ModulerVisionNet(cfg)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 4)
    assert len(model.agirlik_hashi_al()) == 64  # SHA256 uzunluğu


def test_tekrarlanabilir_egitici_dongusu(ornek_yaml_yolu: str):
    """Eğitim motorunun belirtilen epoch sayısı kadar sorunsuz koştuğunu test eder."""
    config = KonfigurasyonYoneticisi.yaml_yukle(ornek_yaml_yolu, override_listesi=["egitim.epoch_sayisi=2", "veri.ornek_sayisi=50"])
    egitici = TekrarlanabilirEgitici(config)
    sonuclar = egitici.egit()

    assert len(sonuclar["gecmis"]["train_loss"]) == 2
    assert len(sonuclar["gecmis"]["val_accuracy"]) == 2
    assert sonuclar["son_val_accuracy"] >= 0.0


def test_determinizm_dogrulayici_tam_eslesme(ornek_yaml_yolu: str):
    """Run A ve Run B'nin sıfır kayıp farkı ve birebir aynı ağırlık hash'ine sahip olduğunu test eder."""
    config = KonfigurasyonYoneticisi.yaml_yukle(ornek_yaml_yolu, override_listesi=["egitim.epoch_sayisi=3", "veri.ornek_sayisi=100"])
    dogrulama = DeterminizmDogrulayici.determinizm_testi_kos(config, farkli_tohum_baseline=99)

    assert dogrulama["deterministik_basarili"] is True
    assert dogrulama["maks_train_loss_delta_ab"] == 0.0
    assert dogrulama["maks_val_loss_delta_ab"] == 0.0
    assert dogrulama["run_a_hash"] == dogrulama["run_b_hash"]
    assert dogrulama["run_a_hash"] != dogrulama["run_c_hash"]


def test_determinizm_gorsellestirici(ornek_yaml_yolu: str, gecici_dizin: str):
    """6 Panelli görsel teşhis panosunun kaydedildiğini test eder."""
    panel_yolu = os.path.join(gecici_dizin, "test_panel.png")
    config = KonfigurasyonYoneticisi.yaml_yukle(ornek_yaml_yolu, override_listesi=["egitim.epoch_sayisi=2", "veri.ornek_sayisi=50"])
    dogrulama = DeterminizmDogrulayici.determinizm_testi_kos(config)

    cizim_yolu = DeterminizmGorsellestirici.panoyu_ciz_ve_kaydet(
        dogrulama_sonuclari=dogrulama,
        cikti_yolu=panel_yolu
    )

    assert os.path.exists(cizim_yolu)
    assert os.path.getsize(cizim_yolu) > 10000
