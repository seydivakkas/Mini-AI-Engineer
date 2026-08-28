"""
Model Kayıt Sistemi, Sürümleme ve Yaşam Döngüsü Birim Testleri
--------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import shutil
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.kayit_motoru import ModelKayitMotoru
from src.model import UretimVisionModeli
from src.kalite_kapisi import ModelKaliteKapisi


@pytest.fixture
def gecici_registry(tmp_path):
    depo_yolu = str(tmp_path / ".test_registry")
    motor = ModelKayitMotoru(depo_dizini=depo_yolu)
    yield motor, depo_yolu
    if os.path.exists(depo_yolu):
        shutil.rmtree(depo_yolu, ignore_errors=True)


def test_kayitli_model_olusturma(gecici_registry):
    motor, _ = gecici_registry
    motor.model_olustur_veya_getir("VisionClassifier", "Test Modeli")
    imlec = motor.baglanti.cursor()
    imlec.execute("SELECT model_adi, aciklama FROM kayitli_modeller WHERE model_adi = 'VisionClassifier'")
    satir = imlec.fetchone()
    assert satir is not None
    assert satir[0] == "VisionClassifier"


def test_surum_ekleme_ve_artan_numara(gecici_registry, tmp_path):
    motor, _ = gecici_registry
    gecici_pt = tmp_path / "dummy.pt"
    torch.save({"w": torch.zeros(2)}, gecici_pt)

    v1 = motor.surum_ekle("TestModel", str(gecici_pt), "run_1")
    v2 = motor.surum_ekle("TestModel", str(gecici_pt), "run_2")
    v3 = motor.surum_ekle("TestModel", str(gecici_pt), "run_3")

    assert v1 == 1
    assert v2 == 2
    assert v3 == 3


def test_asama_gecisi_none_to_staging(gecici_registry, tmp_path):
    motor, _ = gecici_registry
    gecici_pt = tmp_path / "dummy.pt"
    torch.save({"w": torch.zeros(2)}, gecici_pt)

    v1 = motor.surum_ekle("TestModel", str(gecici_pt))
    motor.asama_degistir("TestModel", v1, "STAGING", aciklama="Test aşamasına alındı")

    surumler = motor.tum_surumleri_listele("TestModel")
    assert surumler[0]["asama"] == "STAGING"


def test_asama_gecisi_staging_to_production_otomatik_arsivleme(gecici_registry, tmp_path):
    motor, _ = gecici_registry
    gecici_pt = tmp_path / "dummy.pt"
    torch.save({"w": torch.zeros(2)}, gecici_pt)

    v1 = motor.surum_ekle("TestModel", str(gecici_pt))
    v2 = motor.surum_ekle("TestModel", str(gecici_pt))

    motor.asama_degistir("TestModel", v1, "PRODUCTION")
    assert motor.uretim_modelini_getir("TestModel")["surum_no"] == 1

    motor.asama_degistir("TestModel", v2, "PRODUCTION", mevcut_uretimi_arsivle=True)
    assert motor.uretim_modelini_getir("TestModel")["surum_no"] == 2

    surumler = motor.tum_surumleri_listele("TestModel")
    assert surumler[0]["asama"] == "ARCHIVED"  # v1 arşivlendi
    assert surumler[1]["asama"] == "PRODUCTION"  # v2 üretimde


def test_uretim_modelini_getirme(gecici_registry, tmp_path):
    motor, _ = gecici_registry
    gecici_pt = tmp_path / "dummy.pt"
    torch.save({"w": torch.zeros(2)}, gecici_pt)

    assert motor.uretim_modelini_getir("BilinmeyenModel") is None

    v1 = motor.surum_ekle("M1", str(gecici_pt), metrikler={"acc": 95.0})
    motor.asama_degistir("M1", v1, "PRODUCTION")

    prod = motor.uretim_modelini_getir("M1")
    assert prod is not None
    assert prod["surum_no"] == 1
    assert prod["metrikler"]["acc"] == 95.0


def test_acil_geri_alma_rollback(gecici_registry, tmp_path):
    motor, _ = gecici_registry
    gecici_pt = tmp_path / "dummy.pt"
    torch.save({"w": torch.zeros(2)}, gecici_pt)

    v1 = motor.surum_ekle("RollbackModel", str(gecici_pt))
    v2 = motor.surum_ekle("RollbackModel", str(gecici_pt))

    motor.asama_degistir("RollbackModel", v1, "PRODUCTION")
    motor.asama_degistir("RollbackModel", v2, "PRODUCTION", mevcut_uretimi_arsivle=True)

    # Şu anda v2 Production'da
    assert motor.uretim_modelini_getir("RollbackModel")["surum_no"] == 2

    # Acil Rollback yap -> v1 tekrar Production olmalı
    aktif = motor.geri_al("RollbackModel", aciklama="v2 hatası nedeniyle geri alındı")
    assert aktif["surum_no"] == 1
    assert aktif["asama"] == "PRODUCTION"


def test_kalite_kapisi_onay_ve_ret():
    kalite_kapisi = ModelKaliteKapisi(min_dogruluk=80.0, max_gecikme_ms=50.0, max_ece=0.20)
    model = UretimVisionModeli(sinif_sayisi=2, taban_kanal=8)

    x = torch.randn(10, 3, 32, 32)
    y = torch.randint(0, 2, (10,))
    loader = DataLoader(TensorDataset(x, y), batch_size=5)

    rapor = kalite_kapisi.degerlendir(model, loader, cihaz="cpu")
    assert "gecti_mi" in rapor
    assert "metrikler" in rapor
    assert "val_acc" in rapor["metrikler"]
    assert "latency_ms" in rapor["metrikler"]


def test_model_sema_dogrulama():
    model = UretimVisionModeli(giris_kanali=3, sinif_sayisi=10)
    x_gecerli = torch.randn(2, 3, 32, 32)
    x_gecersiz_kanal = torch.randn(2, 1, 32, 32)
    x_gecersiz_boyut = torch.randn(2, 3, 32)

    assert model(x_gecerli).shape == (2, 10)

    with pytest.raises(ValueError):
        _ = model(x_gecersiz_kanal)

    with pytest.raises(ValueError):
        _ = model(x_gecersiz_boyut)
