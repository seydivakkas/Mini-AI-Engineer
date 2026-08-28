"""Day 23 Birim Testleri: Transfer Öğrenme ve İnce Ayar (ResNet & EfficientNet)."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.model_secici import TransferModelSecici
from src.veri_hazirlayici import TransferVeriYoneticisi, ImageNetDataset
from src.egitici import TransferEgitici, TransferEgitimSonucu
from src.karsilastirici import TransferKarsilastirici
from src.gorsellestirici import TransferGorsellestirici


def test_resnet18_feature_extraction_yapisi():
    """ResNet18 Feature Extraction modunda omurganın dondurulduğunu ve başlığın eklendiğini doğrular."""
    model = TransferModelSecici.resnet18_olustur(num_classes=3, strateji="feature_extraction")
    params = TransferModelSecici.parametre_ozeti(model)

    assert params["total"] > 0
    assert params["frozen"] > 0
    assert params["trainable"] < params["total"]

    # İleri geçiş testi
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 3)


def test_resnet18_fine_tuning_ve_ayrisik_parametreler():
    """ResNet18 Fine-Tuning modunda son bloğun açıldığını ve LR gruplarının oluştuğunu doğrular."""
    model = TransferModelSecici.resnet18_olustur(num_classes=4, strateji="fine_tuning")
    params = TransferModelSecici.parametre_ozeti(model)

    assert params["trainable"] > 0
    assert params["frozen"] > 0

    gruplar = TransferModelSecici.ayrisik_parametre_gruplari(model, lr_omurga=1e-4, lr_baslik=1e-3)
    assert len(gruplar) == 2
    assert gruplar[0]["lr"] == 1e-4
    assert gruplar[1]["lr"] == 1e-3


def test_efficientnet_b0_yapisi():
    """EfficientNet-B0 mimarisinin doğru sınıflandırıcı başlıkla kurulduğunu doğrular."""
    model = TransferModelSecici.efficientnet_b0_olustur(num_classes=4, strateji="feature_extraction")
    params = TransferModelSecici.parametre_ozeti(model)

    assert params["frozen"] > 0
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 4)


def test_imagenet_dataset_ve_dataloader():
    """ImageNetDataset sınıfının z-score normalizasyonu uyguladığını doğrular."""
    gorseller = np.random.rand(10, 64, 64, 3).astype(np.float32)
    etiketler = np.random.randint(0, 4, size=10)

    ds = ImageNetDataset(gorseller, etiketler)
    assert len(ds) == 10

    x_item, y_item = ds[0]
    assert x_item.shape == (3, 64, 64)
    assert isinstance(x_item, torch.Tensor)
    assert isinstance(y_item, torch.Tensor)


def test_transfer_egitici_dongusu():
    """TransferEgitici sınıfının eğitim ve değerlendirmeyi hatasız tamamladığını test eder."""
    model = TransferModelSecici.resnet18_olustur(num_classes=4, strateji="feature_extraction")
    egitici = TransferEgitici(model, device=torch.device("cpu"))

    gorseller = np.random.rand(16, 64, 64, 3).astype(np.float32)
    etiketler = np.array([0, 1, 2, 3] * 4)
    loader = DataLoader(ImageNetDataset(gorseller, etiketler), batch_size=8)

    tarihce, _ = egitici.egit(loader, loader, epochs=1)
    assert "train_loss" in tarihce
    assert len(tarihce["train_loss"]) == 1

    sonuc = egitici.degerlendir(loader, tarihce, model_adi="ResNet18", strateji="FE", egitim_suresi_sn=0.5)
    assert isinstance(sonuc, TransferEgitimSonucu)
    assert 0.0 <= sonuc.test_dogruluk <= 1.0


def test_transfer_karsilastirici_deneyler():
    """TransferKarsilastirici motorunun tüm stratejileri koşturabildiğini doğrular."""
    yonetici = TransferVeriYoneticisi(hedef_boyut=(32, 32), random_state=42)
    X, y, _ = yonetici.sentetik_veri_seti_uret(sinif_basina_ornek=4)
    tr_l, val_l, te_l, _, _ = yonetici.veri_bol_ve_yukleyicileri_olustur(X, y, batch_size=8)

    karsilastirici = TransferKarsilastirici(device=torch.device("cpu"))
    sonuclar = karsilastirici.karsilastirmali_deney_kos(tr_l, val_l, te_l, num_classes=4, epochs=1)

    assert len(sonuclar) == 4
    for s in sonuclar:
        assert isinstance(s, TransferEgitimSonucu)


def test_gorsellestirici_rapor_cizimi(tmp_path):
    """Karşılaştırmalı rapor grafiğinin kaydedildiğini test eder."""
    sonuclar = [
        TransferEgitimSonucu(
            model_adi="ResNet18 (FE)",
            strateji="feature_extraction",
            train_kayiplari=[0.5],
            val_kayiplari=[0.4],
            train_dogruluklari=[0.8],
            val_dogruluklari=[0.85],
            test_kayip=0.3,
            test_dogruluk=0.9,
            f1_macro=0.88,
            precision_macro=0.89,
            recall_macro=0.88,
            karisiklik_matrisi=np.eye(4, dtype=int),
            y_test_gercek=np.array([0, 1, 2, 3]),
            y_test_tahmin=np.array([0, 1, 2, 3]),
            egitim_suresi_sn=1.0,
            ornek_basina_gecikme_ms=0.5,
            egitilebilir_parametre=66000,
            toplam_parametre=11000000,
        )
    ]
    siniflar = ["Vazo", "Kumaş", "Rozet", "Ahşap"]
    hedef = tmp_path / "test_transfer_rapor.png"

    cikti = TransferGorsellestirici.karsilastirma_raporu_ciz(sonuclar, siniflar, hedef_dosya=hedef)
    assert cikti.exists()
    assert cikti.stat().st_size > 0
