"""Day 21 Birim Testleri: PyTorch ile CNN Görsel Sınıflandırma."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.model_mimari import PyTorchVisionCNN, ConvBlok
from src.veri_hazirlayici import SentetikGorselDataset, VeriYoneticisi
from src.egitici import PyTorchEgitici, EgitimSonucu
from src.gorsellestirici import PyTorchGorsellestirici
from src.grad_cam import GradCAM


def test_pytorch_model_mimarisi_ve_cikti_sekli():
    """Modelin ileri geçişte doğru çıktı boyutunu ve parametre sayısını ürettiğini doğrular."""
    model = PyTorchVisionCNN(in_channels=3, num_classes=4, input_size=(64, 64))
    x = torch.randn(2, 3, 64, 64)
    out = model(x)

    assert out.shape == (2, 4)
    param_bilgi = model.count_parameters()
    assert param_bilgi["total"] > 0
    assert param_bilgi["trainable"] == param_bilgi["total"]


def test_pytorch_model_gecersiz_parametreler():
    """Geçersiz sınıf sayısı veya dropout oranında hata fırlatıldığını doğrular."""
    with pytest.raises(ValueError):
        PyTorchVisionCNN(in_channels=3, num_classes=1)

    with pytest.raises(ValueError):
        PyTorchVisionCNN(in_channels=3, num_classes=3, dropout_rate=1.2)


def test_sentetik_dataset_ve_dataloader():
    """SentetikGorselDataset ve DataLoader'ın doğru tensör dönüşümlerini yaptığını test eder."""
    gorseller = np.random.rand(10, 32, 32, 3).astype(np.float32)
    etiketler = np.random.randint(0, 3, size=10)

    dataset = SentetikGorselDataset(gorseller, etiketler)
    assert len(dataset) == 10

    x_item, y_item = dataset[0]
    assert x_item.shape == (3, 32, 32)
    assert isinstance(x_item, torch.Tensor)
    assert isinstance(y_item, torch.Tensor)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (4, 3, 32, 32)
    assert batch_y.shape == (4,)


def test_veri_yoneticisi_bolumleme():
    """Veri yöneticisinin Train/Val/Test bölümlerini doğru boyutlarda ürettiğini doğrular."""
    yonetici = VeriYoneticisi(hedef_boyut=(32, 32), random_state=42)
    X, y, siniflar = yonetici.sentetik_veri_seti_uret(sinif_basina_ornek=10)

    assert len(X) == 40
    assert len(siniflar) == 4

    train_l, val_l, test_l, X_test, y_test = yonetici.veri_bol_ve_yukleyicileri_olustur(
        X, y, val_orani=0.15, test_orani=0.15, batch_size=8
    )

    toplam_ornek = len(train_l.dataset) + len(val_l.dataset) + len(test_l.dataset)
    assert toplam_ornek == 40
    assert len(X_test) == len(test_l.dataset)


def test_pytorch_egitici_ve_erken_durdurma():
    """PyTorch eğitim ve değerlendirme döngüsünün eksiksiz çalıştığını doğrular."""
    yonetici = VeriYoneticisi(hedef_boyut=(32, 32), random_state=42)
    X, y, _ = yonetici.sentetik_veri_seti_uret(sinif_basina_ornek=6)
    train_l, val_l, test_l, _, _ = yonetici.veri_bol_ve_yukleyicileri_olustur(
        X, y, val_orani=0.2, test_orani=0.2, batch_size=8
    )

    model = PyTorchVisionCNN(in_channels=3, num_classes=4, input_size=(32, 32))
    egitici = PyTorchEgitici(model, device=torch.device("cpu"))

    tarihce, en_iyi_model = egitici.egit(
        train_l, val_l, epochs=2, learning_rate=0.005, patience=2
    )

    assert len(tarihce["train_loss"]) > 0
    assert len(tarihce["val_loss"]) > 0

    sonuc = egitici.degerlendir(test_l, tarihce, egitim_suresi_sn=0.5)
    assert isinstance(sonuc, EgitimSonucu)
    assert 0.0 <= sonuc.test_dogruluk <= 1.0
    assert 0.0 <= sonuc.f1_macro <= 1.0


def test_gorsellestirici_cizimi(tmp_path):
    """Teşhis raporunun PNG formatında başarıyla kaydedildiğini test eder."""
    yonetici = VeriYoneticisi(hedef_boyut=(32, 32), random_state=42)
    X, y, siniflar = yonetici.sentetik_veri_seti_uret(sinif_basina_ornek=6)
    train_l, val_l, test_l, X_test, _ = yonetici.veri_bol_ve_yukleyicileri_olustur(X, y)

    model = PyTorchVisionCNN(in_channels=3, num_classes=4, input_size=(32, 32))
    egitici = PyTorchEgitici(model, device=torch.device("cpu"))
    tarihce, _ = egitici.egit(train_l, val_l, epochs=1)
    sonuc = egitici.degerlendir(test_l, tarihce, egitim_suresi_sn=0.2)

    hedef = tmp_path / "test_pytorch_rapor.png"
    cikti = PyTorchGorsellestirici.egitim_raporu_ciz(sonuc, siniflar, X_test, hedef_dosya=hedef)

    assert cikti.exists()
    assert cikti.stat().st_size > 0


def test_grad_cam_aciklanabilirlik():
    """Grad-CAM kancalarının ve ısı haritası üretiminin doğruluğunu test eder."""
    model = PyTorchVisionCNN(in_channels=3, num_classes=4, input_size=(32, 32))
    grad_cam = GradCAM(model=model, hedef_katman=model.blok3.conv)

    x = torch.randn(1, 3, 32, 32)
    isi_haritasi, aciklanan_sinif = grad_cam.isi_haritasi_uret(x)

    assert isi_haritasi.shape == (32, 32)
    assert 0 <= aciklanan_sinif < 4
    assert 0.0 <= isi_haritasi.min() <= isi_haritasi.max() <= 1.0

    orijinal_rgb = np.random.rand(32, 32, 3).astype(np.float32)
    fig = grad_cam.bindirme_ciz(orijinal_rgb, isi_haritasi, sinif_adi="Test")
    assert fig is not None
