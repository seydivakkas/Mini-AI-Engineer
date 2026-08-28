"""Day 27 Birim Testleri: U-Net Mimarisi, Kayıp Fonksiyonları ve Bölütleme Metrikleri."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.unet_modeli import UNet
from src.kayip_ve_metrikler import BolutlemeMetrikleri, ComboLoss, DiceLoss
from src.sentetik_veri_yoneticisi import SentetikBolutlemeDataset, VeriYoneticisi
from src.egitici import BolutlemeEgitici
from src.gorsellestirici import BolutlemeGorsellestirici


def test_unet_ileri_besleme_ve_boyutlar():
    """U-Net modelinin giriş boyutunu koruyarak (N, C, H, W) çıktısı ürettiğini test eder."""
    model = UNet(in_channels=3, num_classes=3, kanal_tabani=16, bilinear=False)
    x = torch.randn(2, 3, 64, 64)

    out = model(x)
    assert out.shape == (2, 3, 64, 64)

    # Bilinear seçeneği
    model_bi = UNet(in_channels=3, num_classes=2, kanal_tabani=16, bilinear=True)
    out_bi = model_bi(x)
    assert out_bi.shape == (2, 2, 64, 64)


def test_dice_ve_combo_loss():
    """Dice ve Combo Loss fonksiyonlarının doğru çalıştığını ve gradyan ürettiğini test eder."""
    dice_fn = DiceLoss()
    combo_fn = ComboLoss(alpha=0.5)

    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 3, (2, 32, 32))

    loss_dice = dice_fn(logits, targets)
    loss_combo = combo_fn(logits, targets)

    assert loss_dice.item() >= 0.0
    assert loss_combo.item() >= 0.0

    loss_combo.backward()
    assert logits.grad is not None


def test_piksel_dogrulugu_ve_iou():
    """Piksel doğruluğu, IoU (Jaccard) ve Dice metriklerinin matematiksel tutarlılığını test eder."""
    true_mask = np.array([
        [0, 1],
        [1, 2],
    ])
    pred_mask = np.array([
        [0, 1],
        [1, 2],
    ])

    acc = BolutlemeMetrikleri.piksel_dogrulugu(pred_mask, true_mask)
    ious, dices = BolutlemeMetrikleri.sinif_iou_ve_dice(pred_mask, true_mask, num_classes=3)

    assert acc == 1.0
    assert ious[0] == 1.0
    assert ious[1] == 1.0
    assert ious[2] == 1.0
    assert dices[0] == 1.0


def test_sentetik_veri_seti_ve_dataloader():
    """Sentetik hücre veri setinin tensör boyutlarını ve etiket aralığını test eder."""
    dataset = SentetikBolutlemeDataset(ornek_sayisi=4, img_size=64, seed=42)
    assert len(dataset) == 4

    img, mask = dataset[0]
    assert img.shape == (3, 64, 64)
    assert mask.shape == (64, 64)
    assert mask.min() >= 0 and mask.max() <= 2


def test_egitici_bir_epok():
    """BolutlemeEgitici sınıfının bir epokluk eğitimi hatasız yürüttüğünü test eder."""
    train_loader, val_loader, _, _ = VeriYoneticisi.dataloader_olustur(
        train_adet=4, val_adet=2, img_size=64, batch_size=2
    )
    model = UNet(in_channels=3, num_classes=3, kanal_tabani=8)
    egitici = BolutlemeEgitici(model, device="cpu", lr=1e-3)

    train_loss = egitici.bir_epok_egit(train_loader)
    val_loss, rapor = egitici.dogrula(val_loader, SentetikBolutlemeDataset.SINIFLAR)

    assert train_loss > 0.0
    assert val_loss > 0.0
    assert "miou" in rapor


def test_dashboard_gorsellestirici(tmp_path):
    """6 panelli bölütleme teşhis panosunun oluşturulduğunu test eder."""
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    dummy_mask = np.zeros((64, 64), dtype=int)
    tarihce = {"train_loss": [1.0], "val_loss": [0.8], "val_miou": [0.5], "val_pixel_acc": [0.7], "val_mean_dice": [0.6]}
    sinif_raporu = {
        "Arka Plan": {"iou": 0.8, "dice": 0.88},
        "Hücre Gövdesi": {"iou": 0.6, "dice": 0.75},
        "Çekirdek": {"iou": 0.5, "dice": 0.66},
    }
    siniflar = list(sinif_raporu.keys())

    hedef = tmp_path / "test_bolutleme_paneli.png"
    cikti = BolutlemeGorsellestirici.dashboard_ciz(
        dummy_img, dummy_mask, dummy_mask, tarihce, sinif_raporu, siniflar, hedef_dosya=hedef
    )

    assert cikti.exists()
    assert cikti.stat().st_size > 0
