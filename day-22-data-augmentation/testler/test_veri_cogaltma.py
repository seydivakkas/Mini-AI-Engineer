"""Day 22 Birim Testleri: Veri Çoğaltma (Albumentations, Torchvision, MixUp, CutMix)."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.albumentations_donusturucu import AlbumentationsDonusturucu
from src.torchvision_donusturucu import TorchvisionDonusturucu
from src.mixup_cutmix import MixUpCutMixUygulayici, MixUpCutMixKayip
from src.karsilastirici import VeriCogaltmaKarsilastirici, StratejiSonucu
from src.gorsellestirici import VeriCogaltmaGorsellestirici


def test_albumentations_tekil_ve_toplu():
    """Albumentations tekil ve toplu görsel dönüştürmelerini test eder."""
    albu = AlbumentationsDonusturucu((64, 64))

    # Float görsel [0.0, 1.0]
    img_float = np.random.rand(64, 64, 3).astype(np.float32)
    aug_float = albu.donustur_tekil(img_float, mod="agir")
    assert aug_float.shape == (64, 64, 3)
    assert aug_float.dtype == np.float32
    assert 0.0 <= aug_float.min() <= aug_float.max() <= 1.0

    # Toplu dönüşüm (Batch)
    batch_img = np.random.rand(4, 64, 64, 3).astype(np.float32)
    batch_aug = albu.donustur_toplu(batch_img, mod="temel")
    assert batch_aug.shape == (4, 64, 64, 3)


def test_torchvision_donusturucu_tensor_ve_numpy():
    """torchvision dönüşümlerinin Tensör ve NumPy üzerinde çalıştığını test eder."""
    tv = TorchvisionDonusturucu((64, 64))

    # Tensör dönüşümü
    t = torch.rand(3, 64, 64)
    t_aug = tv.donustur_tensor(t)
    assert t_aug.shape == (3, 64, 64)

    # NumPy dönüşümü
    np_img = np.random.rand(64, 64, 3).astype(np.float32)
    np_aug = tv.donustur_numpy(np_img)
    assert np_aug.shape == (64, 64, 3)
    assert 0.0 <= np_aug.min() <= np_aug.max() <= 1.0


def test_mixup_uygulayici():
    """MixUp fonksiyonunun tensör boyutunu ve etiket ağırlıklarını koruduğunu doğrular."""
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 4, (8,))

    mix_x, ya, yb, lam = MixUpCutMixUygulayici.uygula_mixup(x, y, alpha=0.8)

    assert mix_x.shape == x.shape
    assert ya.shape == y.shape
    assert yb.shape == y.shape
    assert 0.0 <= lam <= 1.0


def test_cutmix_uygulayici():
    """CutMix fonksiyonunun kesme kutusunu doğru uyguladığını doğrular."""
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 4, (8,))

    cut_x, ya, yb, lam = MixUpCutMixUygulayici.uygula_cutmix(x, y, alpha=1.0)

    assert cut_x.shape == x.shape
    assert ya.shape == y.shape
    assert yb.shape == y.shape
    assert 0.0 <= lam <= 1.0


def test_mixup_cutmix_kayip():
    """Çift etiketli kayıp fonksiyonunun geriye yayılım (backprop) ile çalıştığını doğrular."""
    criterion = MixUpCutMixKayip()
    tahminler = torch.randn(4, 3, requires_grad=True)
    ya = torch.tensor([0, 1, 2, 0])
    yb = torch.tensor([1, 2, 0, 2])
    lam = 0.7

    loss = criterion(tahminler, ya, yb, lam)
    assert loss.dim() == 0
    loss.backward()
    assert tahminler.grad is not None


def test_karsilastirici_egitim_dongusu():
    """Karşılaştırma motorunun mini bir ablation testini başarıyla tamamladığını doğrular."""
    karsilastirici = VeriCogaltmaKarsilastirici(device=torch.device("cpu"))

    X_tr = np.random.rand(16, 64, 64, 3).astype(np.float32)
    y_tr = np.array([0, 1, 2, 3] * 4)
    X_te = np.random.rand(8, 64, 64, 3).astype(np.float32)
    y_te = np.array([0, 1, 2, 3] * 2)

    res = karsilastirici.egit_ve_test_et("MixUp", X_tr, y_tr, X_te, y_te, epochs=1, batch_size=8)
    assert isinstance(res, StratejiSonucu)
    assert 0.0 <= res.test_acc <= 1.0


def test_gorsellestirici_galeri_ve_rapor(tmp_path):
    """Galeri ve karşılaştırma çizelgelerinin PNG formatında oluşturulduğunu test eder."""
    X_ornek = np.random.rand(4, 64, 64, 3).astype(np.float32)
    y_ornek = np.array([0, 1, 2, 3])
    siniflar = ["Vazo", "Kumaş", "Rozet", "Ahşap"]

    galeri_yol = tmp_path / "test_galeri.png"
    cikti1 = VeriCogaltmaGorsellestirici.galeri_ciz(X_ornek, y_ornek, siniflar, hedef_dosya=galeri_yol)
    assert cikti1.exists()
    assert cikti1.stat().st_size > 0

    sonuclar = [
        StratejiSonucu("Baseline", 1.0, 0.9, 0.9, 0.5, 0.9, 1.0),
        StratejiSonucu("MixUp", 1.0, 0.95, 0.95, 0.85, 0.95, 1.2),
    ]
    rapor_yol = tmp_path / "test_rapor.png"
    cikti2 = VeriCogaltmaGorsellestirici.karsilastirma_raporu_ciz(sonuclar, hedef_dosya=rapor_yol)
    assert cikti2.exists()
    assert cikti2.stat().st_size > 0
