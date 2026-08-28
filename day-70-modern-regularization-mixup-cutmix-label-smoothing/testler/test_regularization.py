"""
Day 70: Modern Regülerizasyon Birim Test Paketi
==============================================
Mixup, CutMix, Yumuşatılmış Çapraz Entropi (Label Smoothing),
Model mimarisi ve Görselleştirici doğrulama testleri.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from src.mixup_cutmix import ModernArtirici
from src.kayip_fonksiyonlari import YumusatilmisCrossEntropyLoss
from src.deney_modeli import ModernRegulerVisionNet
from src.reguler_karsilastirici import RegulerizasyonLaboratuvari
from src.gorsellestirici import RegulerizasyonGorsellestirici


@pytest.fixture
def gecici_dizin():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_mixup_uygulamasi_ve_boyut_korunumu():
    """Mixup operasyonunun tensör boyutlarını koruduğunu ve geçerli lambda ürettiğini test eder."""
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 5, (8,))

    x_mix, y_a, y_b, lam = ModernArtirici.uygula_mixup(x, y, alpha=0.8)

    assert x_mix.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0.0 <= lam <= 1.0


def test_cutmix_uygulamasi_ve_kutu_sinirlari():
    """CutMix operasyonunun tensör boyutlarını koruduğunu ve geçerli kesim yaptığını test eder."""
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 5, (8,))

    x_cut, y_a, y_b, lam = ModernArtirici.uygula_cutmix(x, y, alpha=1.0)

    assert x_cut.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0.0 <= lam <= 1.0


def test_rastgele_kutu_olustur_koordinatlar():
    """CutMix kutu koordinatlarının görsel sınırları içinde kaldığını test eder."""
    W, H = 64, 64
    x1, y1, x2, y2 = ModernArtirici.rastgele_kutu_olustur(W, H, lam=0.5)

    assert 0 <= x1 <= x2 <= W
    assert 0 <= y1 <= y2 <= H


def test_yumusatilmis_cross_entropy_tekil_hedef():
    """Tekil etiketle Label Smoothing kaybının standart hesaplandığını test eder."""
    kriter = YumusatilmisCrossEntropyLoss(smoothing=0.1)
    logits = torch.randn(4, 5)
    targets = torch.tensor([0, 2, 4, 1])

    loss = kriter(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_yumusatilmis_cross_entropy_cift_hedef_mixup():
    """Mixup/CutMix çift hedefli interpolasyon kaybını test eder."""
    kriter = YumusatilmisCrossEntropyLoss(smoothing=0.1)
    logits = torch.randn(4, 5)
    y_a = torch.tensor([0, 1, 2, 3])
    y_b = torch.tensor([4, 3, 2, 1])

    loss = kriter(logits, y_a, y_b, lam=0.7)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_gecersiz_smoothing_degeri():
    """Geçersiz smoothing katsayısında ValueError fırlatıldığını test eder."""
    with pytest.raises(ValueError):
        YumusatilmisCrossEntropyLoss(smoothing=-0.1)
    with pytest.raises(ValueError):
        YumusatilmisCrossEntropyLoss(smoothing=1.0)


def test_modern_reguler_vision_net_ileri_yayilim():
    """Modelin ileri yayılımda beklenen sınıf logit tensörünü ürettiğini test eder."""
    model = ModernRegulerVisionNet(girdi_kanali=3, sinif_sayisi=5, taban_kanal=16)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 5)


def test_gorsellestirici_pano_kaydi(gecici_dizin: str):
    """Görselleştiricinin teşhis panosu görselini ürettiğini test eder."""
    panel_yolu = os.path.join(gecici_dizin, "test_reg_paneli.png")
    sonuclar = RegulerizasyonLaboratuvari.tum_laboratuvari_kos(toplam_epoch=2)

    cizim_yolu = RegulerizasyonGorsellestirici.panoyu_ciz_ve_kaydet(
        laboratuvar_sonuclari=sonuclar,
        cikti_yolu=panel_yolu
    )

    assert os.path.exists(cizim_yolu)
    assert os.path.getsize(cizim_yolu) > 10000
