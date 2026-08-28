"""
MiniViT CIFAR-100 Eğitim ve Regülarizasyon Dinamikleri Birim Testleri
---------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.minivit_modeli import MiniVisionTransformer
from src.veri_artirma import MixupCutMixUygulayici, rastgele_sinirlayici_kutu
from src.kayip_fonksiyonlari import YumusatilmisCrossEntropyKaybi
from src.egitici import ayristir_parametre_gruplari, hesapla_dogruluk_top_k, MiniViTEgitici


def test_ayristir_parametre_gruplari_weight_decay():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        giris_kanali=3,
        sinif_sayisi=10,
        gomulme_boyutu=32,
        derinlik=2,
        kafa_sayisi=2
    )

    gruplar = ayristir_parametre_gruplari(model, agirlik_azaltma=0.05)
    assert len(gruplar) == 2
    assert gruplar[0]["weight_decay"] == 0.05
    assert gruplar[1]["weight_decay"] == 0.0

    # pos_embed ve cls_token no_decay grubunda olmalı
    no_decay_params = gruplar[1]["params"]
    assert any(p is model.pos_embed for p in no_decay_params)
    assert any(p is model.cls_token for p in no_decay_params)


def test_mixup_uygulamasi_boyut_ve_aralik():
    aug = MixupCutMixUygulayici(mixup_alpha=0.8, cutmix_alpha=0.0, uygulama_olasiligi=1.0, sinif_sayisi=10)
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])

    aug_x, aug_y = aug(x, y)

    assert aug_x.shape == (4, 3, 32, 32)
    assert aug_y.shape == (4, 10)
    assert torch.all(aug_y >= 0.0) and torch.all(aug_y <= 1.0)
    assert torch.allclose(aug_y.sum(dim=-1), torch.ones(4), atol=1e-4)


def test_cutmix_sinirlayici_kutu():
    w, h = 32, 32
    lam = 0.5
    x1, y1, x2, y2 = rastgele_sinirlayici_kutu(w, h, lam)

    assert 0 <= x1 <= x2 <= w
    assert 0 <= y1 <= y2 <= h


def test_yumusatilmis_cross_entropy_tamsayi():
    kayip_fn = YumusatilmisCrossEntropyKaybi(etiket_yumusatma=0.1)
    logits = torch.randn(4, 10)
    targets = torch.tensor([1, 4, 2, 9])

    loss = kayip_fn(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_yumusatilmis_cross_entropy_soft_targets():
    kayip_fn = YumusatilmisCrossEntropyKaybi(etiket_yumusatma=0.1)
    logits = torch.randn(4, 10)
    # 2D yumuşak olasılık hedefleri
    targets = torch.softmax(torch.randn(4, 10), dim=-1)

    loss = kayip_fn(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_hesapla_dogruluk_top_k():
    # 4 örnek, 5 sınıf
    logits = torch.tensor([
        [10.0, 1.0, 0.0, 0.0, 0.0],  # Sınıf 0 doğru
        [0.0, 10.0, 1.0, 0.0, 0.0],  # Sınıf 1 doğru
        [0.0, 0.0, 10.0, 1.0, 0.0],  # Sınıf 2 doğru
        [0.0, 0.0, 0.0, 1.0, 10.0],  # Sınıf 4 doğru
    ])
    # 3'ü doğru, 1'i yanlış hedef
    targets = torch.tensor([0, 1, 2, 0])

    top1, top5 = hesapla_dogruluk_top_k(logits, targets, top_k=(1, 5))
    # 3/4 = %75 Top-1
    assert top1 == pytest.approx(75.0, abs=1e-3)
    # Hepsi top-5 içinde
    assert top5 == pytest.approx(100.0, abs=1e-3)


def test_egitici_ogrenme_orani_warmup_ve_cosine():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        sinif_sayisi=10,
        gomulme_boyutu=32,
        derinlik=2,
        kafa_sayisi=2
    )

    egitici = MiniViTEgitici(
        model=model,
        cihaz="cpu",
        ogrenme_orani=1e-3,
        min_ogrenme_orani=1e-5,
        toplam_epok=10,
        isinma_epok=3
    )

    # Epok 0 (Warmup başlangıcı)
    lr0 = egitici._ogrenme_orani_ayarla(0)
    assert lr0 < 1e-3

    # Epok 2 (Warmup sonu)
    lr2 = egitici._ogrenme_orani_ayarla(2)
    assert lr2 == pytest.approx(1e-3, rel=1e-2)

    # Epok 9 (Cosine sonu)
    lr9 = egitici._ogrenme_orani_ayarla(9)
    assert lr9 < 1e-3
    assert lr9 >= 1e-5


def test_egitici_tek_epok_egitim_ve_dogrulama():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        giris_kanali=3,
        sinif_sayisi=5,
        gomulme_boyutu=32,
        derinlik=2,
        kafa_sayisi=2
    )

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 5, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)

    egitici = MiniViTEgitici(
        model=model,
        cihaz="cpu",
        ogrenme_orani=1e-3,
        toplam_epok=2,
        isinma_epok=1
    )

    loss, top1, top5, grad_norm = egitici.egitim_adimi(loader, epok=0)
    assert isinstance(loss, float) and loss > 0
    assert 0.0 <= top1 <= 100.0
    assert 0.0 <= top5 <= 100.0
    assert grad_norm > 0

    val_loss, val_top1, val_top5 = egitici.dogrulama_adimi(loader)
    assert isinstance(val_loss, float) and val_loss > 0
    assert 0.0 <= val_top1 <= 100.0
