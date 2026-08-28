"""
Day 59: Transfer Learning ve Dondurulmuş Katmanlarla L2-Normalize Embedding Çıkarımı Birim Testleri.
"""

import os
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.vektor_ekstraktor import DondurulmusEmbeddingEkstraktoru, OmurgaModelFabrikasi
from src.embedding_analizoru import EmbeddingGeometriAnalizoru
from src.gorsellestirici import EmbeddingGorsellestirici


def test_omurga_model_fabrikasi():
    """ResNet ve MiniViT modellerinin doğru boyutlarda embedding tensörü ürettiğini test eder."""
    resnet = OmurgaModelFabrikasi.uret("resnet", feature_dim=128)
    vit = OmurgaModelFabrikasi.uret("vit", embed_dim=64, depth=2, num_heads=2)

    dummy_x = torch.randn(2, 3, 32, 32)
    out_resnet = resnet(dummy_x)
    out_vit = vit(dummy_x)

    assert out_resnet.shape == (2, 128)
    assert out_vit.shape == (2, 64)


def test_katman_dondurma_garantisi():
    """Ekstraktörün tüm omurga parametrelerini dondurduğunu (requires_grad=False) test eder."""
    omurga = OmurgaModelFabrikasi.uret("resnet", feature_dim=64)
    ekstraktor = DondurulmusEmbeddingEkstraktoru(backbone=omurga, device="cpu")

    for param in ekstraktor.backbone.parameters():
        assert param.requires_grad is False
    assert ekstraktor.backbone.training is False


def test_l2_normalizasyon_birim_kure():
    """Çıkarılan tüm vektörlerin L2 normunun kesin olarak 1.0 olduğunu test eder."""
    omurga = OmurgaModelFabrikasi.uret("resnet", feature_dim=64)
    ekstraktor = DondurulmusEmbeddingEkstraktoru(backbone=omurga, normalize=True, device="cpu")

    dummy_x = torch.randn(10, 3, 32, 32)
    emb = ekstraktor(dummy_x).detach().numpy()

    norm_sonuc = EmbeddingGeometriAnalizoru.l2_norm_dogrula(emb)
    assert norm_sonuc["birim_kure_gecerli_mi"] is True
    assert abs(norm_sonuc["ort_norm"] - 1.0) < 1e-4


def test_toplu_embedding_cikartma():
    """DataLoader üzerinden toplu embedding çıkarımının eksiksiz çalıştığını test eder."""
    omurga = OmurgaModelFabrikasi.uret("resnet", feature_dim=32)
    ekstraktor = DondurulmusEmbeddingEkstraktoru(backbone=omurga, normalize=True, device="cpu")

    X = torch.randn(30, 3, 32, 32)
    y = torch.randint(0, 3, (30,))
    loader = DataLoader(TensorDataset(X, y), batch_size=10)

    emb, etiketler = ekstraktor.cikart(loader)
    assert emb.shape == (30, 32)
    assert etiketler.shape == (30,)
    assert emb.dtype == np.float32


def test_benzerlik_ve_ayrisabilirlik_analizi():
    """Benzerlik matrisi, sınıf içi/dışı ortalamalar ve ayrışabilirlik oranının hesaplandığını test eder."""
    np.random.seed(42)
    emb = np.random.randn(40, 32).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([0] * 20 + [1] * 20)

    sonuc = EmbeddingGeometriAnalizoru.benzerlik_analizi(emb, labels)
    assert "ort_intra_benzerlik" in sonuc
    assert "ort_inter_benzerlik" in sonuc
    assert sonuc["ayrisabilirlik_orani"] > 0.0


def test_linear_probing_egitimi():
    """Dondurulmuş embeddingler üzerinde linear probe sınıflandırıcısının çalıştığını test eder."""
    np.random.seed(42)
    train_emb = np.random.randn(50, 16).astype(np.float32)
    train_emb = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    train_y = np.random.randint(0, 2, size=50)

    val_emb = np.random.randn(20, 16).astype(np.float32)
    val_emb = val_emb / np.linalg.norm(val_emb, axis=1, keepdims=True)
    val_y = np.random.randint(0, 2, size=20)

    probe_sonuc = EmbeddingGeometriAnalizoru.linear_probing_egit(
        train_emb, train_y, val_emb, val_y, num_classes=2, epochs=3
    )
    assert 0.0 <= probe_sonuc["nihai_val_dogruluk"] <= 100.0
    assert len(probe_sonuc["dogruluk_gecmisi"]) == 3


def test_gorsellestirici_panel_cizimi(tmp_path):
    """6 panelli görselleştiricinin PNG dosyası ürettiğini test eder."""
    np.random.seed(42)
    emb = np.random.randn(60, 16).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.random.randint(0, 3, size=60)

    norm_bilgisi = EmbeddingGeometriAnalizoru.l2_norm_dogrula(emb)
    benzerlik_bilgisi = EmbeddingGeometriAnalizoru.benzerlik_analizi(emb, labels)
    svd_bilgisi = EmbeddingGeometriAnalizoru.svd_ve_izotropi_analizi(emb)
    probe_bilgisi = {"nihai_val_dogruluk": 85.0, "dogruluk_gecmisi": [60.0, 75.0, 85.0]}

    hedef = str(tmp_path / "test_embedding_paneli.png")
    cikis = EmbeddingGorsellestirici.panel_ciz(
        embeddings=emb,
        labels=labels,
        norm_bilgisi=norm_bilgisi,
        benzerlik_bilgisi=benzerlik_bilgisi,
        svd_bilgisi=svd_bilgisi,
        probe_bilgisi=probe_bilgisi,
        hedef_path=hedef
    )

    assert os.path.exists(cikis)
    assert os.path.getsize(cikis) > 1000
