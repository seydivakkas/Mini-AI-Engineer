"""Day 24 Birim Testleri: Model Değerlendirme ve Hata Analizi."""

from pathlib import Path
import numpy as np
import pytest
from scipy.special import softmax

from src.metrik_hesaplayici import MetrikHesaplayici
from src.kalibrasyon_analizcisi import KalibrasyonAnalizcisi
from src.hata_denetcisi import HataDenetcisi
from src.gorsellestirici import DegerlendirmeGorsellestirici


def test_top_k_dogruluk():
    """Top-k doğruluk fonksiyonunun Top-1 <= Top-2 <= Top-3 özelliğini sağladığını test eder."""
    y_true = np.array([0, 1, 2, 3])
    y_probs = np.array([
        [0.7, 0.2, 0.1, 0.0],
        [0.4, 0.5, 0.1, 0.0],
        [0.1, 0.4, 0.3, 0.2],  # Top-1 yanlış (1), Top-2 doğru (1,2)
        [0.1, 0.2, 0.3, 0.4],
    ])

    top_k = MetrikHesaplayici.top_k_dogruluk(y_true, y_probs, k_listesi=(1, 2, 3))
    assert top_k[1] == 0.75
    assert top_k[2] == 1.0
    assert top_k[3] == 1.0


def test_cok_sinifli_roc_ve_pr_auc():
    """Çok sınıflı ROC-AUC ve PR-AUC fonksiyonlarının geçerli AUC değerleri ürettiğini test eder."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ])

    roc_bilgi = MetrikHesaplayici.cok_sinifli_roc_egrileri(y_true, y_probs, n_classes=3)
    pr_bilgi = MetrikHesaplayici.cok_sinifli_pr_egrileri(y_true, y_probs, n_classes=3)

    assert 0.0 <= roc_bilgi["macro_auc"] <= 1.0
    assert 0.0 <= pr_bilgi["macro_ap"] <= 1.0
    assert len(roc_bilgi["sinif_roclari"]) == 3
    assert len(pr_bilgi["sinif_prleri"]) == 3


def test_kapsamli_rapor():
    """Kapsamlı rapor fonksiyonunun tüm metrik alanlarını eksiksiz ürettiğini test eder."""
    y_true = np.array([0, 1, 2, 3])
    y_probs = np.eye(4)
    siniflar = ["A", "B", "C", "D"]

    rapor = MetrikHesaplayici.kapsamli_rapor(y_true, y_probs, siniflar)
    assert rapor["dogruluk"] == 1.0
    assert rapor["f1_macro"] == 1.0
    assert "top_k" in rapor
    assert "sinif_raporu" in rapor


def test_kalibrasyon_ve_brier():
    """ECE, MCE ve Brier skorlarının matematiksel aralıklarını test eder."""
    y_true = np.array([0, 1, 1, 0])
    y_probs = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.8, 0.2],  # Yanlış ve aşırı güvenli
        [0.6, 0.4],
    ])

    kalib = KalibrasyonAnalizcisi.kalibrasyon_egrisi_ve_ece(y_true, y_probs, n_bins=5)
    brier = KalibrasyonAnalizcisi.brier_skoru(y_true, y_probs, n_classes=2)

    assert 0.0 <= kalib["ece"] <= 1.0
    assert 0.0 <= kalib["mce"] <= 1.0
    assert brier >= 0.0


def test_sicaklik_olcekleme_optimizasyonu():
    """Sıcaklık ölçeklemenin logitleri kalibre edip ECE'yi iyileştirdiğini doğrular."""
    np.random.seed(42)
    y_true = np.random.randint(0, 3, size=50)
    logits = np.random.normal(0, 2.0, (50, 3))
    # Yapay aşırı güven oluştur
    logits *= 3.0

    T_opt = KalibrasyonAnalizcisi.sicaklik_olcekleme_optimize_et(y_true, logits)
    assert T_opt > 0.0

    probs_scaled = KalibrasyonAnalizcisi.sicaklik_uygula(logits, T_opt)
    assert probs_scaled.shape == (50, 3)
    np.testing.assert_almost_equal(probs_scaled.sum(axis=1), np.ones(50), decimal=5)


def test_hata_denetcisi():
    """Aşırı güvenli hataları ve en çok karışan sınıf çiftlerini doğru tespit ettiğini test eder."""
    y_true = np.array([0, 1, 2, 0])
    y_probs = np.array([
        [0.95, 0.05, 0.00],  # Doğru
        [0.90, 0.05, 0.05],  # Aşırı güvenli yanlış (Gerçek 1, Tahmin 0, %90 güven)
        [0.10, 0.10, 0.80],  # Doğru
        [0.20, 0.70, 0.10],  # Yanlış (Gerçek 0, Tahmin 1, %70 güven)
    ])
    siniflar = ["A", "B", "C"]

    hatalar = HataDenetcisi.asiri_guvenli_yanlislar(y_true, y_probs, en_fazla=2)
    assert len(hatalar) == 2
    assert hatalar[0]["ornek_indeks"] == 1
    assert hatalar[0]["guven"] == 0.90

    ciftler = HataDenetcisi.en_cok_karisan_ciftler(y_true, np.argmax(y_probs, axis=1), siniflar)
    assert len(ciftler) == 2


def test_dashboard_gorsellestirici(tmp_path):
    """6 panelli değerlendirme panosunun PNG olarak kaydedildiğini test eder."""
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    logits = np.random.randn(8, 4)
    y_probs = softmax(logits, axis=1)
    siniflar = ["Vazo", "Kumaş", "Rozet", "Ahşap"]

    rapor = MetrikHesaplayici.kapsamli_rapor(y_true, y_probs, siniflar)
    kalib_ham = KalibrasyonAnalizcisi.kalibrasyon_egrisi_ve_ece(y_true, y_probs, n_bins=5)
    kalib_cal = KalibrasyonAnalizcisi.kalibrasyon_egrisi_ve_ece(y_true, y_probs, n_bins=5)
    hatalar = HataDenetcisi.asiri_guvenli_yanlislar(y_true, y_probs, en_fazla=3)

    hedef = tmp_path / "test_dashboard.png"
    cikti = DegerlendirmeGorsellestirici.dashboard_ciz(
        rapor, kalib_ham, kalib_cal, hatalar, siniflar, hedef_dosya=hedef
    )

    assert cikti.exists()
    assert cikti.stat().st_size > 0
