"""
Day 46: Otomatik AI Deney Raporlama Motoru Birim Testleri.
"""

import os
import pytest
import numpy as np
from src.egitim_izleyici import EgitimGecmisi
from src.metrik_hesaplayici import MetrikHesaplayici
from src.raporlayici import OtomatikDeneyRaporlayici
from src.gorsellestirici import DeneyRaporuGorsellestirici


@pytest.fixture
def ornek_telemetri():
    gecmis = EgitimGecmisi(model_adi="TestNet", sabir_patience=3)
    gecmis.epoch_ekle(1, 0.9, 0.95, 0.5, 0.48)
    gecmis.epoch_ekle(2, 0.7, 0.75, 0.65, 0.62)
    gecmis.epoch_ekle(3, 0.4, 0.50, 0.82, 0.80)  # En iyi val loss
    gecmis.epoch_ekle(4, 0.3, 0.55, 0.88, 0.79)
    gecmis.epoch_ekle(5, 0.2, 0.60, 0.92, 0.78)
    gecmis.epoch_ekle(6, 0.15, 0.65, 0.95, 0.77)
    return gecmis


@pytest.fixture
def ornek_tahminler():
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.85, 0.7, 0.2, 0.1, 0.3, 0.8, 0.4, 0.65, 0.15])
    return y_true, y_prob


def test_egitim_gecmisi_analizi(ornek_telemetri):
    """Eğitim geçmişi analizinin en iyi epoch ve overfitting farkını doğru hesapladığını test eder."""
    analiz = ornek_telemetri.analiz_et()
    assert analiz["en_iyi_epoch"] == 3
    assert analiz["en_iyi_val_loss"] == 0.50
    assert analiz["overfitting_gap"] == pytest.approx(0.15, abs=1e-4)
    assert analiz["erken_durdurma_tetiklendi"] is True


def test_karmasiklik_matrisi_hesaplama(ornek_tahminler):
    """Karmaşıklık matrisi ve türetilmiş oranların doğruluğunu test eder."""
    y_true, y_prob = ornek_tahminler
    cm = MetrikHesaplayici.karmasiklik_matrisi_hesapla(y_true, y_prob, esik=0.5)

    assert cm["tp"] == 5
    assert cm["tn"] == 5
    assert cm["fp"] == 0
    assert cm["fn"] == 0
    assert cm["dogruluk_acc"] == 100.0
    assert cm["f1_skoru"] == 1.0


def test_roc_auc_hesaplama(ornek_tahminler):
    """ROC eğrisi ve AUC alanının geçerli aralıkta olduğunu test eder."""
    y_true, y_prob = ornek_tahminler
    roc = MetrikHesaplayici.roc_egrisi_hesapla(y_true, y_prob)

    assert "roc_auc" in roc
    assert 0.8 <= roc["roc_auc"] <= 1.0
    assert len(roc["fpr"]) == 100
    assert len(roc["tpr"]) == 100


def test_pr_egrisi_hesaplama(ornek_tahminler):
    """PR eğrisi ve Average Precision hesabını test eder."""
    y_true, y_prob = ornek_tahminler
    pr = MetrikHesaplayici.pr_egrisi_hesapla(y_true, y_prob)

    assert "average_precision_ap" in pr
    assert pr["taban_oran"] == 0.5
    assert len(pr["precision"]) == 100


def test_erken_durdurma_tetikleyici():
    """Sabır (patience) aşılmadığında erken durdurmanın tetiklenmediğini test eder."""
    gecmis = EgitimGecmisi(sabir_patience=5)
    gecmis.epoch_ekle(1, 0.9, 0.9, 0.5, 0.5)
    gecmis.epoch_ekle(2, 0.8, 0.8, 0.6, 0.6)  # Son epoch en iyi
    analiz = gecmis.analiz_et()
    assert analiz["erken_durdurma_tetiklendi"] is False


def test_html_raporu_olusturma(ornek_telemetri, ornek_tahminler, tmp_path):
    """HTML raporunun başarıyla derlendiğini ve dosyaya yazıldığını test eder."""
    y_true, y_prob = ornek_tahminler
    egitim_analizi = ornek_telemetri.analiz_et()
    cm_analizi = MetrikHesaplayici.karmasiklik_matrisi_hesapla(y_true, y_prob)
    roc_analizi = MetrikHesaplayici.roc_egrisi_hesapla(y_true, y_prob)
    pr_analizi = MetrikHesaplayici.pr_egrisi_hesapla(y_true, y_prob)

    hedef = str(tmp_path / "test_rapor.html")
    yol = OtomatikDeneyRaporlayici.html_raporu_olustur(
        egitim_analizi, cm_analizi, roc_analizi, pr_analizi, {"LR": "0.001"}, hedef_path=hedef
    )

    assert os.path.exists(yol)
    with open(yol, "r", encoding="utf-8") as f:
        icerik = f.read()
        assert "Yapay Zeka Model Deney & Performans Raporu" in icerik
        assert "ROC-AUC Skoru" in icerik


def test_gorsellestirici_panel_cizimi(ornek_telemetri, ornek_tahminler, tmp_path):
    """6 panelli görselleştiricinin PNG dosyası ürettiğini test eder."""
    y_true, y_prob = ornek_tahminler
    egitim_analizi = ornek_telemetri.analiz_et()
    cm_analizi = MetrikHesaplayici.karmasiklik_matrisi_hesapla(y_true, y_prob)
    roc_analizi = MetrikHesaplayici.roc_egrisi_hesapla(y_true, y_prob)
    pr_analizi = MetrikHesaplayici.pr_egrisi_hesapla(y_true, y_prob)

    hedef = str(tmp_path / "test_panel.png")
    yol = DeneyRaporuGorsellestirici.panel_ciz(
        ornek_telemetri, egitim_analizi, cm_analizi, roc_analizi, pr_analizi, hedef_path=hedef
    )

    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
