"""
Day 49: XGBoost Dengesiz Risk Sınıflandırıcısı ve TreeSHAP Birim Testleri.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.risk_veri_ureteci import RiskVeriSimulasyonu
from src.xgboost_risk_egitici import XGBoostRiskSiniflandirici
from src.gorsellestirici import XGBoostRiskGorsellestirici


@pytest.fixture
def ornek_risk_verisi():
    X, y = RiskVeriSimulasyonu.veri_seti_olustur(n_orneklem=300, pozitif_oran=0.10, random_state=42)
    return RiskVeriSimulasyonu.train_val_test_bol(X, y, random_state=42)


def test_risk_veri_uretimi_ve_dengesizlik():
    """Risk veri setinin üretildiğini ve pozitif oranının yaklaşık %5 olduğunu test eder."""
    X, y = RiskVeriSimulasyonu.veri_seti_olustur(n_orneklem=500, pozitif_oran=0.05, random_state=42)
    assert len(X) == 500
    assert len(y) == 500
    assert X.shape[1] == 7
    assert 0.03 <= y.mean() <= 0.07


def test_stratified_bolme_oranlari(ornek_risk_verisi):
    """Stratified bölmenin sınıf oranlarını koruduğunu test eder."""
    X_train, X_val, X_test, y_train, y_val, y_test = ornek_risk_verisi
    assert len(X_train) == 210
    assert len(X_val) == 45
    assert len(X_test) == 45
    assert y_train.sum() > 0
    assert y_val.sum() > 0
    assert y_test.sum() > 0


def test_scale_pos_weight_hesabi(ornek_risk_verisi):
    """scale_pos_weight değerinin negatif/pozitif oranına eşit olduğunu test eder."""
    X_train, X_val, X_test, y_train, y_val, y_test = ornek_risk_verisi
    siniflandirici = XGBoostRiskSiniflandirici(n_estimators=10)
    siniflandirici.fit(X_train, y_train, X_val, y_val)

    beklenen_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    assert siniflandirici.scale_pos_weight == pytest.approx(beklenen_weight, abs=0.1)


def test_model_fit_ve_erken_durdurma(ornek_risk_verisi):
    """XGBoost modelinin eğitildiğini ve değerlendirme geçmişini kaydettiğini test eder."""
    X_train, X_val, X_test, y_train, y_val, y_test = ornek_risk_verisi
    siniflandirici = XGBoostRiskSiniflandirici(n_estimators=30, early_stopping_rounds=5)
    siniflandirici.fit(X_train, y_train, X_val, y_val)

    assert "validation_0" in siniflandirici.egitim_gecmisi
    assert "validation_1" in siniflandirici.egitim_gecmisi


def test_esik_optimizasyonu(ornek_risk_verisi):
    """Eşik optimizasyonunun geçerli bir olasılık eşiği ve F1 skoru ürettiğini test eder."""
    X_train, X_val, X_test, y_train, y_val, y_test = ornek_risk_verisi
    siniflandirici = XGBoostRiskSiniflandirici(n_estimators=20)
    siniflandirici.fit(X_train, y_train, X_val, y_val)

    esik_sonuc = siniflandirici.esik_degeri_optimize_et(X_val, y_val)
    assert 0.1 <= esik_sonuc["optimal_esik"] <= 0.9
    assert 0.0 <= esik_sonuc["en_iyi_val_f1"] <= 1.0


def test_treeshap_katkilari(ornek_risk_verisi):
    """TreeSHAP matrisi ve ortalama mutlak önem sözlüğünü test eder."""
    X_train, X_val, X_test, y_train, y_val, y_test = ornek_risk_verisi
    siniflandirici = XGBoostRiskSiniflandirici(n_estimators=20)
    siniflandirici.fit(X_train, y_train, X_val, y_val)

    shap_sonuc = siniflandirici.shap_katkilarini_cikar(X_test)
    assert shap_sonuc["shap_matrisi"].shape == (len(X_test), X_test.shape[1])
    assert len(shap_sonuc["ortalama_mutlak_shap"]) == X_test.shape[1]


def test_gorsellestirici_panel_cizimi(ornek_risk_verisi, tmp_path):
    """6 panelli teşhis panosunun PNG çıktısını ürettiğini test eder."""
    X_train, X_val, X_test, y_train, y_val, y_test = ornek_risk_verisi
    siniflandirici = XGBoostRiskSiniflandirici(n_estimators=15)
    siniflandirici.fit(X_train, y_train, X_val, y_val)

    esik_sonuc = siniflandirici.esik_degeri_optimize_et(X_val, y_val)
    test_sonuc = siniflandirici.degerlendir(X_test, y_test)
    shap_sonuc = siniflandirici.shap_katkilarini_cikar(X_test)

    cikis_yolu = str(tmp_path / "test_xgboost_paneli.png")
    yol = XGBoostRiskGorsellestirici.panel_ciz(
        test_sonuclari=test_sonuc,
        esik_sonuclari=esik_sonuc,
        shap_sonuclari=shap_sonuc,
        egitim_gecmisi=siniflandirici.egitim_gecmisi,
        scale_pos_weight=siniflandirici.scale_pos_weight,
        y_test=y_test,
        hedef_path=cikis_yolu
    )

    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
