"""
Day 45: Özellik Mühendisliği ve Feature Store Profil Oluşturucu Birim Testleri.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.kodlayicilar import KategorikKodlayici
from src.olcekleyiciler import SayisalOlcekleyici
from src.ozellik_profili import FeatureStoreProfilci
from src.gorsellestirici import OzellikMuhendisligiGorsellestirici


@pytest.fixture
def ornek_veri():
    np.random.seed(42)
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "kategori": ["A", "A", "B", "B", "B", "C"],
        "gelir": [1000.0, 2000.0, 3000.0, 4000.0, 10000.0, 500.0],
        "yas": [25, 35, 45, 20, 50, 60],
        "hedef": [0.1, 0.2, 0.8, 0.9, 0.7, 0.0]
    })


def test_frequency_encoding(ornek_veri):
    """Frekans kodlamasının görülme sıklıklarını doğru oranladığını test eder."""
    kodlayici = KategorikKodlayici()
    kodlayici.fit_frequency_encoding(ornek_veri, ["kategori"])
    res_df = kodlayici.transform_frequency_encoding(ornek_veri, ["kategori"])

    assert "kategori_freq_enc" in res_df.columns
    # B kategorisi 3/6 = 0.5 olmalı
    assert res_df.loc[res_df["kategori"] == "B", "kategori_freq_enc"].iloc[0] == pytest.approx(0.5)


def test_smoothed_target_encoding(ornek_veri):
    """Hedef kodlamanın global ortalamaya doğru pürüzsüzleştirildiğini test eder."""
    kodlayici = KategorikKodlayici(smoothing_weight=2.0)
    kodlayici.fit_target_encoding(ornek_veri, ["kategori"], "hedef")
    res_df = kodlayici.transform_target_encoding(ornek_veri, ["kategori"])

    assert "kategori_target_enc" in res_df.columns
    assert res_df["kategori_target_enc"].isna().sum() == 0
    # B kategorisi ortalaması (0.8+0.9+0.7)/3 = 0.8 civarında olmalı
    assert res_df.loc[res_df["kategori"] == "B", "kategori_target_enc"].iloc[0] > 0.5


def test_one_hot_encoding(ornek_veri):
    """One-Hot Encoding'in beklenen ikili sütunları ürettiğini test eder."""
    kodlayici = KategorikKodlayici()
    res_df = kodlayici.fit_transform_one_hot(ornek_veri, ["kategori"], drop_first=False)

    assert "kategori_A" in res_df.columns
    assert "kategori_B" in res_df.columns
    assert "kategori_C" in res_df.columns
    assert res_df["kategori_A"].sum() == 2.0


def test_standard_scaler(ornek_veri):
    """StandardScaler'ın ortalamayı 0, standart sapmayı 1 yaptığını test eder."""
    olcekleyici = SayisalOlcekleyici()
    olcekleyici.fit_standard_scaler(ornek_veri, ["yas"])
    res_df = olcekleyici.transform_standard_scaler(ornek_veri, ["yas"])

    assert "yas_std_scaled" in res_df.columns
    assert res_df["yas_std_scaled"].mean() == pytest.approx(0.0, abs=1e-6)
    assert res_df["yas_std_scaled"].std() == pytest.approx(1.0, abs=0.1)


def test_robust_scaler(ornek_veri):
    """RobustScaler'ın medyanı 0 yaptığını test eder."""
    olcekleyici = SayisalOlcekleyici()
    olcekleyici.fit_robust_scaler(ornek_veri, ["gelir"])
    res_df = olcekleyici.transform_robust_scaler(ornek_veri, ["gelir"])

    assert "gelir_robust_scaled" in res_df.columns
    assert res_df["gelir_robust_scaled"].median() == pytest.approx(0.0, abs=1e-5)


def test_log1p_and_interaction_features(ornek_veri):
    """Log1p ve oran etkileşimlerinin başarıyla hesaplandığını test eder."""
    olcekleyici = SayisalOlcekleyici()
    res_df = olcekleyici.log1p_donusumu(ornek_veri, ["gelir"])
    assert "gelir_log1p" in res_df.columns
    assert res_df["gelir_log1p"].iloc[0] == pytest.approx(np.log1p(1000.0))

    res_df = olcekleyici.etkilesim_ve_oran_uret(res_df, "gelir", "yas", "gelir_yas_orani")
    assert "gelir_yas_orani" in res_df.columns
    assert res_df["gelir_yas_orani"].iloc[0] == pytest.approx(1000.0 / (25.0 + 1e-4), abs=0.1)


def test_feature_store_profilci_and_gorsellestirici(ornek_veri, tmp_path):
    """Feature store profilci ve görselleştiricinin eksiksiz çalıştığını test eder."""
    profil = FeatureStoreProfilci.profil_cikar(ornek_veri, hedef_kolon="hedef")
    assert profil["toplam_oznitelik_sayisi"] == len(ornek_veri.columns)
    assert "gelir" in profil["oznitelikler"]

    feast_schema = FeatureStoreProfilci.feast_sema_ihrac_et(profil)
    assert feast_schema["name"] == "customer_risk_features"

    cikis_yolu = str(tmp_path / "test_ozellik_paneli.png")
    yol = OzellikMuhendisligiGorsellestirici.panel_ciz(ornek_veri, ornek_veri, profil, hedef_kolon="hedef", hedef_path=cikis_yolu)
    assert os.path.exists(yol)
