"""
Day 47: Scikit-Learn Güvenli Pipeline ve Nested CV Birim Testleri.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.pipeline_mimari import GuvenliPipelineUretici
from src.sizinti_dedektoru import TargetLeakageDedektoru
from src.nested_cv_motoru import NestedCVMotoru
from src.gorsellestirici import PipelineTehisGorsellestirici


@pytest.fixture
def ornek_veri():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "gelir": np.random.uniform(1000, 10000, size=n),
        "yas": np.random.randint(18, 70, size=n),
        "sehir": np.random.choice(["A", "B", "C"], size=n),
        "kategori": np.random.choice(["X", "Y"], size=n),
        "hedef": np.random.choice([0, 1], size=n)
    })


def test_guvenli_pipeline_olusturma():
    """Güvenli pipeline nesnesinin beklenen adımlara sahip olduğunu doğrular."""
    pipe = GuvenliPipelineUretici.guvenli_pipeline_olustur(
        sayisal_kolonlar=["gelir", "yas"],
        kategorik_kolonlar=["sehir", "kategori"]
    )
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps


def test_pipeline_eksik_ve_yeni_kategorileri_yonetir(ornek_veri):
    """Pipeline'ın eksik verileri doldurup yeni/bilinmeyen kategorileri hatasız yönettiğini test eder."""
    pipe = GuvenliPipelineUretici.guvenli_pipeline_olustur(
        sayisal_kolonlar=["gelir", "yas"],
        kategorik_kolonlar=["sehir", "kategori"]
    )
    X = ornek_veri.drop(columns=["hedef"])
    y = ornek_veri["hedef"]
    pipe.fit(X, y)

    # Test verisinde NaN ve yeni bir kategori 'Z' olsun
    test_df = pd.DataFrame({
        "gelir": [np.nan, 5000.0],
        "yas": [30, np.nan],
        "sehir": ["Z", "A"],  # 'Z' eğitimde yoktu
        "kategori": ["X", np.nan]
    })

    tahminler = pipe.predict(test_df)
    assert len(tahminler) == 2
    assert set(tahminler).issubset({0, 1})


def test_target_leakage_dedektoru_temiz_veri(ornek_veri):
    """Temiz veri setinde hedef sızıntısı bulunmadığını test eder."""
    dedektor = TargetLeakageDedektoru(korelasyon_esigi=0.88)
    rapor = dedektor.denetle(ornek_veri, hedef_kolon="hedef")
    assert rapor["durum"] == "GUVENLI_VERI_SETI"
    assert rapor["supheli_kolon_sayisi"] == 0


def test_target_leakage_dedektoru_sizintiyi_yakalar(ornek_veri):
    """Aşırı korele yapay sızıntı kolonunun tespit edildiğini test eder."""
    df_sizintili = ornek_veri.copy()
    # Hedef değişkenden neredeyse birebir türetilmiş sızıntı kolonu
    df_sizintili["yapay_sizinti"] = df_sizintili["hedef"] * 100.0 + np.random.normal(0, 0.01, size=len(df_sizintili))

    dedektor = TargetLeakageDedektoru(korelasyon_esigi=0.88)
    rapor = dedektor.denetle(df_sizintili, hedef_kolon="hedef")

    assert rapor["durum"] == "SIZINTI_RISKI_TESPIT_EDILDI"
    assert rapor["supheli_kolon_sayisi"] >= 1
    assert rapor["supheli_kolonlar"][0]["kolon"] == "yapay_sizinti"


def test_nested_cv_yurutumu(ornek_veri):
    """Nested Cross-Validation motorunun hatasız çalıştığını ve skor ürettiğini test eder."""
    X = ornek_veri.drop(columns=["hedef"])
    y = ornek_veri["hedef"]
    motor = NestedCVMotoru(outer_splits=3, inner_splits=2, random_state=42)

    sonuc = motor.nested_cv_yurut(
        X=X,
        y=y,
        sayisal_kolonlar=["gelir", "yas"],
        kategorik_kolonlar=["sehir", "kategori"],
        param_grid={"classifier__C": [0.1, 1.0]}
    )

    assert len(sonuc["outer_skorlar"]) == 3
    assert 0.0 <= sonuc["ortalama_auc"] <= 1.0


def test_sizintili_vs_guvenli_karsilastirma(ornek_veri):
    """Sızıntılı karşılaştırma simülasyonunun geçerli AUC ürettiğini test eder."""
    X = ornek_veri.drop(columns=["hedef"])
    y = ornek_veri["hedef"]
    motor = NestedCVMotoru(outer_splits=3, inner_splits=2, random_state=42)

    leaky_sonuc = motor.sizintili_karsilastirma_yurut(X, y, ["gelir", "yas"])
    assert "leaky_ortalama_auc" in leaky_sonuc
    assert 0.0 <= leaky_sonuc["leaky_ortalama_auc"] <= 1.0


def test_gorsellestirici_panel_uretimi(tmp_path):
    """Görselleştirici panelinin PNG dosyası ürettiğini test eder."""
    nested_sonuc = {"outer_skorlar": [0.8, 0.82, 0.85], "ortalama_auc": 0.823, "std_auc": 0.02}
    leaky_sonuc = {"leaky_ortalama_auc": 0.865, "leaky_std_auc": 0.015}
    sizinti_raporu = {"supheli_kolon_sayisi": 0, "tum_korelasyonlar": {"gelir": 0.25, "yas": -0.15}}
    katsayilar = {"gelir": 0.45, "yas": -0.30}

    cikis_yolu = str(tmp_path / "test_pipeline_paneli.png")
    yol = PipelineTehisGorsellestirici.panel_ciz(
        nested_sonuc, leaky_sonuc, sizinti_raporu, katsayilar, hedef_path=cikis_yolu
    )

    assert os.path.exists(yol)
    assert os.path.getsize(yol) > 1000
