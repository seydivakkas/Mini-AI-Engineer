"""Keşifçi Veri Analizi (EDA Lab) Birim Testleri.

Bu dosya; korelasyon matrisi hesaplamalarını, yüksek korelasyon çifti filtrelemesini,
VIF çoklu doğrusallık alarmlarını ve PNG grafik üretimini doğrular.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Proje kök dizinini ekler
proje_kok = Path(__file__).resolve().parent.parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

from src.kesifci_analizor import KesifciVeriAnalizoru
from src.grafik_ureteci import EdaGrafikUreteci


def test_korelasyon_matrisi_ozellikleri():
    """Korelasyon matrisinin köşegeninin 1.0 ve simetrik olduğunu test eder."""
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(50, 3), columns=["x1", "x2", "x3"])
    analizor = KesifciVeriAnalizoru(df)
    rapor = analizor.korelasyon_analizi()

    matris = rapor.pearson_matrisi
    assert np.allclose(np.diag(matris), 1.0)
    assert np.allclose(matris.to_numpy(), matris.to_numpy().T)


def test_yuksek_korelasyon_cifti_tespiti():
    """Tam doğrusal bağımlı (y = 3x) iki sütunun alarm listesine girdiğini doğrular."""
    x = np.linspace(1, 100, 50)
    df = pd.DataFrame({
        "x": x,
        "y": 3.0 * x + np.random.normal(0, 0.1, 50),
        "bagimsiz": np.random.randn(50)
    })

    analizor = KesifciVeriAnalizoru(df)
    rapor = analizor.korelasyon_analizi(esik_degeri=0.80)

    assert len(rapor.yuksek_korelasyonlu_ciftler) >= 1
    cift = rapor.yuksek_korelasyonlu_ciftler[0]
    assert "x" in cift[:2] and "y" in cift[:2]
    assert cift[2] > 0.95


def test_vif_coklu_dogrusallik_alalarm():
    """Tam doğrusal bağımlı sütunda VIF değerinin çok yüksek çıktığını test eder."""
    np.random.seed(42)
    x1 = np.random.randn(100)
    x2 = np.random.randn(100)
    # x3, x1 ve x2'nin doğrudan toplamı (mükemmel çoklu doğrusallık)
    x3 = 2.0 * x1 + 3.0 * x2 + np.random.normal(0, 0.01, 100)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    analizor = KesifciVeriAnalizoru(df)
    vif_sonuclari = analizor.vif_analizi()

    vif_degerleri = {v.sutun_adi: v.vif_degeri for v in vif_sonuclari}
    # x3'ün VIF skoru 10'un çok üzerinde çıkmalıdır
    assert vif_degerleri["x3"] > 10.0


def test_hedef_degisken_analizi():
    """Hedef değişken ile ilişkilerin sözlük olarak doğru hesaplandığını kontrol eder."""
    df = pd.DataFrame({
        "ozellik_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "kategori": ["A", "A", "B", "B", "B"],
        "hedef": [2.0, 4.0, 6.0, 8.0, 10.0]  # Tam pozitif korelasyon r = 1.0
    })

    analizor = KesifciVeriAnalizoru(df)
    rapor = analizor.hedef_iliskisi_analizi(hedef_sutun="hedef")

    assert np.isclose(rapor.sayisal_korelasyonlar["ozellik_1"], 1.0)
    assert "kategori" in rapor.kategorik_dagilimlar


def test_grafik_uretimi_ve_png_olusturma(tmp_path):
    """Matplotlib çizim fonksiyonlarının diske fiziksel geçerli PNG dosyası yazdığını doğrular."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [10, 20, 30, 40]
    })
    hedef_png = tmp_path / "test_korelasyon.png"
    kor_matrisi = df.corr()

    cikti = EdaGrafikUreteci.korelasyon_isi_haritasi(kor_matrisi, hedef_png)
    assert cikti.exists()
    assert cikti.stat().st_size > 0
