"""Pandas Tabüler Veri Temizleyici Birim Testleri.

Bu dosya; eksik veri doldurma doğruluğunu, IQR aykırı değer kırpmasını, mükerrer
kayıt temizliğini, bellek indirgemesini ve veri sızıntısı izolasyonunu test eder.
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

from src.veri_temizleyici import TabulerVeriTemizleyici
from src.sentetik_veri_ureticisi import kirli_veri_kumesi_uret


def test_fit_edilmeden_transform_cagrisi():
    """fit() çağrılmadan önce transform() çağrılırsa RuntimeError fırlatılmalıdır."""
    temizleyici = TabulerVeriTemizleyici()
    df = pd.DataFrame({"sayi": [1, 2, 3]})
    with pytest.raises(RuntimeError):
        temizleyici.transform(df)


def test_bos_veri_hatasi():
    """Boş DataFrame ile fit edilmeye çalışıldığında ValueError fırlatılmalıdır."""
    temizleyici = TabulerVeriTemizleyici()
    with pytest.raises(ValueError):
        temizleyici.fit(pd.DataFrame())


def test_eksik_veri_tamamlama_medyan_ve_mod():
    """Sayısal sütunların medyanla, kategorik sütunların modla doldurulduğunu test eder."""
    df = pd.DataFrame({
        "sayi": [10.0, 20.0, 30.0, np.nan],  # Medyan: 20.0
        "kategori": ["A", "B", "A", None]     # Mod: 'A'
    })

    temizleyici = TabulerVeriTemizleyici()
    temiz_df = temizleyici.fit_transform(df, sayisal_strateji="medyan", kategorik_strateji="mod")

    assert temiz_df["sayi"].isna().sum() == 0
    assert temiz_df["sayi"].iloc[3] == 20.0
    assert temiz_df["kategori"].isna().sum() == 0
    assert temiz_df["kategori"].iloc[3] == "A"


def test_aykiri_deger_budama_iqr():
    """Aykırı uç değerlerin belirlenen IQR sınırlarına kırpıldığını (clipping) doğrular."""
    # Normal değerler: 10, 11, 12, 13, 14, 15 + Aşırı aykırı: 9999, -500
    df = pd.DataFrame({
        "deger": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 9999.0, -500.0]
    })

    temizleyici = TabulerVeriTemizleyici()
    temiz_df = temizleyici.fit_transform(df, iqr_carpani=1.5, aykirilari_buda=True)

    # 9999 ve -500 değerleri kırpılmış olmalıdır
    assert temiz_df["deger"].max() < 100.0
    assert temiz_df["deger"].min() > 0.0


def test_yineleme_eleme():
    """Mükerrer satırların başarıyla temizlendiğini doğrular."""
    df = pd.DataFrame({
        "a": [1, 2, 2, 3],
        "b": ["x", "y", "y", "z"]
    })

    temizleyici = TabulerVeriTemizleyici()
    temiz_df = temizleyici.fit_transform(df, yinelemeleri_ele=True)

    assert len(temiz_df) == 3


def test_bellek_optimizasyonu_kazanci():
    """Tiplerin optimize edilerek bellek tüketiminin azaldığını doğrular."""
    veri = kirli_veri_kumesi_uret(satir_sayisi=300)
    temizleyici = TabulerVeriTemizleyici()
    temiz_veri = temizleyici.fit_transform(veri, tipleri_optimize_et=True)

    baslangic_kb = veri.memory_usage(deep=True).sum() / 1024.0
    bitis_kb = temiz_veri.memory_usage(deep=True).sum() / 1024.0

    assert bitis_kb < baslangic_kb


def test_veri_sizintisi_izolasyonu():
    """Eğitimden öğrenilen medyanın test kümesine aynen uygulandığını doğrular."""
    egitim = pd.DataFrame({"deger": [10.0, 20.0, 30.0]})  # Medyan = 20.0
    test = pd.DataFrame({"deger": [100.0, np.nan, 300.0]})

    temizleyici = TabulerVeriTemizleyici()
    temizleyici.fit(egitim)
    temiz_test = temizleyici.transform(test)

    # Test kümesindeki eksik değer, testin değil EĞİTİMİN medyanı olan 20.0 ile dolmalıdır
    assert temiz_test["deger"].iloc[1] == 20.0
