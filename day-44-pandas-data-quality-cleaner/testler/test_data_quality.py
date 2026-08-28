"""
Day 44: Pandas ile Üretim Seviyesi Şema Doğrulama ve Veri Kalitesi Birim Testleri.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.sema import KolonKurali, TabloSemasi
from src.dogrulayici import SemaDogrulayici
from src.temizleyici import OtomatikVeriTemizleyici
from src.gorsellestirici import VeriKaliteGorsellestirici


@pytest.fixture
def ornek_sema():
    return TabloSemasi(
        tablo_adi="TestTablo",
        kolon_kurallari=[
            KolonKurali(ad="id", tip=int, zorunlu=True, benzersiz=True),
            KolonKurali(ad="puan", tip=float, zorunlu=True, min_deger=0.0, max_deger=100.0, varsayilan_doldurma="median"),
            KolonKurali(ad="durum", tip=str, zorunlu=True, kategoriler=["AKTIF", "PASIF"]),
            KolonKurali(ad="kod", tip=str, zorunlu=False, regex_kalibi=r"^TR-[0-9]{3}$")
        ]
    )


def test_kusursuz_tablo_dogrulama(ornek_sema):
    """Kusursuz tablonun onaylandığını test eder."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "puan": [45.0, 78.5, 92.0],
        "durum": ["AKTIF", "PASIF", "AKTIF"],
        "kod": ["TR-101", "TR-102", "TR-103"]
    })
    dogrulayici = SemaDogrulayici(ornek_sema)
    rapor = dogrulayici.dogrula(df)

    assert rapor["karar"] == "GECERLI_MUKEMMEL"
    assert rapor["kalite_skoru"] == 100.0
    assert rapor["toplam_ihlal_sayisi"] == 0


def test_eksik_kolon_kritik_red(ornek_sema):
    """Zorunlu kolon eksik olduğunda tablonun KRITIK_RED aldığını test eder."""
    df = pd.DataFrame({
        "puan": [50.0],
        "durum": ["AKTIF"]
    })
    dogrulayici = SemaDogrulayici(ornek_sema)
    rapor = dogrulayici.dogrula(df)

    assert rapor["karar"] == "KRITIK_RED"
    assert any(ih["kod"] == "EKSIK_ZORUNLU_KOLON" for ih in rapor["ihlaller"])


def test_aralik_disi_deger_tespiti(ornek_sema):
    """Aralık dışı değerlerin yakalanıp kırpıldığını test eder."""
    df = pd.DataFrame({
        "id": [1, 2],
        "puan": [-15.0, 140.0],  # Aralık [0, 100]
        "durum": ["AKTIF", "AKTIF"],
        "kod": ["TR-101", "TR-102"]
    })
    dogrulayici = SemaDogrulayici(ornek_sema)
    temizleyici = OtomatikVeriTemizleyici(ornek_sema)

    rapor = dogrulayici.dogrula(df)
    assert any(ih["kod"] == "ARALIK_DISI_DEGER" for ih in rapor["ihlaller"])

    temiz_df, _ = temizleyici.temizle_ve_iyilestir(df)
    assert float(temiz_df["puan"].min()) >= 0.0
    assert float(temiz_df["puan"].max()) <= 100.0


def test_kategorik_gecersizlik_ve_duzeltme(ornek_sema):
    """Geçersiz kategorilerin tespit edilip düzeltildiğini test eder."""
    df = pd.DataFrame({
        "id": [1, 2],
        "puan": [50.0, 60.0],
        "durum": ["AKTIF", "BILINMEYEN_DURUM"],
        "kod": ["TR-101", "TR-102"]
    })
    dogrulayici = SemaDogrulayici(ornek_sema)
    temizleyici = OtomatikVeriTemizleyici(ornek_sema)

    rapor = dogrulayici.dogrula(df)
    assert any(ih["kod"] == "GECERSIZ_KATEGORI" for ih in rapor["ihlaller"])

    temiz_df, _ = temizleyici.temizle_ve_iyilestir(df)
    assert all(d in ["AKTIF", "PASIF"] for d in temiz_df["durum"])


def test_regex_desen_kontrolu(ornek_sema):
    """RegEx formatına uymayan verilerin yakalandığını test eder."""
    df = pd.DataFrame({
        "id": [1],
        "puan": [50.0],
        "durum": ["AKTIF"],
        "kod": ["GECERSIZ-KOD-XYZ"]
    })
    dogrulayici = SemaDogrulayici(ornek_sema)
    rapor = dogrulayici.dogrula(df)

    assert any(ih["kod"] == "REGEX_DESEN_IHLALI" for ih in rapor["ihlaller"])


def test_otomatik_temizleyici_null_impütasyonu(ornek_sema):
    """Null değerlerin başarıyla medyan ile doldurulduğunu test eder."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "puan": [10.0, np.nan, 30.0],
        "durum": ["AKTIF", "AKTIF", "AKTIF"],
        "kod": ["TR-101", "TR-102", "TR-103"]
    })
    temizleyici = OtomatikVeriTemizleyici(ornek_sema)
    temiz_df, rapor = temizleyici.temizle_ve_iyilestir(df)

    assert temiz_df["puan"].isna().sum() == 0
    assert temiz_df.loc[1, "puan"] == 20.0  # Medyan (10 ve 30'un ortası)


def test_gorsellestirici_png_olusturma(ornek_sema, tmp_path):
    """6 panelli teşhis panosunun başarıyla PNG ürettiğini test eder."""
    df = pd.DataFrame({
        "id": [1, 2],
        "puan": [50.0, 60.0],
        "durum": ["AKTIF", "PASIF"],
        "kod": ["TR-101", "TR-102"]
    })
    dogrulayici = SemaDogrulayici(ornek_sema)
    rapor = dogrulayici.dogrula(df)

    cikis_yolu = str(tmp_path / "test_kalite_paneli.png")
    yol = VeriKaliteGorsellestirici.panel_ciz(df, df, rapor, hedef_path=cikis_yolu)
    assert os.path.exists(yol)
