"""
Day 65: SQLite AI Çıkarım Günlüğü ve Yönetim Paneli Birim Testleri.
"""

import pytest
import os
import pandas as pd

from src.veritabani_yoneticisi import AIVeritabaniYoneticisi
from src.analiz_motoru import AITelemetriAnalizci
from src.gorsellestirici import DashboardGorsellestirici


@pytest.fixture
def gecici_db(tmp_path):
    """Geçici bir SQLite test veritabanı örneği sağlar."""
    db_yolu = os.path.join(tmp_path, "test_ai.db")
    return AIVeritabaniYoneticisi(db_yolu=db_yolu)


def test_tablolari_olustur(gecici_db):
    """Veritabanı tablolarının başarıyla oluşturulduğunu doğrular."""
    with gecici_db._baglanti_al() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        assert "cikarim_loglari" in tables
        assert "nesne_tespitleri" in tables


def test_cikarim_ekle_ve_getir(gecici_db):
    """Çıkarım kaydı ekleme ve okuma işlemini test eder."""
    tespitler = [
        {"sinif_adi": "araba", "guven_skoru": 0.95, "kutu": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}},
        {"sinif_adi": "insan", "guven_skoru": 0.88, "kutu": {"x_min": 0.6, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9}}
    ]
    cikarim_id = gecici_db.cikarim_ekle(
        istek_id="req_test_001",
        model_adi="YOLOv8x-Vision",
        gorsel_meta={"genislik": 1920, "yukseklik": 1080, "format": "JPEG"},
        tespitler=tespitler,
        gecikme_ms=4.25,
        basarili=True
    )
    assert cikarim_id > 0

    df = gecici_db.cikarimlari_getir(limit=10)
    assert len(df) == 1
    assert df.iloc[0]["istek_id"] == "req_test_001"
    assert df.iloc[0]["tespit_sayisi"] == 2


def test_filtreleme_ve_sorgulama(gecici_db):
    """Model adına ve güven skoruna göre filtrelemeyi test eder."""
    AITelemetriAnalizci.sentetik_veri_doldur(gecici_db, kayit_sayisi=20)
    
    df_yolo = gecici_db.cikarimlari_getir(limit=50, model_adi="YOLOv8x-Vision")
    if not df_yolo.empty:
        assert (df_yolo["model_adi"] == "YOLOv8x-Vision").all()

    df_high_conf = gecici_db.cikarimlari_getir(limit=50, min_guven=0.80)
    if not df_high_conf.empty:
        assert (df_high_conf["ortalama_guven"] >= 0.80).all()


def test_insan_geri_bildirimi_guncelle(gecici_db):
    """İnsan denetimi (Human-in-the-Loop) etiket güncellemesini test eder."""
    tespitler = [{"sinif_adi": "araba", "guven_skoru": 0.90, "kutu": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}}]
    gecici_db.cikarim_ekle(
        istek_id="req_feedback_001",
        model_adi="YOLOv8x-Vision",
        gorsel_meta={"genislik": 1920, "yukseklik": 1080, "format": "JPEG"},
        tespitler=tespitler,
        gecikme_ms=3.12
    )

    guncellendi = gecici_db.geri_bildirim_guncelle(
        istek_id="req_feedback_001",
        dogru_mu=True,
        aciklama="Uzman denetçi tarafından onaylandı"
    )
    assert guncellendi is True

    df = gecici_db.cikarimlari_getir(limit=1)
    assert df.iloc[0]["insan_dogrulamasi"] == 1
    assert df.iloc[0]["aciklama"] == "Uzman denetçi tarafından onaylandı"


def test_cikarim_sil(gecici_db):
    """Çıkarım kaydı ve ilişkili tespitlerin silinmesini test eder."""
    tespitler = [{"sinif_adi": "araba", "guven_skoru": 0.90, "kutu": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}}]
    gecici_db.cikarim_ekle(
        istek_id="req_delete_001",
        model_adi="YOLOv8x-Vision",
        gorsel_meta={"genislik": 1920, "yukseklik": 1080, "format": "JPEG"},
        tespitler=tespitler,
        gecikme_ms=2.50
    )

    silindi = gecici_db.cikarim_sil("req_delete_001")
    assert silindi is True

    df = gecici_db.cikarimlari_getir(limit=10)
    assert len(df) == 0


def test_genel_istatistikler_ve_sinif_dagilimi(gecici_db):
    """Genel istatistiklerin ve sınıf dağılımının hesaplanmasını test eder."""
    AITelemetriAnalizci.sentetik_veri_doldur(gecici_db, kayit_sayisi=15)
    stats = gecici_db.genel_istatistikleri_al()
    assert stats["toplam_istek"] == 15
    assert stats["ortalama_gecikme_ms"] > 0.0

    df_sinif = AITelemetriAnalizci.sinif_dagilimi_al(gecici_db)
    assert not df_sinif.empty
    assert "sinif_adi" in df_sinif.columns
    assert "adet" in df_sinif.columns


def test_gorsellestirici_paneli(tmp_path, gecici_db):
    """6 panelli teşhis panosu çizimini test eder."""
    AITelemetriAnalizci.sentetik_veri_doldur(gecici_db, kayit_sayisi=25)
    stats = gecici_db.genel_istatistikleri_al()
    df_loglar = gecici_db.cikarimlari_getir(limit=25)
    df_siniflar = AITelemetriAnalizci.sinif_dagilimi_al(gecici_db)

    hedef = os.path.join(tmp_path, "test_dashboard_pano.png")
    cikis = DashboardGorsellestirici.panel_ciz(stats, df_loglar, df_siniflar, hedef_path=hedef)
    assert os.path.exists(cikis)
