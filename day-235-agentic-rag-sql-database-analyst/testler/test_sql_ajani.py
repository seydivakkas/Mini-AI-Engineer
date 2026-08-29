"""
PyTest Birim Testleri - Day 235: SQL ve Veritabanı Analisti Ajan Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sql_ajani_motoru import (
    DatabaseSchema,
    SQLQueryReport,
    AgenticSQLAnalyst,
)
from src.sql_profilleyici import SQLProfilleyici
from src.gorsellestirici import SQLGorsellestirici


def test_database_schema_formatting():
    """1. DatabaseSchema nesnesi şema metnini doğru formatlamalıdır."""
    sema = DatabaseSchema({"users": ["id", "email"]})
    assert "users" in sema.sema_metni()
    assert "email" in sema.sema_metni()


def test_sql_query_report_initialization():
    """2. SQLQueryReport alanları doğru başlatmalıdır."""
    rep = SQLQueryReport("SELECT 1", True, [(1,)], ["col"], None, 0, "İçgörü")
    assert rep.basarili_mi is True
    assert rep.donen_satirlar == [(1,)]


def test_agent_database_initialization():
    """3. AgenticSQLAnalyst SQLite bellek içi tabloları doldurmalıdır."""
    analyst = AgenticSQLAnalyst()
    cur = analyst.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM musteriler")
    count = cur.fetchone()[0]
    assert count == 4


def test_agent_successful_query_execution():
    """4. AgenticSQLAnalyst geçerli SQL sorgusunu başarıyla çalıştırmalıdır."""
    analyst = AgenticSQLAnalyst()
    rep = analyst.sorgula_ve_analiz_et("Tüm müşteriler", ["SELECT ad_soyad FROM musteriler"])
    assert rep.basarili_mi is True
    assert len(rep.donen_satirlar) == 4


def test_agent_self_correction_on_error():
    """5. AgenticSQLAnalyst hatalı sorguyu bir sonraki düzeltilmiş sorguyla onarmalıdır."""
    analyst = AgenticSQLAnalyst()
    hatali = "SELECT bozuk_sutun FROM musteriler"
    dogru = "SELECT ad_soyad FROM musteriler"
    rep = analyst.sorgula_ve_analiz_et("Müşteri listesi", [hatali, dogru])
    assert rep.basarili_mi is True
    assert rep.otonom_duzeltme_sayisi == 1


def test_agent_syntax_error_handling():
    """6. AgenticSQLAnalyst düzeltilemeyen sorguda basarili_mi=False dönmelidir."""
    analyst = AgenticSQLAnalyst()
    rep = analyst.sorgula_ve_analiz_et("Geçersiz sorgu", ["SELECT * FORM invalid syntax"])
    assert rep.basarili_mi is False
    assert rep.hata_mesaji is not None


def test_profiler_sql_metrics():
    """7. Profilleyici Agentic SQL analisti başarısının %90 üzerinde olduğunu doğrulamalıdır."""
    prof = SQLProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["karmasik_sql_basarisi"]["Agentic_SQL_Analisti"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. SQLGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_sql_paneli.png")
    profil = SQLProfilleyici.basarim_profili_cikar()

    SQLGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
