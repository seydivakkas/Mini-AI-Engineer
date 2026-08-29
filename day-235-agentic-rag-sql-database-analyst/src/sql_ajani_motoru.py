"""
SQL ve Veritabanı Analisti Ajan Motoru (Day 235 - FAZ 12).
Şema Bağlama (Schema Linking), Kendi Hatasını Düzelten SQL İcrası (DIN-SQL / Spider / BIRD).
"""

from typing import Dict, Any, List, Optional, Tuple
import sqlite3


class DatabaseSchema:
    """Veritabanı Şeması ve Tablo/Sütun Kataloğu."""

    def __init__(self, tablolar: Dict[str, List[str]]):
        self.tablolar = tablolar

    def sema_metni(self) -> str:
        metin = "VERİTABANI ŞEMASI:\n"
        for tablo, sutunlar in self.tablolar.items():
            metin += f"  • Tablo: {tablo} -> Sütunlar: [{', '.join(sutunlar)}]\n"
        return metin


class SQLQueryReport:
    """SQL Sorgu İcra ve Analiz Raporu."""

    def __init__(
        self,
        sql_sorgusu: str,
        basarili_mi: bool,
        donen_satirlar: List[Tuple[Any, ...]],
        sutun_adlari: List[str],
        hata_mesaji: Optional[str] = None,
        otonom_duzeltme_sayisi: int = 0,
        dogal_dil_icgorusu: str = "",
    ):
        self.sql_sorgusu = sql_sorgusu
        self.basarili_mi = basarili_mi
        self.donen_satirlar = donen_satirlar
        self.sutun_adlari = sutun_adlari
        self.hata_mesaji = hata_mesaji
        self.otonom_duzeltme_sayisi = otonom_duzeltme_sayisi
        self.dogal_dil_icgorusu = dogal_dil_icgorusu


class AgenticSQLAnalyst:
    """Doğal Dilden SQL Üreten ve SQLite Üzerinde Doğrulayan Otonom Analist."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self._ornek_veritabani_olustur()
        self.sema = DatabaseSchema(
            tablolar={
                "musteriler": ["musteri_id", "ad_soyad", "sehir", "kayit_yili"],
                "siparisler": ["siparis_id", "musteri_id", "tutar", "tarih", "durum"],
            }
        )

    def _ornek_veritabani_olustur(self):
        """Bellek içi örnek e-ticaret veritabanı kurar."""
        cursor = self.conn.cursor()
        cursor.execute(
            "CREATE TABLE musteriler (musteri_id INT, ad_soyad TEXT, sehir TEXT, kayit_yili INT)"
        )
        cursor.execute(
            "CREATE TABLE siparisler (siparis_id INT, musteri_id INT, tutar REAL, tarih TEXT, durum TEXT)"
        )

        cursor.executemany(
            "INSERT INTO musteriler VALUES (?, ?, ?, ?)",
            [
                (1, "Ahmet Yılmaz", "İstanbul", 2024),
                (2, "Ayşe Kaya", "Ankara", 2025),
                (3, "Mehmet Demir", "İzmir", 2026),
                (4, "Zeynep Çelik", "İstanbul", 2026),
            ],
        )

        cursor.executemany(
            "INSERT INTO siparisler VALUES (?, ?, ?, ?, ?)",
            [
                (101, 1, 15000.0, "2026-01-15", "Tamamlandı"),
                (102, 1, 8000.0, "2026-02-10", "Tamamlandı"),
                (103, 2, 4500.0, "2026-03-01", "Tamamlandı"),
                (104, 3, 32000.0, "2026-02-20", "Tamamlandı"),
                (105, 4, 18500.0, "2026-03-12", "Tamamlandı"),
            ],
        )
        self.conn.commit()

    def sorgula_ve_analiz_et(
        self,
        kullanici_sorusu: str,
        aday_sql_adimlari: List[str],
    ) -> SQLQueryReport:
        """SQL sorgularını koşturur; hata olursa otonom olarak sonraki adımla düzeltir."""
        cursor = self.conn.cursor()
        duzeltme_sayisi = 0
        nihai_sql = aday_sql_adimlari[0]
        satirlar: List[Tuple[Any, ...]] = []
        sutunlar: List[str] = []
        hata: Optional[str] = None
        basarili = False

        for adim, sql in enumerate(aday_sql_adimlari):
            nihai_sql = sql
            try:
                cursor.execute(sql)
                satirlar = cursor.fetchall()
                sutunlar = [d[0] for d in cursor.description] if cursor.description else []
                basarili = True
                hata = None
                duzeltme_sayisi = adim
                break
            except Exception as e:
                hata = str(e)
                basarili = False

        # Doğal Dil İçgörüsü Üretimi
        if basarili and len(satirlar) > 0:
            if len(satirlar[0]) > 1 and isinstance(satirlar[0][1], (int, float)):
                icgoru = (
                    f"Sorgu başarıyla icra edildi ({len(satirlar)} kayıt döndü). "
                    f"Lider kayıt '{satirlar[0][0]}' ({satirlar[0][1]:,.2f} TL)."
                )
            else:
                icgoru = f"Sorgu başarıyla icra edildi ({len(satirlar)} kayıt döndü). İlk kayıt: '{satirlar[0][0]}'."
        elif basarili:
            icgoru = "Sorgu başarıyla icra edildi fakat 0 kayıt döndü."
        else:
            icgoru = f"Sorgu çalıştırılamadı: {hata}"

        return SQLQueryReport(
            sql_sorgusu=nihai_sql,
            basarili_mi=basarili,
            donen_satirlar=satirlar,
            sutun_adlari=sutunlar,
            hata_mesaji=hata,
            otonom_duzeltme_sayisi=duzeltme_sayisi,
            dogal_dil_icgorusu=icgoru,
        )
