"""
SQLite Destekli Kalıcı AI Çıkarım Günlüğü ve CRUD Yönetim Motoru.
"""

from typing import List, Dict, Any, Optional, Tuple
import sqlite3
import os
import json
import pandas as pd


class AIVeritabaniYoneticisi:
    """Yapay zeka çıkarım logları, nesne tespitleri ve insan geri bildirimlerini yöneten SQLite CRUD motoru."""

    def __init__(self, db_yolu: str = "ai_yonetim.db"):
        self.db_yolu = db_yolu
        dizin = os.path.dirname(os.path.abspath(db_yolu))
        os.makedirs(dizin, exist_ok=True)
        self.tablolari_olustur()

    def _baglanti_al(self) -> sqlite3.Connection:
        """WAL (Write-Ahead Logging) modunda optimize edilmiş veritabanı bağlantısı açar."""
        conn = sqlite3.connect(self.db_yolu)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def tablolari_olustur(self) -> None:
        """İlişkisel tabloları ve B-Tree indekslerini oluşturur."""
        with self._baglanti_al() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS cikarim_loglari (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                istek_id TEXT UNIQUE NOT NULL,
                model_adi TEXT NOT NULL,
                gorsel_genislik INTEGER DEFAULT 1920,
                gorsel_yukseklik INTEGER DEFAULT 1080,
                gorsel_format TEXT DEFAULT 'JPEG',
                tespit_sayisi INTEGER DEFAULT 0,
                ortalama_guven REAL DEFAULT 0.0,
                gecikme_ms REAL DEFAULT 0.0,
                basarili INTEGER DEFAULT 1,
                olusturma_tarihi TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                insan_dogrulamasi INTEGER DEFAULT NULL,
                aciklama TEXT DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS nesne_tespitleri (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cikarim_id INTEGER NOT NULL,
                sinif_adi TEXT NOT NULL,
                guven_skoru REAL NOT NULL,
                x_min REAL NOT NULL,
                y_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_max REAL NOT NULL,
                FOREIGN KEY (cikarim_id) REFERENCES cikarim_loglari(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_cikarim_istek_id ON cikarim_loglari(istek_id);
            CREATE INDEX IF NOT EXISTS idx_cikarim_model ON cikarim_loglari(model_adi);
            CREATE INDEX IF NOT EXISTS idx_cikarim_tarih ON cikarim_loglari(olusturma_tarihi);
            CREATE INDEX IF NOT EXISTS idx_tespit_sinif ON nesne_tespitleri(sinif_adi);
            """)

    def cikarim_ekle(
        self,
        istek_id: str,
        model_adi: str,
        gorsel_meta: Dict[str, Any],
        tespitler: List[Dict[str, Any]],
        gecikme_ms: float,
        basarili: bool = True
    ) -> int:
        """Yeni bir model çıkarım olayını ve tespit edilen nesneleri atomik (transaction) olarak kaydeder."""
        tespit_sayisi = len(tespitler)
        ortalama_guven = float(sum(t.get("guven_skoru", 0.0) for t in tespitler) / max(tespit_sayisi, 1))

        with self._baglanti_al() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cikarim_loglari (
                    istek_id, model_adi, gorsel_genislik, gorsel_yukseklik,
                    gorsel_format, tespit_sayisi, ortalama_guven, gecikme_ms, basarili
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                istek_id,
                model_adi,
                gorsel_meta.get("genislik", 1920),
                gorsel_meta.get("yukseklik", 1080),
                gorsel_meta.get("format", "JPEG"),
                tespit_sayisi,
                ortalama_guven,
                gecikme_ms,
                1 if basarili else 0
            ))
            cikarim_id = cursor.lastrowid

            for t in tespitler:
                kutu = t.get("kutu", {})
                cursor.execute("""
                    INSERT INTO nesne_tespitleri (
                        cikarim_id, sinif_adi, guven_skoru, x_min, y_min, x_max, y_max
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    cikarim_id,
                    t.get("sinif_adi", "bilinmeyen"),
                    float(t.get("guven_skoru", 0.0)),
                    float(kutu.get("x_min", 0.0)),
                    float(kutu.get("y_min", 0.0)),
                    float(kutu.get("x_max", 1.0)),
                    float(kutu.get("y_max", 1.0))
                ))
            conn.commit()
            return cikarim_id

    def cikarimlari_getir(
        self,
        limit: int = 100,
        model_adi: Optional[str] = None,
        min_guven: Optional[float] = None
    ) -> pd.DataFrame:
        """Filtrelere uygun çıkarım kayıtlarını Pandas DataFrame olarak çeker."""
        sorgu = "SELECT * FROM cikarim_loglari WHERE 1=1"
        parametreler = []

        if model_adi:
            sorgu += " AND model_adi = ?"
            parametreler.append(model_adi)
        if min_guven is not None:
            sorgu += " AND ortalama_guven >= ?"
            parametreler.append(min_guven)

        sorgu += " ORDER BY id DESC LIMIT ?"
        parametreler.append(limit)

        with self._baglanti_al() as conn:
            df = pd.read_sql_query(sorgu, conn, params=parametreler)
            return df

    def geri_bildirim_guncelle(self, istek_id: str, dogru_mu: bool, aciklama: Optional[str] = None) -> bool:
        """İnsan denetçinin (Human-in-the-Loop) model çıkarımı için verdiği doğruluk etiketini günceller."""
        with self._baglanti_al() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE cikarim_loglari
                SET insan_dogrulamasi = ?, aciklama = ?
                WHERE istek_id = ?
            """, (1 if dogru_mu else 0, aciklama, istek_id))
            conn.commit()
            return cursor.rowcount > 0

    def cikarim_sil(self, istek_id: str) -> bool:
        """Belirtilen çıkarım kaydını ve bağlı nesne tespitlerini siler."""
        with self._baglanti_al() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cikarim_loglari WHERE istek_id = ?", (istek_id,))
            conn.commit()
            return cursor.rowcount > 0

    def genel_istatistikleri_al(self) -> Dict[str, Any]:
        """Gösterge paneli KPI kartları için özet metrikleri hesaplar."""
        with self._baglanti_al() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as toplam_istek,
                    COALESCE(AVG(gecikme_ms), 0.0) as ortalama_gecikme,
                    COALESCE(AVG(ortalama_guven), 0.0) as ortalama_guven,
                    COALESCE(SUM(tespit_sayisi), 0) as toplam_tespit,
                    COALESCE(SUM(CASE WHEN insan_dogrulamasi = 1 THEN 1 ELSE 0 END), 0) as dogrulanan_adet,
                    COALESCE(SUM(CASE WHEN insan_dogrulamasi = 0 THEN 1 ELSE 0 END), 0) as yanlis_adet
                FROM cikarim_loglari
            """)
            row = cursor.fetchone()
            return {
                "toplam_istek": row["toplam_istek"],
                "ortalama_gecikme_ms": round(row["ortalama_gecikme"], 2),
                "ortalama_guven": round(row["ortalama_guven"], 4),
                "toplam_tespit": row["toplam_tespit"],
                "dogrulanan_adet": row["dogrulanan_adet"],
                "yanlis_adet": row["yanlis_adet"]
            }
