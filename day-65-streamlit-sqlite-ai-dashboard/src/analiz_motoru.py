"""
AI Çıkarım Telemetrisi ve Sinyal Analiz Motoru.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .veritabani_yoneticisi import AIVeritabaniYoneticisi


class AITelemetriAnalizci:
    """Model çıkarım günlüğü üzerinde istatistiksel trend, anomali ve kayan pencere analizleri yapar."""

    @staticmethod
    def sentetik_veri_doldur(db_yoneticisi: AIVeritabaniYoneticisi, kayit_sayisi: int = 200) -> None:
        """Veritabanını test ve analiz için gerçekçi çıkarım verileriyle doldurur."""
        modeller = ["YOLOv8x-Vision", "MiniViT-Embedder", "Defect-Detector-V2"]
        siniflar = ["araba", "insan", "bisiklet", "kusur_dokuma", "kusur_leke"]

        for i in range(kayit_sayisi):
            req_id = f"trace_{i:05d}"
            model = np.random.choice(modeller)
            n_det = np.random.randint(1, 4)
            tespitler = []
            for _ in range(n_det):
                cls_name = np.random.choice(siniflar)
                conf = float(np.random.beta(a=8, b=2))  # [0.6, 0.99] arası gerçekçi güven dağılımı
                tespitler.append({
                    "sinif_adi": cls_name,
                    "guven_skoru": round(conf, 4),
                    "kutu": {
                        "x_min": 0.1, "y_min": 0.1,
                        "x_max": 0.5, "y_max": 0.5
                    }
                })

            gecikme = float(np.random.normal(loc=1.8, scale=0.4))
            db_yoneticisi.cikarim_ekle(
                istek_id=req_id,
                model_adi=model,
                gorsel_meta={"genislik": 1920, "yukseklik": 1080, "format": "JPEG"},
                tespitler=tespitler,
                gecikme_ms=max(0.2, round(gecikme, 2)),
                basarili=True
            )

    @staticmethod
    def sinif_dagilimi_al(db_yoneticisi: AIVeritabaniYoneticisi) -> pd.DataFrame:
        """Nesne tespiti sınıflarının frekans dağılımını SQL üzerinden çeker."""
        with db_yoneticisi._baglanti_al() as conn:
            query = """
                SELECT sinif_adi, COUNT(*) as adet, AVG(guven_skoru) as ortalama_guven
                FROM nesne_tespitleri
                GROUP BY sinif_adi
                ORDER BY adet DESC
            """
            return pd.read_sql_query(query, conn)

    @staticmethod
    def anomali_ve_dusuk_guven_filtrele(df: pd.DataFrame, esik: float = 0.65) -> pd.DataFrame:
        """İnceleme gerektiren düşük güvenli veya şüpheli çıkarımları filtreler."""
        if df.empty:
            return df
        return df[df["ortalama_guven"] < esik]
