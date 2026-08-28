"""
Feature Store Profil Oluşturucu ve Metadata Kataloğu (Feature Store Registry & Profiler).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


class FeatureStoreProfilci:
    """Türetilen özelliklerin istatistiksel dağılımını, veri tiplerini ve metadata kataloğunu üretir."""

    @classmethod
    def profil_cikar(
        cls,
        df: pd.DataFrame,
        hedef_kolon: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tüm sütunlar için Feature Store profil ve metadata sözlüğünü oluşturur."""
        toplam_satir = len(df)
        kolon_profilleri = {}

        for col in df.columns:
            seri = df[col]
            dtype_str = str(seri.dtype)
            null_sayisi = int(seri.isna().sum())
            kardinalite = int(seri.nunique())

            stat_dict: Dict[str, Any] = {
                "dtype": dtype_str,
                "null_sayisi": null_sayisi,
                "null_orani": float(round(null_sayisi / max(toplam_satir, 1) * 100.0, 2)),
                "kardinalite": kardinalite,
                "tur": "sayisal" if pd.api.types.is_numeric_dtype(seri) else "kategorik"
            }

            if pd.api.types.is_numeric_dtype(seri):
                dolu = seri.dropna()
                if len(dolu) > 0:
                    stat_dict["min"] = float(round(dolu.min(), 3))
                    stat_dict["max"] = float(round(dolu.max(), 3))
                    stat_dict["ortalama"] = float(round(dolu.mean(), 3))
                    stat_dict["medyan"] = float(round(dolu.median(), 3))
                    stat_dict["std"] = float(round(dolu.std(), 3))
                    stat_dict["carpiklik_skew"] = float(round(stats.skew(dolu), 3))

                    if hedef_kolon and hedef_kolon in df.columns and col != hedef_kolon:
                        if pd.api.types.is_numeric_dtype(df[hedef_kolon]):
                            corr = float(dolu.corr(df[hedef_kolon]))
                            stat_dict["hedef_korelasyonu"] = float(round(corr, 4)) if not np.isnan(corr) else 0.0

            kolon_profilleri[col] = stat_dict

        return {
            "toplam_oznitelik_sayisi": len(df.columns),
            "toplam_satir_sayisi": toplam_satir,
            "sayisal_oznitelik_sayisi": sum(1 for p in kolon_profilleri.values() if p["tur"] == "sayisal"),
            "kategorik_oznitelik_sayisi": sum(1 for p in kolon_profilleri.values() if p["tur"] == "kategorik"),
            "oznitelikler": kolon_profilleri
        }

    @classmethod
    def feast_sema_ihrac_et(cls, profil_raporu: Dict[str, Any], feature_view_adi: str = "customer_risk_features") -> Dict[str, Any]:
        """Feast Feature Store formatında deklaratif YAML/JSON şema tanımı üretir."""
        feast_schema = {
            "name": feature_view_adi,
            "entities": ["musteri_id"],
            "features": []
        }
        for col, meta in profil_raporu["oznitelikler"].items():
            if col == "musteri_id":
                continue
            feast_type = "FLOAT" if meta["tur"] == "sayisal" else "STRING"
            feast_schema["features"].append({
                "name": col,
                "dtype": feast_type,
                "description": f"Engineered feature with cardinality {meta['kardinalite']}"
            })
        return feast_schema
