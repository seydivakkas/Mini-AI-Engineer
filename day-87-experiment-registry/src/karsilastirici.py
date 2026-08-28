"""
Deney Karşılaştırıcı ve Liderlik Tablosu (Leaderboard) Motoru
------------------------------------------------------------
Farklı deney koşularının parametre-metrik matrisini çıkaran,
Pareto optimal sınırını hesaplayan ve modelleri derecelendiren modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class DeneyKarsilastirici:
    """
    Kayıtlı deney koşularını analiz eden ve karşılaştıran motor.
    """
    @staticmethod
    def karsilastirma_tablosu(kosular: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Tüm koşuların parametre ve nihai metriklerini birleştiren DataFrame üretir.
        """
        satirlar = []
        for k in kosular:
            satir = {
                "run_id": k["run_id"],
                "run_name": k["tags"].get("run_name", k["run_id"]),
                "durum": k["durum"],
                "sure_sn": round((k["bitis_zamani"] - k["baslangic_zamani"]), 2) if k["bitis_zamani"] else 0.0,
            }
            # Parametreler
            for p_k, p_v in k["params"].items():
                try:
                    satir[f"p_{p_k}"] = float(p_v) if "." in p_v or p_v.isdigit() else p_v
                except Exception:
                    satir[f"p_{p_k}"] = p_v

            # Metrikler
            for m_k, m_v in k["metrics"].items():
                satir[f"m_{m_k}"] = m_v

            satirlar.append(satir)

        df = pd.DataFrame(satirlar)
        if "m_val_acc" in df.columns:
            df = df.sort_values(by="m_val_acc", ascending=False).reset_index(drop=True)
        return df

    @staticmethod
    def pareto_optimal_noktalari(
        df: pd.DataFrame,
        x_kolon: str = "p_param_count",
        y_kolon: str = "m_val_acc"
    ) -> pd.DataFrame:
        """
        Doğruluk maksimizasyonu ve Parametre/Maliyet minimizasyonu için Pareto cephesini bulur.
        """
        if x_kolon not in df.columns or y_kolon not in df.columns:
            return pd.DataFrame()

        sirali = df.sort_values(by=[x_kolon, y_kolon], ascending=[True, False]).copy()
        pareto_indeksleri = []
        max_y = -float("inf")

        for idx, row in sirali.iterrows():
            if row[y_kolon] > max_y:
                pareto_indeksleri.append(idx)
                max_y = row[y_kolon]

        return df.loc[pareto_indeksleri].reset_index(drop=True)
