"""
Sayısal Özellik Ölçekleyicileri ve Etkileşim Terimleri Üreticisi (Scalers & Interactions).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


class SayisalOlcekleyici:
    """Sayısal değişkenleri standartlaştırır, dayanıklı ölçekler ve etkileşim özellikleri türetir."""

    def __init__(self):
        self.standard_parametreleri: Dict[str, Tuple[float, float]] = {}
        self.robust_parametreleri: Dict[str, Tuple[float, float]] = {}

    def fit_standard_scaler(self, df: pd.DataFrame, kolonlar: List[str]) -> "SayisalOlcekleyici":
        """Z-Score Standardizasyon parametrelerini (Ortalama, Standart Sapma) hesaplar."""
        for col in kolonlar:
            mu = float(df[col].mean())
            sigma = float(df[col].std()) + 1e-8
            self.standard_parametreleri[col] = (mu, sigma)
        return self

    def transform_standard_scaler(self, df: pd.DataFrame, kolonlar: List[str]) -> pd.DataFrame:
        """Kayıtlı parametrelerle Z-Score normalizasyonu uygular."""
        sonuc_df = df.copy()
        for col in kolonlar:
            if col in self.standard_parametreleri:
                mu, sigma = self.standard_parametreleri[col]
                sonuc_df[f"{col}_std_scaled"] = (df[col] - mu) / sigma
        return sonuc_df

    def fit_robust_scaler(self, df: pd.DataFrame, kolonlar: List[str]) -> "SayisalOlcekleyici":
        """Outlier'lara dayanıklı Robust Scaler parametrelerini (Medyan, IQR) hesaplar."""
        for col in kolonlar:
            q25 = float(df[col].quantile(0.25))
            q50 = float(df[col].quantile(0.50))
            q75 = float(df[col].quantile(0.75))
            iqr = max(q75 - q25, 1e-6)
            self.robust_parametreleri[col] = (q50, iqr)
        return self

    def transform_robust_scaler(self, df: pd.DataFrame, kolonlar: List[str]) -> pd.DataFrame:
        """Kayıtlı Medyan ve IQR ile Robust normalizasyon uygular."""
        sonuc_df = df.copy()
        for col in kolonlar:
            if col in self.robust_parametreleri:
                q50, iqr = self.robust_parametreleri[col]
                sonuc_df[f"{col}_robust_scaled"] = (df[col] - q50) / iqr
        return sonuc_df

    @classmethod
    def log1p_donusumu(cls, df: pd.DataFrame, kolonlar: List[str]) -> pd.DataFrame:
        """Çarpık dağılımları log1p ile normale yaklaştırır."""
        sonuc_df = df.copy()
        for col in kolonlar:
            # Negatif değerleri 0'a kırpıp log1p uygula
            dizi = np.maximum(df[col].values, 0.0)
            sonuc_df[f"{col}_log1p"] = np.log1p(dizi)
        return sonuc_df

    @classmethod
    def etkilesim_ve_oran_uret(
        cls,
        df: pd.DataFrame,
        pay_kolon: str,
        payda_kolon: str,
        yeni_ad: str,
        eps: float = 1e-4
    ) -> pd.DataFrame:
        """İki sayısal değişken arasında oran (ratio) ve etkileşim özelliği türetir."""
        sonuc_df = df.copy()
        sonuc_df[yeni_ad] = df[pay_kolon] / (df[payda_kolon].abs() + eps)
        sonuc_df[f"{pay_kolon}_x_{payda_kolon}"] = df[pay_kolon] * df[payda_kolon]
        return sonuc_df
