"""
Kategorik Özellik Kodlayıcıları (One-Hot, Frequency ve Smoothed Target Encoding).
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd


class KategorikKodlayici:
    """Kategorik değişkenleri makine öğrenimi modellerine uygun sayısal formatlara dönüştürür."""

    def __init__(self, smoothing_weight: float = 10.0):
        self.smoothing_weight = smoothing_weight
        self.freq_haritalari: Dict[str, Dict[str, float]] = {}
        self.target_haritalari: Dict[str, Dict[str, float]] = {}
        self.global_target_ortalamalari: Dict[str, float] = {}
        self.ohe_kategorileri: Dict[str, List[str]] = {}

    def fit_frequency_encoding(self, df: pd.DataFrame, kolonlar: List[str]) -> "KategorikKodlayici":
        """Frekans (Görülme Sıklığı) kodlaması istatistiklerini hesaplar."""
        for col in kolonlar:
            counts = df[col].astype(str).value_counts(normalize=True)
            self.freq_haritalari[col] = counts.to_dict()
        return self

    def transform_frequency_encoding(self, df: pd.DataFrame, kolonlar: List[str]) -> pd.DataFrame:
        """Kayıtlı frekans haritasına göre sütunları dönüştürür."""
        sonuc_df = df.copy()
        for col in kolonlar:
            if col in self.freq_haritalari:
                harita = self.freq_haritalari[col]
                yeni_kolon_adi = f"{col}_freq_enc"
                sonuc_df[yeni_kolon_adi] = df[col].astype(str).map(harita).fillna(0.0)
        return sonuc_df

    def fit_target_encoding(
        self,
        df: pd.DataFrame,
        kolonlar: List[str],
        hedef_kolon: str
    ) -> "KategorikKodlayici":
        """Düzeltilmiş (Smoothed Empirical Bayes) Hedef Kodlama istatistiklerini öğrenir."""
        y_global = float(df[hedef_kolon].mean())

        for col in kolonlar:
            self.global_target_ortalamalari[col] = y_global
            grup = df.groupby(col)[hedef_kolon].agg(["count", "mean"])
            
            # S(c) = (n_c * y_c + m * y_global) / (n_c + m)
            n_c = grup["count"]
            y_c = grup["mean"]
            m = self.smoothing_weight

            smoothed = (n_c * y_c + m * y_global) / (n_c + m)
            self.target_haritalari[col] = smoothed.to_dict()

        return self

    def transform_target_encoding(self, df: pd.DataFrame, kolonlar: List[str]) -> pd.DataFrame:
        """Kayıtlı hedef kodlama ağırlıklarını yeni verilere uygular."""
        sonuc_df = df.copy()
        for col in kolonlar:
            if col in self.target_haritalari:
                harita = self.target_haritalari[col]
                global_ort = self.global_target_ortalamalari[col]
                yeni_kolon_adi = f"{col}_target_enc"
                sonuc_df[yeni_kolon_adi] = df[col].map(harita).fillna(global_ort)
        return sonuc_df

    def fit_transform_one_hot(
        self,
        df: pd.DataFrame,
        kolonlar: List[str],
        drop_first: bool = False
    ) -> pd.DataFrame:
        """Düşük kardinaliteli kategoriler için One-Hot Encoding uygular."""
        ohe_df = pd.get_dummies(df[kolonlar], prefix=kolonlar, drop_first=drop_first, dtype=float)
        for col in kolonlar:
            self.ohe_kategorileri[col] = [c for c in ohe_df.columns if c.startswith(f"{col}_")]
        return pd.concat([df, ohe_df], axis=1)
