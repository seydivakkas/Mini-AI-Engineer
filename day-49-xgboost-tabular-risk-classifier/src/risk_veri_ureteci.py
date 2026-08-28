"""
Dengesiz Tabüler Risk ve Dolandırıcılık Veri Seti Simülasyonu (Imbalanced Risk Generator).
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class RiskVeriSimulasyonu:
    """Yüksek oranda dengesiz (%5 pozitif sınıf) finansal/endüstriyel risk veri seti üretir."""

    @classmethod
    def veri_seti_olustur(
        cls,
        n_orneklem: int = 2000,
        pozitif_oran: float = 0.05,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        np.random.seed(random_state)

        # 1. Özelliklerin Üretimi
        islem_tutari = np.random.lognormal(mean=7.5, sigma=1.1, size=n_orneklem)
        gece_islemi = np.random.poisson(lam=0.8, size=n_orneklem)
        hesap_yasi_gun = np.random.randint(10, 1800, size=n_orneklem)
        cihaz_degisimi = np.random.beta(a=1.5, b=4.0, size=n_orneklem)
        konum_mesafe_km = np.random.exponential(scale=45.0, size=n_orneklem)
        basarisiz_giris = np.random.poisson(lam=0.4, size=n_orneklem)
        harcama_hizi_z = np.random.normal(loc=0.0, scale=1.0, size=n_orneklem)

        # 2. Risk Olasılık Skoru (Nonlinear Logit Modeli)
        logit = (
            (islem_tutari / 3000.0) * 0.85
            + (gece_islemi * 0.70)
            - (hesap_yasi_gun / 400.0) * 0.60
            + (cihaz_degisimi * 2.20)
            + (konum_mesafe_km / 80.0) * 0.50
            + (basarisiz_giris * 1.10)
            + (harcama_hizi_z * 0.75)
            + np.random.normal(0, 0.4, size=n_orneklem)
        )

        # İstenen pozitif orana (%5) göre eşikleme
        esik_degeri = np.quantile(logit, 1.0 - pozitif_oran)
        hedef_etiket = (logit >= esik_degeri).astype(int)

        df = pd.DataFrame({
            "islem_tutari": np.round(islem_tutari, 2),
            "gece_islemi_sayisi": gece_islemi,
            "hesap_yasi_gun": hesap_yasi_gun,
            "cihaz_degisim_orani": np.round(cihaz_degisimi, 3),
            "konum_mesafe_km": np.round(konum_mesafe_km, 2),
            "basarisiz_giris_sayisi": basarisiz_giris,
            "harcama_hizi_zscore": np.round(harcama_hizi_z, 3)
        })

        return df, pd.Series(hedef_etiket, name="risk_etiketi")

    @classmethod
    def train_val_test_bol(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Stratified bölme ile sınıf oranını koruyarak %70 Train / %15 Val / %15 Test üretir."""
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
