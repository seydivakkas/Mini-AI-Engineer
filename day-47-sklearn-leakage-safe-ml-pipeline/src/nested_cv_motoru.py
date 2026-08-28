"""
İç-Dış Katmanlı Çapraz Doğrulama Motoru (Nested Cross-Validation Engine).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .pipeline_mimari import GuvenliPipelineUretici


class NestedCVMotoru:
    """Hiperparametre optimizasyonunu dış test katmanından izole eden Nested Cross-Validation motoru."""

    def __init__(self, outer_splits: int = 5, inner_splits: int = 3, random_state: int = 42):
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.random_state = random_state

    def nested_cv_yurut(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sayisal_kolonlar: List[str],
        kategorik_kolonlar: List[str],
        param_grid: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Güvenli Pipeline ile tam izole Nested Cross-Validation yürütür."""
        outer_cv = StratifiedKFold(n_splits=self.outer_splits, shuffle=True, random_state=self.random_state)
        inner_cv = StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=self.random_state)

        base_pipe = GuvenliPipelineUretici.guvenli_pipeline_olustur(
            sayisal_kolonlar=sayisal_kolonlar,
            kategorik_kolonlar=kategorik_kolonlar,
            model_turu="logistic",
            random_state=self.random_state
        )

        grid_search = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        outer_skorlar = []
        en_iyi_parametreler = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            # İç döngüde hiperparametre aranır (dış test katmanı asla görülmez!)
            grid_search.fit(X_tr, y_tr)
            best_model = grid_search.best_estimator_

            y_pred_proba = best_model.predict_proba(X_te)[:, 1]
            fold_auc = float(roc_auc_score(y_te, y_pred_proba))

            outer_skorlar.append(fold_auc)
            en_iyi_parametreler.append(grid_search.best_params_)

        return {
            "outer_skorlar": [float(round(s, 4)) for s in outer_skorlar],
            "ortalama_auc": float(round(np.mean(outer_skorlar), 4)),
            "std_auc": float(round(np.std(outer_skorlar), 4)),
            "en_iyi_parametreler": en_iyi_parametreler
        }

    def sizintili_karsilastirma_yurut(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sayisal_kolonlar: List[str]
    ) -> Dict[str, Any]:
        """Veri bölünmeden önce global ölçeklendirme yapılan sızıntılı senaryoyu simüle eder."""
        # YANLIŞ & SIZINTILI YAKLAŞIM: Tüm veri üzerinde fit_transform
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()

        X_sayisal = X[sayisal_kolonlar].copy()
        X_leaky = scaler.fit_transform(imputer.fit_transform(X_sayisal))

        model = LogisticRegression(random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.outer_splits, shuffle=True, random_state=self.random_state)
        leaky_skorlar = cross_val_score(model, X_leaky, y, cv=cv, scoring="roc_auc")

        return {
            "leaky_skorlar": [float(round(s, 4)) for s in leaky_skorlar],
            "leaky_ortalama_auc": float(round(np.mean(leaky_skorlar), 4)),
            "leaky_std_auc": float(round(np.std(leaky_skorlar), 4))
        }
