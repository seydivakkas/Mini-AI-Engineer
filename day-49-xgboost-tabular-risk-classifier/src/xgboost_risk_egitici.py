"""
XGBoost Dengesiz Risk Sınıflandırıcısı ve TreeSHAP Açıklayıcısı (XGBoost Risk Classifier).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, confusion_matrix


class XGBoostRiskSiniflandirici:
    """scale_pos_weight, Erken Durdurma, Eşik Optimizasyonu ve TreeSHAP özellik katkısı sağlar."""

    def __init__(
        self,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        n_estimators: int = 250,
        early_stopping_rounds: int = 15,
        random_state: int = 42
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self.model: Optional[xgb.XGBClassifier] = None
        self.optimal_esik: float = 0.50
        self.scale_pos_weight: float = 1.0
        self.egitim_gecmisi: Dict[str, Any] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> "XGBoostRiskSiniflandirici":
        """scale_pos_weight ve validation PR-AUC ile erken durdurmalı eğitim yapar."""
        n_neg = int(np.sum(y_train == 0))
        n_pos = max(int(np.sum(y_train == 1)), 1)
        self.scale_pos_weight = float(round(n_neg / n_pos, 2))

        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric=["logloss", "aucpr"],
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            subsample=0.85,
            colsample_bytree=0.85
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        self.egitim_gecmisi = self.model.evals_result()
        return self

    def esik_degeri_optimize_et(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Validation seti üzerinde F1 skorunu maksimize eden optimal karar eşiğini (tau) bulur."""
        if self.model is None:
            raise RuntimeError("Model henüz eğitilmedi!")

        y_prob = self.model.predict_proba(X_val)[:, 1]
        esikler = np.linspace(0.1, 0.9, 81)
        f1_list = []

        for th in esikler:
            pred = (y_prob >= th).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            f1_list.append(f1)

        en_iyi_idx = int(np.argmax(f1_list))
        self.optimal_esik = float(round(esikler[en_iyi_idx], 3))

        return {
            "optimal_esik": self.optimal_esik,
            "en_iyi_val_f1": float(round(f1_list[en_iyi_idx], 4)),
            "esikler": esikler,
            "f1_skorlari": f1_list
        }

    def degerlendir(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        esik: Optional[float] = None
    ) -> Dict[str, Any]:
        """Test veri seti üzerinde ROC-AUC, PR-AUC, Confusion Matrix ve F1 skorunu hesaplar."""
        if self.model is None:
            raise RuntimeError("Model henüz eğitilmedi!")

        secili_esik = self.optimal_esik if esik is None else esik
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= secili_esik).astype(int)

        roc_auc = float(round(roc_auc_score(y_test, y_prob), 4))
        pr_auc = float(round(average_precision_score(y_test, y_prob), 4))
        f1 = float(round(f1_score(y_test, y_pred, zero_division=0), 4))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = float(round(tp / max(tp + fp, 1) * 100.0, 2))
        recall = float(round(tp / max(tp + fn, 1) * 100.0, 2))

        return {
            "esik": secili_esik,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_skoru": f1,
            "precision_yuzde": precision,
            "recall_yuzde": recall,
            "confusion_matrix": cm,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "y_prob": y_prob
        }

    def shap_katkilarini_cikar(self, X: pd.DataFrame) -> Dict[str, Any]:
        """XGBoost Booster'ın yerel TreeSHAP mekanizmasıyla öznitelik önemlilik matrisini çıkarır."""
        if self.model is None:
            raise RuntimeError("Model henüz eğitilmedi!")

        booster = self.model.get_booster()
        dmat = xgb.DMatrix(X)
        shap_matrisi = booster.predict(dmat, pred_contribs=True)

        # Son kolon bias terimidir, ilk D kolon özellik katkılarıdır
        shap_degerleri = shap_matrisi[:, :-1]
        ortalama_mutlak_shap = np.mean(np.abs(shap_degerleri), axis=0)

        kolonlar = list(X.columns)
        onem_sozlugu = {
            col: float(round(val, 4))
            for col, val in zip(kolonlar, ortalama_mutlak_shap)
        }

        # Azalan sıraya göre sırala
        sirali_onem = dict(sorted(onem_sozlugu.items(), key=lambda x: x[1], reverse=True))

        return {
            "shap_matrisi": shap_degerleri,
            "ortalama_mutlak_shap": sirali_onem
        }
