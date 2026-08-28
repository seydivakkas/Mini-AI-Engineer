"""Geleneksel Görsel Sınıflandırıcı Modülü.

Support Vector Machines (SVM) ve Random Forest algoritmalarını
StandardScaler ile veri sızıntısını (Data Leakage) önleyen Pipeline mimarisinde eğitir,
çapraz doğrulama yapar ve metrikleri raporlar.
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SiniflandiriciTipi(Enum):
    """Desteklenen klasik makine öğrenmesi model türleri."""

    SVM_RBF = "SVM (RBF Kernel)"
    SVM_LINEAR = "SVM (Linear Kernel)"
    RANDOM_FOREST = "Random Forest"


@dataclass
class ModelSonucu:
    """Modelin test değerlendirme çıktısı ve performans metrikleri."""

    model_adi: str
    pipeline: Pipeline
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    egitim_suresi_ms: float
    tahmin_suresi_ms: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    feature_importances: Optional[np.ndarray] = None

    def ozet(self) -> str:
        """Sonucun tek satırlık özetini basar."""
        return (
            f"[{self.model_adi}] Doğruluk (Acc): %{self.accuracy * 100:.2f} | "
            f"F1-Macro: {self.f1_macro:.4f} | "
            f"Precision: {self.precision_macro:.4f} | "
            f"Recall: {self.recall_macro:.4f} | "
            f"Eğitim: {self.egitim_suresi_ms:.1f}ms | Çıkarım: {self.tahmin_suresi_ms:.2f}ms"
        )


class GorselSiniflandirici:
    """SVM ve Random Forest görsel sınıflandırma modellerini yöneten sınıf."""

    def __init__(self, random_state: int = 42) -> None:
        """Sınıflandırıcı yöneticisini ilklendirir."""
        self.random_state = random_state

    def _pipeline_olustur(self, model_tipi: SiniflandiriciTipi, **kwargs: Any) -> Pipeline:
        """Veri sızıntısını önlemek için StandardScaler + Tahminci içeren Pipeline kurar."""
        if model_tipi == SiniflandiriciTipi.SVM_RBF:
            c_val = kwargs.get("C", 10.0)
            gamma_val = kwargs.get("gamma", "scale")
            estimator = SVC(
                C=c_val,
                kernel="rbf",
                gamma=gamma_val,
                probability=True,
                class_weight="balanced",
                random_state=self.random_state,
            )
        elif model_tipi == SiniflandiriciTipi.SVM_LINEAR:
            c_val = kwargs.get("C", 1.0)
            estimator = SVC(
                C=c_val,
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=self.random_state,
            )
        elif model_tipi == SiniflandiriciTipi.RANDOM_FOREST:
            n_est = kwargs.get("n_estimators", 150)
            max_d = kwargs.get("max_depth", None)
            estimator = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Bilinmeyen model tipi: {model_tipi}")

        return Pipeline([
            ("olceklendirici", StandardScaler()),
            ("siniflandirici", estimator),
        ])

    def egit_ve_degerlendir(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_tipi: SiniflandiriciTipi,
        **model_parametreleri: Any,
    ) -> ModelSonucu:
        """Modeli eğitim kümesinde eğitir ve test kümesinde değerlendirir."""
        pipeline = self._pipeline_olustur(model_tipi, **model_parametreleri)

        # Eğitim süresi ölçümü
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        t_egitim = (time.perf_counter() - t0) * 1000.0

        # Çıkarım süresi ve tahminler
        t1 = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        t_tahmin = (time.perf_counter() - t1) * 1000.0

        y_prob = None
        if hasattr(pipeline.named_steps["siniflandirici"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)

        # Öznitelik önemleri (Random Forest için)
        feature_imp = None
        if model_tipi == SiniflandiriciTipi.RANDOM_FOREST:
            rf_estimator: RandomForestClassifier = pipeline.named_steps["siniflandirici"]
            feature_imp = rf_estimator.feature_importances_

        # Metrikler
        acc = float(accuracy_score(y_test, y_pred))
        f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

        return ModelSonucu(
            model_adi=model_tipi.value,
            pipeline=pipeline,
            accuracy=acc,
            f1_macro=f1_m,
            f1_weighted=f1_w,
            precision_macro=prec,
            recall_macro=rec,
            egitim_suresi_ms=t_egitim,
            tahmin_suresi_ms=t_tahmin,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            feature_importances=feature_imp,
        )

    def capraz_dogrulama_yap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_tipi: SiniflandiriciTipi,
        k_kat: int = 5,
        **model_parametreleri: Any,
    ) -> Tuple[float, float]:
        """Stratified K-Fold ile veri sızıntısız çapraz doğrulama skorları döndürür."""
        pipeline = self._pipeline_olustur(model_tipi, **model_parametreleri)
        skf = StratifiedKFold(n_splits=k_kat, shuffle=True, random_state=self.random_state)
        skorlar = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
        return float(np.mean(skorlar)), float(np.std(skorlar))

    def en_iyi_hiperparametreleri_bul_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 3,
    ) -> Tuple[Dict[str, Any], float, Pipeline]:
        """GridSearchCV ile SVM için optimal C ve gamma değerlerini veri sızıntısız saptar.

        Args:
            X_train: Eğitim öznitelikleri matrisi.
            y_train: Eğitim etiketleri.
            param_grid: Taranacak parametre uzayı sözlüğü.
            cv: Çapraz doğrulama kat sayısı.

        Returns:
            Tuple[Dict[str, Any], float, Pipeline]: (En iyi parametreler, En iyi F1 skoru, En iyi eğitilmiş Pipeline).
        """
        if param_grid is None:
            param_grid = {
                "siniflandirici__C": [0.1, 1.0, 10.0, 50.0],
                "siniflandirici__gamma": ["scale", "auto", 0.01, 0.1],
            }
        base_pipeline = self._pipeline_olustur(SiniflandiriciTipi.SVM_RBF)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        grid = GridSearchCV(
            base_pipeline,
            param_grid=param_grid,
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return grid.best_params_, float(grid.best_score_), grid.best_estimator_
