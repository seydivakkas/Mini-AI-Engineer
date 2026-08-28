"""CNN Model Eğitici ve Değerlendirici Modülü.

Keras modelinin EarlyStopping ve ReduceLROnPlateau callback'leriyle eğitimini,
tarihçe (history) kaydını ve test çıkarımını yönetir.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple
import keras
from keras import callbacks
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EgitimSonucu:
    """Model eğitim döngüsü ve test değerlendirme çıktıları."""

    model: keras.Model
    tarihce: Dict[str, list]
    test_kayip: float
    test_dogruluk: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    egitim_suresi_sn: float
    ornek_basina_gecikme_ms: float
    y_test_gercek: np.ndarray
    y_test_tahmin: np.ndarray
    y_test_olasiliklar: np.ndarray

    def ozet(self) -> str:
        """Sonuçların tek satırlık özetini döndürür."""
        return (
            f"Test Doğruluğu: %{self.test_dogruluk * 100:.2f} | "
            f"F1-Macro: {self.f1_macro:.4f} | "
            f"Precision: {self.precision_macro:.4f} | "
            f"Recall: {self.recall_macro:.4f} | "
            f"Eğitim Süresi: {self.egitim_suresi_sn:.2f}s | "
            f"Gecikme: {self.ornek_basina_gecikme_ms:.2f}ms/örnek"
        )


class ModelEgitici:
    """CNN model eğitim döngüsünü yöneten sınıf."""

    def __init__(self, model: keras.Model) -> None:
        """Eğiticiyi ilklendirir."""
        self.model = model

    def egit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 25,
        batch_size: int = 16,
        patience: int = 7,
    ) -> keras.callbacks.History:
        """Modeli belirtilen epoch ve erken durdurma (early stopping) ile eğitir."""
        cb_list = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        tarihce = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb_list,
            verbose=0,
        )

        return tarihce

    def degerlendir(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        tarihce: keras.callbacks.History,
        egitim_suresi_sn: float,
    ) -> EgitimSonucu:
        """Modeli test kümesinde kapsamlı metriklerle değerlendirir."""
        # Çıkarım süresi ölçümü
        t0 = time.perf_counter()
        olasiliklar = self.model.predict(X_test, verbose=0)
        t_cikarim = time.perf_counter() - t0
        gecikme_ms = (t_cikarim / len(X_test)) * 1000.0

        tahminler = np.argmax(olasiliklar, axis=1)

        test_kayip, test_acc = self.model.evaluate(X_test, y_test, verbose=0)

        f1_m = float(f1_score(y_test, tahminler, average="macro", zero_division=0))
        prec_m = float(precision_score(y_test, tahminler, average="macro", zero_division=0))
        rec_m = float(recall_score(y_test, tahminler, average="macro", zero_division=0))

        return EgitimSonucu(
            model=self.model,
            tarihce=tarihce.history,
            test_kayip=float(test_kayip),
            test_dogruluk=float(test_acc),
            f1_macro=f1_m,
            precision_macro=prec_m,
            recall_macro=rec_m,
            egitim_suresi_sn=egitim_suresi_sn,
            ornek_basina_gecikme_ms=gecikme_ms,
            y_test_gercek=y_test,
            y_test_tahmin=tahminler,
            y_test_olasiliklar=olasiliklar,
        )
