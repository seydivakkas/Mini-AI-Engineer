"""
Olasılık Kalibrasyonu ve Brier Skoru Hesaplayıcısı (Probability Calibration & ECE Engine).
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


class OlasilikKalibratoru:
    """Model tahmin olasılıklarının güvenilirliğini Brier Skoru ve ECE ile ölçer, izotonik kalibrasyon uygular."""

    @classmethod
    def kalibrasyon_analizi_yap(
        cls,
        y_gercek: np.ndarray,
        y_olasilik: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Brier Skoru, Güvenilirlik Eğrisi ve Beklenen Kalibrasyon Hatasını (ECE) hesaplar."""
        y_true = np.asarray(y_gercek, dtype=int)
        y_prob = np.asarray(y_olasilik, dtype=float)

        brier = float(round(brier_score_loss(y_true, y_prob), 4))

        # Kalibrasyon eğrisi hesaplama
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            bin_count = np.sum(mask)
            if bin_count > 0:
                acc = np.mean(y_true[mask])
                conf = np.mean(y_prob[mask])
                ece += (bin_count / n_total) * abs(acc - conf)

        return {
            "brier_skoru": brier,
            "ece_skoru": float(round(ece, 4)),
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "kalibrasyon_durumu": "MÜKEMMEL" if ece < 0.05 else "ORTA" if ece < 0.12 else "KALİBRE_EDİLMELİ"
        }

    @classmethod
    def izotonik_kalibre_et(
        cls,
        y_train_true: np.ndarray,
        y_train_prob: np.ndarray,
        y_test_prob: np.ndarray
    ) -> np.ndarray:
        """İzotonik regresyon ile test olasılıklarını monotonik olarak kalibre eder."""
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_train_prob, y_train_true)
        kalibre_prob = iso.predict(y_test_prob)
        return np.clip(kalibre_prob, 0.001, 0.999)
