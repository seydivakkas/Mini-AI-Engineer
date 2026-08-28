"""Model Olasılık Kalibrasyonu ve Sıcaklık Ölçekleme Modülü.

Bu modül; modelin ürettiği softmax güven skorlarının (Confidence) gerçek doğrulukla
uyumunu (Reliability) ölçen Beklenen Kalibrasyon Hatası (Expected Calibration Error - ECE),
Brier Skoru ve aşırı güveni düzelten Sıcaklık Ölçekleme (Temperature Scaling) optimizasyonunu sağlar.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F


class KalibrasyonAnalizcisi:
    """Olasılık kalibrasyonu analizi ve sıcaklık ölçekleme uygulayan sınıf."""

    @staticmethod
    def brier_skoru(y_true: np.ndarray, y_probs: np.ndarray, n_classes: int) -> float:
        """Çok sınıflı Brier Skorunu hesaplar (Düşük değer = İyi kalibrasyon)."""
        N = len(y_true)
        # One-hot matrisi oluştur
        y_one_hot = np.zeros((N, n_classes), dtype=np.float32)
        y_one_hot[np.arange(N), y_true] = 1.0

        brier = np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1))
        return float(brier)

    @staticmethod
    def kalibrasyon_egrisi_ve_ece(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        n_bins: int = 10,
    ) -> Dict:
        """Beklenen Kalibrasyon Hatası (ECE), Maksimum Kalibrasyon Hatası (MCE) ve Güvenilirlik Çizelgesi verilerini hesaplar."""
        y_pred = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies = (y_pred == y_true).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0
        bin_accs = []
        bin_confs = []
        bin_counts = []

        N = len(y_true)

        for lower, upper in zip(bin_lowers, bin_uppers):
            # Bin aralığına düşen örnekler
            in_bin = (confidences > lower) & (confidences <= upper)
            prop_in_bin = np.mean(in_bin)
            count = int(np.sum(in_bin))
            bin_counts.append(count)

            if count > 0:
                acc_in_bin = float(np.mean(accuracies[in_bin]))
                conf_in_bin = float(np.mean(confidences[in_bin]))
                gap = abs(acc_in_bin - conf_in_bin)

                ece += gap * (count / N)
                mce = max(mce, gap)

                bin_accs.append(acc_in_bin)
                bin_confs.append(conf_in_bin)
            else:
                bin_accs.append(0.0)
                bin_confs.append((lower + upper) / 2.0)

        return {
            "ece": float(ece),
            "mce": float(mce),
            "bin_accs": bin_accs,
            "bin_confs": bin_confs,
            "bin_counts": bin_counts,
            "bin_boundaries": bin_boundaries,
        }

    @staticmethod
    def sicaklik_olcekleme_optimize_et(
        y_true: np.ndarray, logits: np.ndarray, max_iter: int = 50
    ) -> float:
        """NLL (Negative Log Likelihood) kaybını minimize eden optimal T* sıcaklığını bulur (Guo et al., 2017)."""
        y_tensor = torch.from_numpy(y_true).long()
        logits_tensor = torch.from_numpy(logits).float()

        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            t_clamped = torch.clamp(temperature, min=0.01, max=10.0)
            loss = criterion(logits_tensor / t_clamped, y_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        optimal_T = float(temperature.item())
        return float(np.clip(optimal_T, 0.01, 10.0))

    @staticmethod
    def sicaklik_uygula(logits: np.ndarray, T: float) -> np.ndarray:
        """Logit tensörünü T sıcaklığı ile ölçekleyip yeni kalibre softmax olasılıklarını döner."""
        T_safe = max(1e-4, T)
        scaled = logits / T_safe
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
        return probs
