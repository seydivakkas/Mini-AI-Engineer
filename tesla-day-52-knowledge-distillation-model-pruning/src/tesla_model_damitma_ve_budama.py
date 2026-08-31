r"""
Tesla Model Damıtma (Knowledge Distillation) ve Yapısal Budama Çekirdeği
========================================================================
Bu modül; Devasa Dojo Öğretmen (Teacher) modelinden Kompakt HW3/HW4 Öğrenci
(Student) modeline Sıcaklık Yumuşatmalı ($T$) Bilgi Damıtmayı ve L1-Norm
tabanlı Yapısal Kanalsal Budamayı (%30 Pruning) gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaKnowledgeDistiller:
    """
    Teacher-Student Bilgi Damıtma ve Budama Motoru.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        self.T = temperature
        self.alpha = alpha

    def compute_soft_probabilities(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        Sıcaklık Yumuşatmalı Softmax:
        p_i(z, T) = exp(z_i / T) / sum(exp(z_j / T))
        """
        scaled_logits = logits / temperature
        exp_vals = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def compute_distillation_loss(
        self,
        teacher_logits: np.ndarray,
        student_logits: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        L_KD = alpha * T^2 * KL(p_T || p_S) + (1 - alpha) * CrossEntropy(y, p_S)
        """
        p_T = self.compute_soft_probabilities(teacher_logits, self.T)
        p_S = self.compute_soft_probabilities(student_logits, self.T)
        p_S_hard = self.compute_soft_probabilities(student_logits, temperature=1.0)

        # KL-Divergence: sum( p_T * log(p_T / p_S) )
        kl_div = float(np.sum(p_T * np.log(np.maximum(p_T, 1e-12) / np.maximum(p_S, 1e-12))))
        loss_soft = (self.T ** 2) * kl_div

        # Cross Entropy (Hard labels)
        ce_loss = float(-np.sum(true_labels * np.log(np.maximum(p_S_hard, 1e-12))))

        total_loss = self.alpha * loss_soft + (1.0 - self.alpha) * ce_loss

        return {
            "total_loss": total_loss,
            "loss_soft_kd": loss_soft,
            "loss_hard_ce": ce_loss,
            "kl_divergence": kl_div
        }

    def prune_channels_l1_norm(self, conv_weights: np.ndarray, prune_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        L1-Norm Bazlı Yapısal Kanal Budama (Structured Channel Pruning):
        En düşük ||W_c||_1 değerine sahip kanallar maskelenir.
        conv_weights: (C_out, C_in, K, K)
        """
        # Her çıkış kanalı için L1 normu
        c_out = conv_weights.shape[0]
        channel_norms = np.sum(np.abs(conv_weights.reshape(c_out, -1)), axis=1)

        k_prune = int(c_out * prune_ratio)
        sorted_indices = np.argsort(channel_norms)
        pruned_channel_indices = sorted_indices[:k_prune]

        mask = np.ones(c_out, dtype=bool)
        mask[pruned_channel_indices] = False

        pruned_weights = conv_weights.copy()
        pruned_weights[~mask] = 0.0

        actual_sparsity = float(k_prune / c_out)
        return pruned_weights, mask, actual_sparsity
