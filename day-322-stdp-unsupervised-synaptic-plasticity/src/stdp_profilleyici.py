"""
Day 322: Spike-Timing-Dependent Plasticity (STDP) & Unsupervised Learning
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; STDP plastisite öğrenme sürecinde sinaptik ağırlık kaymasını (drift),
bimodal kutupsallaşma indeksini ve WTA uzmanlaşma metriklerini hesaplar.
"""

from typing import Dict, Any
import torch
import numpy as np


class STDPProfilleyici:
    """
    STDP Plastisite ve Denetimsiz Öğrenme Profilleyicisi.
    """
    @staticmethod
    def profille(
        initial_weights: np.ndarray,
        final_weights: np.ndarray,
        spikes_seq: torch.Tensor
    ) -> Dict[str, Any]:
        """
        STDP plastisite değişim metriklerini ve WTA rekabet verimliliğini profiller.
        """
        # 1. Ağırlık Kayması (Plasticity Drift)
        weight_diff = np.abs(final_weights - initial_weights)
        mean_drift = float(np.mean(weight_diff))
        max_drift = float(np.max(weight_diff))

        # 2. Bimodal Kutupsallaşma İndeksi (Bimodality Index)
        # STDP tam eğitildiğinde ağırlıklar 0 (LTD) veya 1 (LTP) uçlarına çekilir.
        dist_to_bounds = np.minimum(final_weights - 0.0, 1.0 - final_weights)
        bimodality_score = float(1.0 - 2.0 * np.mean(dist_to_bounds))

        # 3. Ağırlık Entropisi (Shannon Entropy)
        hist, _ = np.histogram(final_weights, bins=10, range=(0.0, 1.0), density=True)
        hist_norm = hist / (np.sum(hist) + 1e-9)
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-9)))

        # 4. Winner-Take-All Uzmanlaşma Oranı
        spikes_np = spikes_seq.cpu().numpy()  # (Batch, T, Out)
        total_spikes_per_neuron = np.sum(spikes_np, axis=(0, 1))
        active_neurons = int(np.sum(total_spikes_per_neuron > 0))
        total_neurons = len(total_spikes_per_neuron)
        specialization_ratio = float(active_neurons / (total_neurons + 1e-9))

        return {
            "mean_weight_drift": mean_drift,
            "max_weight_drift": max_drift,
            "bimodality_score": bimodality_score,
            "weight_entropy": entropy,
            "active_neurons_count": active_neurons,
            "total_neurons_count": total_neurons,
            "specialization_ratio": specialization_ratio,
        }
