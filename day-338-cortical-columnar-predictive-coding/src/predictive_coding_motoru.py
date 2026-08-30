"""
Day 338: Cortical Column Architecture & Hierarchical Predictive Coding
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Kortikal Kolon Katmanını (L2/3 Hata Nöronları & L5/6 Üretken Durumlar),
Hiyerarşik Öngörücü Kodlama Ağını (V1 -> V2 -> V4) ve Serbest Enerji En Küçükleme Motorunu içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class CorticalColumnLayer:
    """
    Tekil Kortikal Kolon Katmanı.
    L2/3 Katmanı: Tahmin Hatasını (Prediction Error eps = y - y_hat) hesaplar.
    L5/6 Katmanı: Üst Seviye Durum Temsilini (Representation State r) ve Üretken Tahmini (y_hat = W * r) tutar.
    """
    def __init__(self, in_dim: int, state_dim: int, lr_state: float = 0.05, lr_weight: float = 0.01):
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.lr_state = lr_state
        self.lr_weight = lr_weight
        
        # Ağırlık matrisi W (girdi_boyutu x durum_boyutu)
        self.W = np.random.randn(in_dim, state_dim) * 0.1
        self.r = np.zeros(state_dim)  # Katman iç durumu
        self.error = np.zeros(in_dim)  # Tahmin hatası eps

    def forward_prediction(self) -> np.ndarray:
        """Yukarıdan aşağıya (Top-down) üretken tahmini hesaplar: y_hat = W * r"""
        return np.dot(self.W, self.r)

    def compute_error(self, y_input: np.ndarray) -> np.ndarray:
        """Tahmin hatasını hesaplar: eps = y_input - y_hat"""
        y_hat = self.forward_prediction()
        self.error = y_input - y_hat
        return self.error

    def update_state(self):
        """Serbest enerji türevine göre iç durum r güncellemesi: dr/dt = W^T * eps - gamma * r"""
        grad_r = np.dot(self.W.T, self.error) - 0.01 * self.r
        self.r += self.lr_state * grad_r

    def update_weights(self):
        """Hebbian öğrenme kuralı ile ağırlık güncellemesi: dW = eta * eps * r^T"""
        grad_W = np.outer(self.error, self.r)
        self.W += self.lr_weight * grad_W


class HierarchicalCorticalNetwork:
    """
    Hiyerarşik Kortikal Kolon Ağı (V1 -> V2 -> V4 Hiyerarşisi).
    """
    def __init__(self, layer_dims: List[int] = [64, 32, 16, 8]):
        self.layers: List[CorticalColumnLayer] = []
        for i in range(len(layer_dims) - 1):
            layer = CorticalColumnLayer(in_dim=layer_dims[i], state_dim=layer_dims[i+1])
            self.layers.append(layer)

    def infer_and_reconstruct(self, sensory_input: np.ndarray, n_steps: int = 30) -> Dict[str, Any]:
        """
        Duyusal girdi üzerinde yukarıdan aşağıya ve aşağıdan yukarıya çıkarım (Inference) yaparak hatayı en küçükler.
        """
        free_energy_history = []

        for step in range(n_steps):
            current_input = sensory_input.copy()
            total_energy = 0.0

            # 1. Aşağıdan Yukarıya (Bottom-Up) Hata Hesaplama
            for l_idx, layer in enumerate(self.layers):
                error = layer.compute_error(current_input)
                total_energy += 0.5 * np.sum(error ** 2)
                current_input = layer.r.copy()  # Bir üst katmanın girdisi mevcut durum r olur

            free_energy_history.append(float(total_energy))

            # 2. İç Durumların (r) Güncellenmesi
            for layer in self.layers:
                layer.update_state()

            # 3. Ağırlıkların (W) Güncellenmesi
            for layer in self.layers:
                layer.update_weights()

        # Rekonstrüksiyon tahmini (V1 katmanından üretilen girdi tahmini)
        reconstructed_input = self.layers[0].forward_prediction()

        return {
            "reconstructed_input": reconstructed_input,
            "free_energy_history": free_energy_history,
            "final_free_energy": free_energy_history[-1],
            "layer_errors": [layer.error for layer in self.layers],
            "layer_states": [layer.r for layer in self.layers],
        }


class FreeEnergyMinimizer:
    """
    Karl Friston Serbest Enerji En Küçükleme Metrikleri ve Çözümleyicisi.
    """
    @staticmethod
    def calculate_free_energy_reduction(initial_energy: float, final_energy: float) -> float:
        """Serbest enerji düşüş oranını (%) hesaplar."""
        if initial_energy <= 1e-9:
            return 100.0
        reduction = (1.0 - (final_energy / initial_energy)) * 100.0
        return max(0.0, float(reduction))
