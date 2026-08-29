"""
Day 282 (FAZ 15): Meta-Learning (MAML & Meta-SGD) Motoru.
Model-Agnostic Meta-Learning ve İç/Dış Döngülü Hızlı Görev Adaptasyonu.
"""

from typing import Dict, Any, Tuple, List
import numpy as np


class MetaTask:
    """Few-Shot Meta-Öğrenme Görevi (Sinüzoid veya Sınıflandırma)."""
    def __init__(self, amplitude: float, phase: float):
        self.amplitude = amplitude
        self.phase = phase

    def sample_data(self, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Girdiye karşılık gelen hedef değerleri üretir."""
        x = np.random.uniform(-5.0, 5.0, size=(num_points, 1))
        y = self.amplitude * np.sin(x + self.phase)
        return x, y


class MAMLEngine:
    """
    FAZ 15 MAML & Meta-SGD Meta-Öğrenme Motoru.
    
    Özellikler:
    - İç Döngü (Inner Loop): 1-3 Gradyan Adımında Hızlı Görev Adaptasyonu
    - Dış Döngü (Outer Loop / Meta-Update): Başlangıç Ağırlıklarının (θ) Optimizasyonu
    - Meta-SGD: Parametre Başına Öğrenilebilir Adaptasyon Oranı (α vektörü)
    - Few-Shot (0-Shot -> 5-Shot) Genelleme Kabiliyeti
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 40,
        output_dim: int = 1,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr

        # Başlangıç Ağırlıkları (θ)
        np.random.seed(42)
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # Meta-SGD Öğrenilebilir Öğrenme Oranları
        self.alpha_w1 = np.ones_like(self.w1) * inner_lr
        self.alpha_w2 = np.ones_like(self.w2) * inner_lr

    def forward(
        self,
        x: np.ndarray,
        weights: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None,
    ) -> np.ndarray:
        """İleri geçiş (Forward pass)."""
        w1, b1, w2, b2 = weights if weights is not None else (self.w1, self.b1, self.w2, self.b2)
        h = np.maximum(0, np.dot(x, w1) + b1)  # ReLU
        out = np.dot(h, w2) + b2
        return out

    def compute_loss_and_grads(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """MSE Kaybı ve analitik gradyan hesaplaması."""
        w1, b1, w2, b2 = weights if weights is not None else (self.w1, self.b1, self.w2, self.b2)
        z1 = np.dot(x, w1) + b1
        h = np.maximum(0, z1)
        pred = np.dot(h, w2) + b2

        loss = float(np.mean((pred - y) ** 2))

        # Geriye yayılım (Backprop)
        d_out = 2.0 * (pred - y) / x.shape[0]
        d_w2 = np.dot(h.T, d_out)
        d_b2 = np.sum(d_out, axis=0, keepdims=True)

        d_h = np.dot(d_out, w2.T)
        d_z1 = d_h * (z1 > 0)
        d_w1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        grads = {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2}
        return loss, grads

    def adapt_inner_loop(
        self,
        task: MetaTask,
        k_shots: int = 5,
        inner_steps: int = 1,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]:
        """İç Döngü Adaptasyonu: Support Set üzerinde k_shots veriyle θ -> θ' günceller."""
        x_supp, y_supp = task.sample_data(k_shots)
        current_w = (self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

        for _ in range(inner_steps):
            loss, grads = self.compute_loss_and_grads(x_supp, y_supp, weights=current_w)
            w1_new = current_w[0] - self.alpha_w1 * grads["w1"]
            b1_new = current_w[1] - self.inner_lr * grads["b1"]
            w2_new = current_w[2] - self.alpha_w2 * grads["w2"]
            b2_new = current_w[3] - self.inner_lr * grads["b2"]
            current_w = (w1_new, b1_new, w2_new, b2_new)

        post_loss, _ = self.compute_loss_and_grads(x_supp, y_supp, weights=current_w)
        return current_w, post_loss

    def train_meta_step(
        self,
        tasks: List[MetaTask],
        k_shots: int = 5,
        q_queries: int = 15,
    ) -> Dict[str, float]:
        """Dış Döngü: Meta-Batch görevleri üzerinden θ ağırlıklarını günceller."""
        meta_w1_grad = np.zeros_like(self.w1)
        meta_b1_grad = np.zeros_like(self.b1)
        meta_w2_grad = np.zeros_like(self.w2)
        meta_b2_grad = np.zeros_like(self.b2)

        total_pre_loss = 0.0
        total_post_loss = 0.0

        for task in tasks:
            # 1. Support Set ile Adaptasyon
            adapted_w, pre_loss = self.adapt_inner_loop(task, k_shots=k_shots)
            total_pre_loss += pre_loss

            # 2. Query Set ile Meta-Loss
            x_query, y_query = task.sample_data(q_queries)
            q_loss, q_grads = self.compute_loss_and_grads(x_query, y_query, weights=adapted_w)
            total_post_loss += q_loss

            meta_w1_grad += q_grads["w1"]
            meta_b1_grad += q_grads["b1"]
            meta_w2_grad += q_grads["w2"]
            meta_b2_grad += q_grads["b2"]

        n = len(tasks)
        self.w1 -= self.meta_lr * (meta_w1_grad / n)
        self.b1 -= self.meta_lr * (meta_b1_grad / n)
        self.w2 -= self.meta_lr * (meta_w2_grad / n)
        self.b2 -= self.meta_lr * (meta_b2_grad / n)

        return {
            "pre_adapt_loss": float(total_pre_loss / n),
            "post_adapt_loss": float(total_post_loss / n),
            "meta_gradient_norm": float(np.linalg.norm(meta_w1_grad) / n),
        }
