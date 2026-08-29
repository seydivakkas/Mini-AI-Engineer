"""
Day 290 (FAZ 15): Mekanistik Yorumlanabilirlik ve Seyrek Otokodlayıcılar (SAE) Motoru.
Monosemantic Feature Extraction, Activation Steering ve Nöral Devre İncelemesi.
"""

from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SparseAutoencoder(nn.Module):
    """
    Aşırı Tamamlanmış Seyrek Otokodlayıcı (Overcomplete Sparse Autoencoder - SAE).
    
    Özellikler:
    - Çok Anlamlı Nöron Süperpozisyonunu (Polysemanticity) Tek Anlamlı Özniteliklere (Monosemantic) Ayrıştırma
    - L1 Seyreklik Regülarizasyonu (L1 Sparsity Loss) ile Düşük L0 Aktivasyonu (Seyreklik <= 8.2)
    - Residual Akım Boyutu d_in -> Genişletilmiş Sözlük Boyutu d_sae (4x - 32x Genişleme)
    """

    def __init__(self, d_in: int = 64, d_sae: int = 256, l1_coeff: float = 0.005):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff

        # Kodlayıcı (Encoder) ve Kod Çözücü (Decoder)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * (1.0 / np.sqrt(d_in)))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * (1.0 / np.sqrt(d_sae)))

        # Kod çözücü sütunlarını birim norma normalize et (Unit Norm Decoder)
        self.normalize_decoder_weights()

    def normalize_decoder_weights(self):
        """Decoder ağırlık sütunlarını L2 normuna göre normalize eder."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Residual aktivasyonları seyrek öznitelik uzayına f(x) kodlar."""
        x_centered = x - self.b_dec
        hidden_pre = torch.matmul(x_centered, self.W_enc) + self.b_enc
        return F.relu(hidden_pre)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Seyrek özniteliklerden residual akımı yeniden inşa eder (x_hat)."""
        return torch.matmul(f, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """İleri Geçiş: x -> f(x) -> x_hat, Yeniden İnşa Hatası ve L1 Seyreklik Kaybı."""
        f = self.encode(x)
        x_hat = self.decode(f)

        # L2 Yeniden İnşa Kaybı (Reconstruction Loss)
        l2_loss = F.mse_loss(x_hat, x)

        # L1 Seyreklik Kaybı (Sparsity Loss)
        l1_loss = torch.sum(torch.abs(f)) / x.shape[0]

        total_loss = l2_loss + self.l1_coeff * l1_loss
        return x_hat, f, l2_loss, total_loss


class ActivationSteeringEngine:
    """Nöral Aktivasyon Yönlendirme ve Müdahale Motoru (Golden Gate Claude Stili)."""

    @classmethod
    def steer_activation(
        cls,
        x: torch.Tensor,
        sae: SparseAutoencoder,
        feature_idx: int,
        alpha: float = 3.5,
    ) -> torch.Tensor:
        """Belirli bir tek anlamlı öznitelik yönünü doğrudan residual akıma enjekte eder."""
        steering_vector = sae.W_dec[feature_idx, :]
        return x + alpha * steering_vector
