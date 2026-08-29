"""
Artık Vektör Kuantalama (Residual Vector Quantizer - RVQ) Modülü (Day 169 - FAZ 9).
Sürekli gizli ses özniteliklerini N_q kademeli ayrık kod defteri indekslerine böler.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class VektorKuantalayici(nn.Module):
    """Tek bir VQ Kod Defteri Katmanı."""

    def __init__(self, codebook_size: int = 1024, dim: int = 128):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: [B, T, D]
        Döner: (z_q, indices, commit_loss)
        """
        # z ile kod defteri vektörleri arasındaki L2 mesafesi: (z - e)^2 = z^2 + e^2 - 2*z*e
        d = (
            torch.sum(z ** 2, dim=-1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=-1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )

        # En yakın kod defteri indeksi
        indices = torch.argmin(d, dim=-1)  # [B, T]
        z_q = self.embedding(indices)  # [B, T, D]

        # Commitment ve Codebook Kaybı
        commit_loss = F.mse_loss(z_q.detach(), z) + 0.25 * F.mse_loss(z_q, z.detach())

        # Düz Geçiş Gradyanı (Straight-Through Estimator - STE)
        z_q = z + (z_q - z).detach()

        return z_q, indices, commit_loss


class ResidualVectorQuantizer(nn.Module):
    """N_q kademeli Artık Vektör Kuantalayıcı (EnCodec / SoundStream Çekirdeği)."""

    def __init__(self, num_quantizers: int = 8, codebook_size: int = 1024, dim: int = 128):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.dim = dim

        self.quantizers = nn.ModuleList([
            VektorKuantalayici(codebook_size=codebook_size, dim=dim)
            for _ in range(num_quantizers)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: [B, T, D]
        Döner: (z_q_toplam, tum_indeksler, toplam_commit_loss)
        """
        residual = z
        z_q_total = 0.0
        all_indices = []
        total_loss = 0.0

        for quantizer in self.quantizers:
            z_q_i, indices_i, loss_i = quantizer(residual)
            residual = residual - z_q_i
            z_q_total = z_q_total + z_q_i
            all_indices.append(indices_i)
            total_loss = total_loss + loss_i

        # [Num_Q, B, T] -> [B, Num_Q, T]
        all_indices = torch.stack(all_indices, dim=1)
        return z_q_total, all_indices, total_loss
