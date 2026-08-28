"""
Sıfırdan Katman Normalizasyonu (Custom Layer Normalization)
----------------------------------------------------------
Ba Ba et al. (2016) formülasyonunu saf PyTorch ile uygulayan modül:
y = ((x - mu) / sqrt(sigma^2 + eps)) * gamma + beta

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import torch
import torch.nn as nn


class OzelLayerNorm(nn.Module):
    """
    Sıfırdan Katman Normalizasyonu:
    Öznitelik boyutu (D) boyunca ortalama ve varyans hesaplayarak normalize eder.
    """
    def __init__(self, normalize_edilecek_boyut: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # gamma (ölçek) ve beta (kaydırma) parametreleri
        self.gamma = nn.Parameter(torch.ones(normalize_edilecek_boyut))
        self.beta = nn.Parameter(torch.zeros(normalize_edilecek_boyut))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, Seq_Len, D)
        """
        # Son boyut (D) üzerinden ortalama ve varyans: (B, N, 1)
        ortalama = x.mean(dim=-1, keepdim=True)
        varyans = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalizasyon
        x_norm = (x - ortalama) / torch.sqrt(varyans + self.eps)

        # Yeniden ölçekleme ve kaydırma
        return x_norm * self.gamma + self.beta
