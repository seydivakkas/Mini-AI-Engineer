"""
Sıfırdan Ölçekli Nokta Çarpım Dikkat Mekanizması (Scaled Dot-Product Attention)
-------------------------------------------------------------------------------
Vaswani et al. (2017) "Attention Is All You Need" makalesindeki matematiksel
formülasyonu saf PyTorch tensör operasyonlarıyla sıfırdan uygulayan modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OlcekliNoktaCarpimDikkat(nn.Module):
    """
    Ölçekli Nokta Çarpım Dikkat Mekanizması:
    Attention(Q, K, V) = Softmax((Q * K^T) / sqrt(d_k) + M) * V
    """
    def __init__(self, dropout_orani: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Girdiler:
        - q: (Batch, Heads, Seq_Q, d_k)
        - k: (Batch, Heads, Seq_K, d_k)
        - v: (Batch, Heads, Seq_K, d_v)
        - mask: İsteğe bağlı dikkat maskesi (Örn: Causal veya Padding mask)
        
        Çıktılar:
        - cikti: (Batch, Heads, Seq_Q, d_v)
        - dikkat_agirliklari: (Batch, Heads, Seq_Q, Seq_K)
        """
        d_k = q.size(-1)
        olcek = 1.0 / math.sqrt(d_k)

        # 1. Skor Hesabı: (Q @ K^T) / sqrt(d_k)
        # q: (B, H, N_q, d_k), k^T: (B, H, d_k, N_k) -> skorlar: (B, H, N_q, N_k)
        skorlar = torch.matmul(q, k.transpose(-2, -1)) * olcek

        # 2. Maskeleme (Opsiyonel)
        if mask is not None:
            # 0 olan yerleri çok küçük bir negatif sayıya (-1e9) eşitle
            skorlar = skorlar.masked_fill(mask == 0, -1e9)

        # 3. Softmax ile Olasılık Dağılımına Dönüştürme
        dikkat_agirliklari = F.softmax(skorlar, dim=-1)

        # 4. Dropout (Opsiyonel)
        if self.dropout is not None:
            uygulanan_agirliklar = self.dropout(dikkat_agirliklari)
        else:
            uygulanan_agirliklar = dikkat_agirliklari

        # 5. Değerler (V) ile Ağırlıklı Toplam: (B, H, N_q, N_k) @ (B, H, N_k, d_v) -> (B, H, N_q, d_v)
        cikti = torch.matmul(uygulanan_agirliklar, v)

        return cikti, dikkat_agirliklari
