"""
Sıfırdan Çok Kafalı Öz Dikkat Mekanizması
-----------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CokKafaliOzDikkat(nn.Module):
    """
    Çok Kafalı Öz Dikkat (Multi-Head Self-Attention)
    """
    def __init__(self, model_boyutu: int = 64, kafa_sayisi: int = 4, dropout_orani: float = 0.0, bias: bool = True):
        super().__init__()
        assert model_boyutu % kafa_sayisi == 0, "Model boyutu kafa sayısına tam bölünmelidir."
        self.model_boyutu = model_boyutu
        self.kafa_sayisi = kafa_sayisi
        self.d_k = model_boyutu // kafa_sayisi

        self.w_q = nn.Linear(model_boyutu, model_boyutu, bias=bias)
        self.w_k = nn.Linear(model_boyutu, model_boyutu, bias=bias)
        self.w_v = nn.Linear(model_boyutu, model_boyutu, bias=bias)
        self.w_o = nn.Linear(model_boyutu, model_boyutu, bias=bias)
        self.dropout = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, seq_len, _ = x.size()

        # Projeksiyon ve Kafalara Bölme: (B, H, N, d_k)
        q = self.w_q(x).view(b, seq_len, self.kafa_sayisi, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, seq_len, self.kafa_sayisi, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, seq_len, self.kafa_sayisi, self.d_k).transpose(1, 2)

        # Ölçekli Nokta Çarpım Dikkati
        skorlar = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            skorlar = skorlar.masked_fill(mask == 0, -1e9)
        
        dikkat_haritalari = F.softmax(skorlar, dim=-1)
        if self.dropout is not None:
            agirliklar = self.dropout(dikkat_haritalari)
        else:
            agirliklar = dikkat_haritalari

        out = torch.matmul(agirliklar, v) # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.model_boyutu)
        return self.w_o(out), dikkat_haritalari
