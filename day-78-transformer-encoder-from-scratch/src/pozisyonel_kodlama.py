"""
Sıfırdan Pozisyonel Kodlama Modülü (Positional Encoding)
-------------------------------------------------------
Transformer ağlarında sıra/konum bilgisini enjekte eden Sinüzoidal (Vaswani et al.)
ve Öğrenilebilir (ViT / BERT) pozisyonel gömülme katmanları.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPozisyonelKodlama(nn.Module):
    """
    Sabit Sinüzoidal Pozisyonel Kodlama:
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    def __init__(self, model_boyutu: int = 64, maksimum_uzunluk: int = 512, dropout_orani: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else None

        # Pozisyon tablosu oluştur: (maksimum_uzunluk, model_boyutu)
        pe = torch.zeros(maksimum_uzunluk, model_boyutu)
        pozisyon = torch.arange(0, maksimum_uzunluk, dtype=torch.float).unsqueeze(1)
        
        bolen = torch.exp(
            torch.arange(0, model_boyutu, 2).float() * (-math.log(10000.0) / model_boyutu)
        )

        pe[:, 0::2] = torch.sin(pozisyon * bolen)
        pe[:, 1::2] = torch.cos(pozisyon * bolen)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, Seq_Len, Model_Boyutu)
        Çıktı: Girdi + PE (Batch, Seq_Len, Model_Boyutu)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class OgrenilebilirPozisyonelKodlama(nn.Module):
    """
    Vision Transformer (ViT) ve BERT tarzı öğrenilebilir 1D pozisyonel parametre tablosu.
    """
    def __init__(self, model_boyutu: int = 64, maksimum_uzunluk: int = 512, dropout_orani: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, maksimum_uzunluk, model_boyutu) * 0.02)
        self.dropout = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        if self.dropout is not None:
            x = self.dropout(x)
        return x
