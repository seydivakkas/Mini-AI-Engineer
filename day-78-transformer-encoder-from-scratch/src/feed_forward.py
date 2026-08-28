"""
Sıfırdan Pozisyonel İleri Beslemeli Ağ (Position-wise Feed-Forward Network - FFN)
--------------------------------------------------------------------------------
Transformer bloklarındaki iki katmanlı doğrusal genişleme (D -> 4D -> D) ve
GELU aktivasyonlu MLP modülü.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Optional
import torch
import torch.nn as nn


class BeslemeliIleriAg(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP):
    FFN(x) = Activation(x * W_1 + b_1) * W_2 + b_2
    """
    def __init__(
        self,
        model_boyutu: int = 64,
        genisleme_faktoru: int = 4,
        dropout_orani: float = 0.0,
        aktivasyon: str = "gelu"
    ):
        super().__init__()
        gizli_boyut = model_boyutu * genisleme_faktoru

        self.fc1 = nn.Linear(model_boyutu, gizli_boyut)
        self.fc2 = nn.Linear(gizli_boyut, model_boyutu)

        if aktivasyon.lower() == "gelu":
            self.akt = nn.GELU()
        elif aktivasyon.lower() == "relu":
            self.akt = nn.ReLU()
        else:
            raise ValueError(f"Desteklenmeyen aktivasyon fonksiyonu: {aktivasyon}")

        self.dropout = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, Seq_Len, Model_Boyutu)
        Çıktı: (Batch, Seq_Len, Model_Boyutu)
        """
        x = self.fc1(x)
        x = self.akt(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
