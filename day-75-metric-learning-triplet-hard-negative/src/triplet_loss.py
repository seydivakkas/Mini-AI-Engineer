"""
Modüler Triplet Margin Kayıp Fonksiyonu
---------------------------------------
Farklı madencilik politikalarını ve dinamik marjin parametresini destekleyen kayıp modülü.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from .mining_motoru import TripletMadencisi


class ModulerTripletMarginLoss(nn.Module):
    """
    Modüler Triplet Margin Loss Modülü.
    
    Matematiksel Tanım:
    L(a, p, n) = max(0, d(a, p) - d(a, n) + alpha)
    """
    def __init__(self, marjin: float = 0.3, strateji: str = "batch_semi_hard"):
        super().__init__()
        if marjin <= 0.0:
            raise ValueError(f"Marjin pozitif olmalıdır. Verilen: {marjin}")
        self.marjin = marjin
        self.strateji = strateji
        self.madenci = TripletMadencisi(marjin=marjin)

    def forward(
        self,
        gomulmeler: torch.Tensor,
        etiketler: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.madenci.madencilik_yap(
            gomulmeler=gomulmeler,
            etiketler=etiketler,
            strateji=self.strateji
        )
