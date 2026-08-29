"""
Bradley-Terry Tercih Modeli ve Ödül Kayıp Fonksiyonu Modülü (Day 108).
Çiftli karşılaştırma (y_w > y_l), marjin desteği ve sayısal kararlı Log-Sigmoid kaybı.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BradleyTerryLoss(nn.Module):
    """
    Bradley-Terry İkili Tercih Kayıp Fonksiyonu.
    Formül: Loss = -E[log(sigmoid(r_w - r_l - margin))] + lambda * (r_w^2 + r_l^2)
    """

    def __init__(self, margin: float = 0.0, reg_lambda: float = 0.001):
        super().__init__()
        self.margin = margin
        self.reg_lambda = reg_lambda

    def forward(
        self,
        r_w: torch.Tensor,
        r_l: torch.Tensor,
        custom_margin: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        r_w: Tercih edilen (kazanan) yanıt skaler ödülleri [B]
        r_l: Reddedilen (kaybeden) yanıt skaler ödülleri [B]
        Çıktı: (toplam_kayip, tercih_dogrulugu)
        """
        m = custom_margin if custom_margin is not None else self.margin
        fark = r_w - r_l - m

        # Sayısal olarak kararlı -log(sigmoid(x)) = -F.logsigmoid(x)
        kayip_bt = -F.logsigmoid(fark).mean()

        # Ödüllerin sonsuza patlamasını engelleyen L2 regülarizasyonu
        kayip_reg = self.reg_lambda * (r_w.pow(2).mean() + r_l.pow(2).mean())

        toplam_kayip = kayip_bt + kayip_reg
        dogruluk = (r_w > r_l).float().mean()

        return toplam_kayip, dogruluk


def tercih_olasiligi(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry modeline göre kazanma olasılığı: P(y_w > y_l) = sigmoid(r_w - r_l)
    """
    return torch.sigmoid(r_w - r_l)


def tercih_dogrulugu(r_w: torch.Tensor, r_l: torch.Tensor) -> float:
    """Doğru sıralama yüzdesi: r_w > r_l olan örneklerin oranı."""
    return float((r_w > r_l).float().mean().item())
