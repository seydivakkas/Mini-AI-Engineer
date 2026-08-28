"""
Yumuşatılmış Çapraz Entropi Kaybı (Label Smoothing Cross-Entropy)
================================================================
Aşırı güveni (Overconfidence) ve logit patlamalarını önleyen,
Mixup/CutMix etiket çiftleriyle uyumlu çalışan düzenlileştirilmiş kayıp fonksiyonu.

Referans: Szegedy et al., 'Rethinking the Inception Architecture for Computer Vision', CVPR 2016.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class YumusatilmisCrossEntropyLoss(nn.Module):
    """
    Etiket Yumuşatma (Label Smoothing) içeren Çapraz Entropi Kaybı.

    Formül:
        q_k = (1 - epsilon) * 1_{k=y} + (epsilon / K)
        L = (1 - epsilon) * (-log p_y) + (epsilon / K) * sum(-log p_k)

    Args:
        smoothing (float): Etiket yumuşatma katsayısı epsilon in [0.0, 1.0)
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"Gecersiz smoothing katsayisi: {smoothing}. [0.0, 1.0) araliginda olmalidir.")
        self.smoothing = smoothing

    def forward(
        self,
        tahminler: torch.Tensor,
        hedef_a: torch.Tensor,
        hedef_b: Optional[torch.Tensor] = None,
        lam: float = 1.0
    ) -> torch.Tensor:
        """
        Kayıp hesaplar. Eğer hedef_b ve lam verilmişse Mixup/CutMix interpolasyon kaybı hesaplar.
        """
        log_olasiliklar = F.log_softmax(tahminler, dim=-1)
        sinif_sayisi = tahminler.size(-1)

        # 1. Hedef A için Yumuşatılmış Kayıp
        nll_loss_a = -log_olasiliklar.gather(dim=-1, index=hedef_a.unsqueeze(1)).squeeze(1)
        smooth_loss_a = -log_olasiliklar.mean(dim=-1)
        loss_a = (1.0 - self.smoothing) * nll_loss_a + self.smoothing * smooth_loss_a

        if hedef_b is None or lam >= 1.0:
            return loss_a.mean()

        # 2. Hedef B için Yumuşatılmış Kayıp (Mixup/CutMix Durumu)
        nll_loss_b = -log_olasiliklar.gather(dim=-1, index=hedef_b.unsqueeze(1)).squeeze(1)
        smooth_loss_b = -log_olasiliklar.mean(dim=-1)
        loss_b = (1.0 - self.smoothing) * nll_loss_b + self.smoothing * smooth_loss_b

        # 3. Doğrusal Ağırlıklı Toplam
        toplam_kayip = lam * loss_a + (1.0 - lam) * loss_b
        return toplam_kayip.mean()
