"""MixUp ve CutMix İleri Düzey Veri Karıştırma Modülü.

Bu modül; modern derin öğrenme modellerinin genelleme kapasitesini artıran ve aşırı
güveni (overconfidence) kıran MixUp (Zhang et al., 2018) ve CutMix (Yun et al., 2019)
algoritmalarını ve bunların ağırlıklı kayıp fonksiyonlarını (Loss Functions) uygular.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class MixUpCutMixUygulayici:
    """MixUp ve CutMix dönüşümlerini mini-batch seviyesinde gerçekleştiren sınıf."""

    @staticmethod
    def uygula_mixup(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Mini-batch tensörlerine MixUp dönüşümü uygular.

        Denklem:
            x_tilde = lambda * x_i + (1 - lambda) * x_j
            y_tilde = lambda * y_i + (1 - lambda) * y_j

        Args:
            x: (B, C, H, W) girdi tensörü.
            y: (B,) hedef etiket tensörü.
            alpha: Beta dağılımı hiperparametresi (Beta(alpha, alpha)).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
                - mixup_x: Karıştırılmış görsel tensörü.
                - y_a: İlk etiketler (y_i).
                - y_b: İkinci etiketler (y_j).
                - lam: Karıştırma katsayısı lambda.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        indeksler = torch.randperm(batch_size, device=x.device)

        mixup_x = lam * x + (1.0 - lam) * x[indeksler]
        y_a, y_b = y, y[indeksler]
        return mixup_x, y_a, y_b, float(lam)

    @staticmethod
    def _rastgele_kutu(
        boyutlar: Tuple[int, int], lam: float
    ) -> Tuple[int, int, int, int]:
        """CutMix için rastgele kırpma kutusunun koordinatlarını hesaplar."""
        H, W = boyutlar
        kesim_orani = np.sqrt(1.0 - lam)
        kesim_w = int(W * kesim_orani)
        kesim_h = int(H * kesim_orani)

        # Kutunun merkez koordinatlarını rastgele seç
        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)

        x1 = np.clip(cx - kesim_w // 2, 0, W)
        y1 = np.clip(cy - kesim_h // 2, 0, H)
        x2 = np.clip(cx + kesim_w // 2, 0, W)
        y2 = np.clip(cy + kesim_h // 2, 0, H)

        return int(x1), int(y1), int(x2), int(y2)

    @classmethod
    def uygula_cutmix(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Mini-batch tensörlerine CutMix dönüşümü uygular.

        Görselin rastgele bir dikdörtgen bölgesini kesip diğer görselin aynı bölgesine yapıştırır.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
                - cutmix_x: Kes-yapıştır uygulanmış tensör.
                - y_a: İlk etiketler.
                - y_b: İkinci etiketler.
                - lam: Gerçek piksel oranına göre düzeltilmiş lambda katsayısı.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        indeksler = torch.randperm(batch_size, device=x.device)

        H, W = x.size(2), x.size(3)
        x1, y1, x2, y2 = cls._rastgele_kutu((H, W), lam)

        cutmix_x = x.clone()
        cutmix_x[:, :, y1:y2, x1:x2] = x[indeksler, :, y1:y2, x1:x2]

        # Gerçek kesilen alan oranına göre lambda katsayısını güncelle
        gercek_lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * H))
        y_a, y_b = y, y[indeksler]
        return cutmix_x, y_a, y_b, float(gercek_lam)


class MixUpCutMixKayip(nn.Module):
    """MixUp ve CutMix için çift hedefli (Dual Target) CrossEntropy kayıp fonksiyonu."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        tahminler: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Kayıp hesabı: L = lambda * L(y_a) + (1 - lambda) * L(y_b)."""
        kayip_a = self.criterion(tahminler, y_a)
        kayip_b = self.criterion(tahminler, y_b)
        return lam * kayip_a + (1.0 - lam) * kayip_b
