"""
Modern Veri Artırma Yöntemleri: Mixup ve CutMix
==============================================
Piksel interpolasyonu (Mixup) ve bölgesel yama kesip-yapıştırma (CutMix) tekniklerini
doğrudan PyTorch tensör batch'leri üzerinde yüksek hızda uygulayan modül.

Referanslar:
- Zhang et al., 'mixup: Beyond Empirical Risk Minimization', ICLR 2018.
- Yun et al., 'CutMix: Regularization Strategy to Train Strong Classifiers', ICCV 2019.
"""

from typing import Tuple, Optional
import numpy as np
import torch


class ModernArtirici:
    """
    Mixup ve CutMix veri artırma operasyonlarını yürüten sınıf.
    """

    @staticmethod
    def rastgele_kutu_olustur(
        genislik: int,
        yukseklik: int,
        lam: float
    ) -> Tuple[int, int, int, int]:
        """
        CutMix için lambda alan oranına uygun rastgele sınırlayıcı kutu koordinatları (x1, y1, x2, y2) üretir.
        """
        kesim_orani = np.sqrt(1.0 - lam)
        kesim_g = int(genislik * kesim_orani)
        kesim_y = int(yukseklik * kesim_orani)

        # Rastgele merkez noktası
        merkez_x = np.random.randint(0, genislik)
        merkez_y = np.random.randint(0, yukseklik)

        x1 = np.clip(merkez_x - kesim_g // 2, 0, genislik)
        y1 = np.clip(merkez_y - kesim_y // 2, 0, yukseklik)
        x2 = np.clip(merkez_x + kesim_g // 2, 0, genislik)
        y2 = np.clip(merkez_y + kesim_y // 2, 0, yukseklik)

        return int(x1), int(y1), int(x2), int(y2)

    @classmethod
    def uygula_mixup(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Görselleri ve etiketleri doğrusal olarak harmanlar (Mixup).
        x_mix = lam * x_a + (1 - lam) * x_b
        """
        if alpha > 0.0:
            lam = float(np.random.beta(alpha, alpha))
        else:
            lam = 1.0

        batch_size = x.size(0)
        indeksler = torch.randperm(batch_size, device=x.device)

        x_mix = lam * x + (1.0 - lam) * x[indeksler]
        y_a = y
        y_b = y[indeksler]

        return x_mix, y_a, y_b, lam

    @classmethod
    def uygula_cutmix(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Bir görselin rastgele bir bölgesini diğer görselden kesilen yama ile değiştirir (CutMix).
        """
        if alpha > 0.0:
            lam = float(np.random.beta(alpha, alpha))
        else:
            lam = 1.0

        batch_size, _, H, W = x.size()
        indeksler = torch.randperm(batch_size, device=x.device)

        x1, y1, x2, y2 = cls.rastgele_kutu_olustur(W, H, lam)

        x_cut = x.clone()
        x_cut[:, :, y1:y2, x1:x2] = x[indeksler, :, y1:y2, x1:x2]

        # Gerçek piksel alanına göre ayarlanmış lambda
        gercek_lam = 1.0 - float((x2 - x1) * (y2 - y1) / (W * H))
        y_a = y
        y_b = y[indeksler]

        return x_cut, y_a, y_b, gercek_lam
