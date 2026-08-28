"""
Ogrenme Orani Zamanlayicilari (Learning Rate Schedulers)
======================================================
Linear Warmup ve Cosine Annealing dinamiklerini birlestiren,
derin ogrenme ve Vision Transformer egitimlerinde kararlilik saglayan zamanlayici.
"""

from typing import List, Optional
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineScheduler(_LRScheduler):
    """
    Doğrusal Isınma (Linear Warmup) ve Kosinüs Sönümleme (Cosine Annealing) Zamanlayıcısı.

    Args:
        optimizer: PyTorch Optimizer nesnesi
        warmup_epochs (int): Doğrusal ısınmanın süreceği epoch sayısı
        max_epochs (int): Eğitimin toplam epoch sayısı
        eta_min (float): Ulaşılacak minimum öğrenme oranı
        last_epoch (int): Son epoch indeksi
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ) -> None:
        if warmup_epochs < 0 or warmup_epochs > max_epochs:
            raise ValueError(f"Gecersiz warmup_epochs: {warmup_epochs} (Toplam: {max_epochs})")
        if max_epochs <= 0:
            raise ValueError(f"Gecersiz max_epochs: {max_epochs}")

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Geçerli epoch için her parametre grubunun öğrenme oranını hesaplar."""
        current_epoch = self.last_epoch

        # 1. Faz: Doğrusal Isınma (Linear Warmup)
        if current_epoch < self.warmup_epochs:
            # 0'dan base_lr'a dogrusal artis (0 bolunmesini onlemek icin max(1, warmup))
            progress = (current_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * progress for base_lr in self.base_lrs]

        # 2. Faz: Kosinüs Sönümleme (Cosine Annealing)
        else:
            cosine_epochs = self.max_epochs - self.warmup_epochs
            if cosine_epochs <= 0:
                return [self.eta_min for _ in self.base_lrs]

            decay_step = current_epoch - self.warmup_epochs
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_step / cosine_epochs))

            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]
