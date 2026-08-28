"""
Vision Transformer İçin Etiket Yumuşatmalı Cross-Entropy Kayıp Fonksiyonu
-------------------------------------------------------------------------
Szegedy et al. (2016) Label Smoothing ve Yumuşak Hedef Dağılımları (Soft Targets)
destekleyen sayısal olarak kararlı kayıp fonksiyonu.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YumusatilmisCrossEntropyKaybi(nn.Module):
    """
    Hem tamsayı etiketleri (label smoothing ile) hem de Mixup/CutMix ile
    üretilen yumuşak hedef olasılık dağılımlarını (soft targets) destekleyen kayıp fonksiyonu.
    """
    def __init__(self, etiket_yumusatma: float = 0.1):
        super().__init__()
        assert 0.0 <= etiket_yumusatma < 1.0, "Etiket yumuşatma katsayısı [0, 1) aralığında olmalıdır!"
        self.etiket_yumusatma = etiket_yumusatma

    def forward(
        self,
        tahmin_logitleri: torch.Tensor,
        hedefler: torch.Tensor
    ) -> torch.Tensor:
        """
        Girdi:
          tahmin_logitleri: (Batch, Num_Classes)
          hedefler: (Batch,) tamsayı etiketler VEYA (Batch, Num_Classes) olasılık dağılımı
        Çıktı:
          Skaler kayıp değeri
        """
        num_classes = tahmin_logitleri.size(-1)
        log_olasiliklar = F.log_softmax(tahmin_logitleri, dim=-1)

        # 1. Durum: Hedefler tamsayı etiket vektörü ise
        if hedefler.ndim == 1:
            with torch.no_grad():
                yumusak_hedefler = torch.full_like(
                    log_olasiliklar,
                    self.etiket_yumusatma / num_classes
                )
                yumusak_hedefler.scatter_(
                    1,
                    hedefler.unsqueeze(1),
                    1.0 - self.etiket_yumusatma + (self.etiket_yumusatma / num_classes)
                )
            kayip = -(yumusak_hedefler * log_olasiliklar).sum(dim=-1).mean()

        # 2. Durum: Hedefler Mixup/CutMix yumuşak olasılık dağılımı ise
        elif hedefler.ndim == 2:
            # Hedeflere isteğe bağlı ek label smoothing uygula
            if self.etiket_yumusatma > 0.0:
                hedefler = (1.0 - self.etiket_yumusatma) * hedefler + self.etiket_yumusatma / num_classes
            kayip = -(hedefler * log_olasiliklar).sum(dim=-1).mean()

        else:
            raise ValueError(f"Geçersiz hedef tensör boyutu: {hedefler.shape}")

        return kayip
