"""
Vision Transformer İçin İleri Düzey Veri Artırma Yöntemleri: Mixup & CutMix
---------------------------------------------------------------------------
Zhang et al. (2017) Mixup ve Yun et al. (2019) CutMix formülasyonları ile
Vision Transformer'ların CIFAR-100 gibi küçük veri setlerinde aşırı uydurmasını (overfitting)
önleyen regülarizasyon motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def rastgele_sinirlayici_kutu(
    genislik: int,
    yukseklik: int,
    lam: float
) -> Tuple[int, int, int, int]:
    """
    CutMix için lambda alan oranına uygun rastgele kırpma koordinatları (x1, y1, x2, y2) üretir.
    """
    kirpma_orani = np.sqrt(1.0 - lam)
    kirpma_g = int(genislik * kirpma_orani)
    kirpma_y = int(yukseklik * kirpma_orani)

    merkez_x = np.random.randint(genislik)
    merkez_y = np.random.randint(yukseklik)

    x1 = np.clip(merkez_x - kirpma_g // 2, 0, genislik)
    y1 = np.clip(merkez_y - kirpma_y // 2, 0, yukseklik)
    x2 = np.clip(merkez_x + kirpma_g // 2, 0, genislik)
    y2 = np.clip(merkez_y + kirpma_y // 2, 0, yukseklik)

    return x1, y1, x2, y2


class MixupCutMixUygulayici:
    """
    Mixup ve CutMix veri artırma işlemlerini batch düzeyinde uygulayan ve
    yumuşatılmış hedef olasılık dağılımları (soft targets) üreten sınıf.
    """
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        uygulama_olasiligi: float = 1.0,
        sinif_sayisi: int = 100
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.uygulama_olasiligi = uygulama_olasiligi
        self.sinif_sayisi = sinif_sayisi

    def __call__(
        self,
        gorseller: torch.Tensor,
        etiketler: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Girdi:
          gorseller: (Batch, C, H, W)
          etiketler: (Batch,) tamsayı etiketler veya (Batch, Num_Classes) one-hot
        Çıktı:
          artirilmis_gorseller: (Batch, C, H, W)
          yumusak_etiketler: (Batch, Num_Classes)
        """
        # Etiketleri one-hot vektörüne dönüştür
        if etiketler.ndim == 1:
            y_one_hot = F.one_hot(etiketler, num_classes=self.sinif_sayisi).float()
        else:
            y_one_hot = etiketler.float()

        if np.random.rand() > self.uygulama_olasiligi:
            return gorseller, y_one_hot

        b, c, h, w = gorseller.shape
        perm = torch.randperm(b, device=gorseller.device)

        # Mixup mı CutMix mi seçimi (%50 - %50)
        cutmix_secimi = np.random.rand() < 0.5

        if cutmix_secimi and self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            x1, y1, x2, y2 = rastgele_sinirlayici_kutu(w, h, lam)
            
            # Gerçek kırpılan alan oranına göre lambda'yı güncelle
            gercek_lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
            
            artirilmis_x = gorseller.clone()
            artirilmis_x[:, :, y1:y2, x1:x2] = gorseller[perm, :, y1:y2, x1:x2]
            yumusak_y = gercek_lam * y_one_hot + (1.0 - gercek_lam) * y_one_hot[perm]

        elif self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            artirilmis_x = lam * gorseller + (1.0 - lam) * gorseller[perm]
            yumusak_y = lam * y_one_hot + (1.0 - lam) * y_one_hot[perm]
        else:
            artirilmis_x = gorseller
            yumusak_y = y_one_hot

        return artirilmis_x, yumusak_y
