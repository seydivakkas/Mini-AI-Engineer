"""Anlamsal Bölütleme Kayıp Fonksiyonları ve Değerlendirme Metrikleri Modülü.

Bu modül; Çok Sınıflı Dice Kaybı (Dice Loss), Cross-Entropy + Dice Kombine Kaybı (Combo Loss),
Piksel Doğruluğu (Pixel Accuracy), Sınıf Bazında ve Mean IoU (Jaccard İndeksi) ile
Dice Katsayısı (F1 Skoru) hesaplamalarını içerir.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Çok sınıflı Soft Dice Loss (Sınıf dengesizliğine karşı dayanıklı kayıp)."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, C, H, W)
        targets: (N, H, W) int etiketler
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Targets'ı one-hot formatına çevir: (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dice_per_class = []
        for c in range(num_classes):
            p_c = probs[:, c].contiguous().view(-1)
            t_c = targets_one_hot[:, c].contiguous().view(-1)

            intersection = (p_c * t_c).sum()
            cardinality = p_c.sum() + t_c.sum()

            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_per_class.append(dice)

        dice_mean = torch.stack(dice_per_class).mean()
        return 1.0 - dice_mean


class ComboLoss(nn.Module):
    """Cross-Entropy + Dice Loss Hibrit Kayıp Fonksiyonu."""

    def __init__(self, alpha: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.alpha * loss_ce + (1.0 - self.alpha) * loss_dice


class BolutlemeMetrikleri:
    """Anlamsal bölütleme performans metriklerini hesaplayan sınıf."""

    @staticmethod
    def piksel_dogrulugu(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
        """Toplam doğru tahmin edilen piksel oranını hesaplar."""
        dogru = np.sum(pred_mask == true_mask)
        toplam = true_mask.size
        return float(dogru / toplam) if toplam > 0 else 0.0

    @staticmethod
    def sinif_iou_ve_dice(
        pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Her bir sınıf için IoU (Jaccard) ve Dice (F1) skorlarını hesaplar."""
        iou_skorlari = {}
        dice_skorlari = {}

        for c in range(num_classes):
            p_c = pred_mask == c
            t_c = true_mask == c

            intersection = np.logical_and(p_c, t_c).sum()
            union = np.logical_or(p_c, t_c).sum()
            cardinality = p_c.sum() + t_c.sum()

            # IoU
            if union == 0:
                iou = 1.0 if cardinality == 0 else 0.0
            else:
                iou = float(intersection / union)

            # Dice
            if cardinality == 0:
                dice = 1.0
            else:
                dice = float((2.0 * intersection) / cardinality)

            iou_skorlari[c] = iou
            dice_skorlari[c] = dice

        return iou_skorlari, dice_skorlari

    @classmethod
    def kapsamli_rapor(
        cls,
        pred_masks: np.ndarray,
        true_masks: np.ndarray,
        sinif_isimleri: List[str],
    ) -> Dict:
        """Toplu veri seti üzerinde mIoU, Mean Dice ve Piksel Doğruluğunu hesaplar."""
        num_classes = len(sinif_isimleri)
        N = len(pred_masks)

        pixel_accs = []
        sinif_ious = {c: [] for c in range(num_classes)}
        sinif_dices = {c: [] for c in range(num_classes)}

        for i in range(N):
            p_m = pred_masks[i]
            t_m = true_masks[i]

            acc = cls.piksel_dogrulugu(p_m, t_m)
            pixel_accs.append(acc)

            ious, dices = cls.sinif_iou_ve_dice(p_m, t_m, num_classes)
            for c in range(num_classes):
                sinif_ious[c].append(ious[c])
                sinif_dices[c].append(dices[c])

        sinif_raporu = {}
        for c, ad in enumerate(sinif_isimleri):
            sinif_raporu[ad] = {
                "iou": float(np.mean(sinif_ious[c])),
                "dice": float(np.mean(sinif_dices[c])),
            }

        miou = float(np.mean([d["iou"] for d in sinif_raporu.values()]))
        mean_dice = float(np.mean([d["dice"] for d in sinif_raporu.values()]))
        mean_acc = float(np.mean(pixel_accs))

        return {
            "miou": miou,
            "mean_dice": mean_dice,
            "pixel_accuracy": mean_acc,
            "sinif_raporu": sinif_raporu,
        }
