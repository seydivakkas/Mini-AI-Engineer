"""
Mask R-CNN Mimarisi, RoIAlign Mekanizması ve FCN Maske Başlığı (Mask Head).
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


class RoIAlignModulu(nn.Module):
    """
    RoIAlign (Region of Interest Align):
    Piksel kuantizasyon hatalarını (RoIPool rounding errors) ortadan kaldıran,
    çift doğrusal interpolasyon (Bilinear Interpolation) tabanlı öznitelik örnekleyici.
    """

    def __init__(self, cikti_boyutu: Tuple[int, int] = (14, 14), spatial_scale: float = 1.0 / 16.0, ornekleme_orani: int = 2):
        super().__init__()
        self.cikti_boyutu = cikti_boyutu
        self.spatial_scale = spatial_scale
        self.ornekleme_orani = ornekleme_orani

    def forward(self, feature_map: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Girdi:
            feature_map: (B, C, H, W) boyutlu özellik haritası
            rois: (N, 5) boyutlu [batch_idx, x1, y1, x2, y2] tensörü
        Çıktı:
            (N, C, cikti_h, cikti_w) boyutlu hizalanmış özellik tensörü
        """
        return ops.roi_align(
            feature_map,
            rois,
            output_size=self.cikti_boyutu,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.ornekleme_orani,
            aligned=True
        )


class FCNMaskeBasligi(nn.Module):
    """
    Mask R-CNN FCN Maske Dalı:
    Her nesne adayı için (C x 14 x 14) boyutlu RoIAlign çıktısını
    (K x 28 x 28) ikili maske olasılıklarına dönüştürür.
    """

    def __init__(self, girdi_kanali: int = 256, sinif_sayisi: int = 4):
        super().__init__()
        self.sinif_sayisi = sinif_sayisi

        # 4 adet 3x3 Conv katmanı
        self.conv_blok = nn.Sequential(
            nn.Conv2d(girdi_kanali, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 2x2 Transposed Conv (Yukarı Örnekleme: 14x14 -> 28x28)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 Conv (Sınıf Başına 1 İkili Maske Kanalı)
        self.mask_logits = nn.Conv2d(256, sinif_sayisi, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (N, 256, 14, 14)
        Çıktı: (N, sinif_sayisi, 28, 28)
        """
        feat = self.conv_blok(x)
        up = self.relu(self.deconv(feat))
        logits = self.mask_logits(up)
        return logits


class MaskRCNNYoneticisi(nn.Module):
    """
    Mask R-CNN Çok Görevli Model ve Çıkarım Yöneticisi:
    L = L_cls + L_box + L_mask çoklu kayıp yapısını ve çıkarımı yönetir.
    """

    def __init__(self, girdi_kanali: int = 256, sinif_sayisi: int = 4):
        super().__init__()
        self.sinif_sayisi = sinif_sayisi
        self.roi_align = RoIAlignModulu(cikti_boyutu=(14, 14), spatial_scale=1.0, ornekleme_orani=2)
        self.mask_head = FCNMaskeBasligi(girdi_kanali=girdi_kanali, sinif_sayisi=sinif_sayisi)

        # Sınıflandırma ve BBox Dalı
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(girdi_kanali * 14 * 14, 512),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(512, sinif_sayisi)
        self.box_head = nn.Linear(512, sinif_sayisi * 4)

    def forward(self, feature_map: torch.Tensor, rois: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        İleri Besleme:
            feature_map: (B, C, H, W)
            rois: (N, 5) [batch_id, x1, y1, x2, y2]
        """
        roi_feats = self.roi_align(feature_map, rois)
        mask_logits = self.mask_head(roi_feats)

        flat = self.fc(roi_feats)
        cls_logits = self.cls_head(flat)
        box_deltas = self.box_head(flat)

        return {
            "cls_logits": cls_logits,
            "box_deltas": box_deltas,
            "mask_logits": mask_logits,
            "roi_features": roi_feats
        }

    @staticmethod
    def coklu_gorev_kaybi(
        cls_logits: torch.Tensor,
        box_deltas: torch.Tensor,
        mask_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_masks: torch.Tensor,
        w_cls: float = 1.0,
        w_box: float = 1.0,
        w_mask: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-Task Loss: L = w_cls * L_cls + w_box * L_box + w_mask * L_mask

        Mask Loss: Sınıflandırmadan bağımsızlaştırılmış ikili çapraz entropi (BCE).
        Her RoI için yalnızca GT sınıfına karşılık gelen maske kanalında BCE hesaplanır.
        """
        # 1. Sınıflandırma Kaybı (Cross-Entropy)
        l_cls = F.cross_entropy(cls_logits, gt_labels)

        # 2. Kutu Regresyon Kaybı (Smooth L1 Loss)
        # GT sınıfının kutu deltasını seç
        n = gt_labels.size(0)
        idx = torch.arange(n, device=gt_labels.device)
        box_pred = box_deltas.view(n, -1, 4)[idx, gt_labels]
        l_box = F.smooth_l1_loss(box_pred, gt_boxes)

        # 3. Maske Kaybı (BCE Loss)
        # Yalnızca GT sınıf kanalındaki maske lojitini al
        mask_pred = mask_logits[idx, gt_labels]  # (N, 28, 28)
        # gt_masks (N, 28, 28) ile ikili çapraz entropi
        l_mask = F.binary_cross_entropy_with_logits(mask_pred, gt_masks.float())

        toplam_kayip = w_cls * l_cls + w_box * l_box + w_mask * l_mask

        return {
            "toplam_kayip": toplam_kayip,
            "l_cls": l_cls,
            "l_box": l_box,
            "l_mask": l_mask
        }
