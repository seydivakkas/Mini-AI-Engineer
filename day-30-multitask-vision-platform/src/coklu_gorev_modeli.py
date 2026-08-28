"""
Birleşik Çoklu Görev Görsel Modeli (Unified Multi-Task Vision Architecture)
ve Homoscedastic Belirsizlik Ağırlıklı Çoklu Kayıp Fonksiyonu.
"""

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CokluGorevGorselModeli(nn.Module):
    """
    Tek bir paylaşımlı omurga (Shared Backbone) üzerinden eşzamanlı olarak:
    1. Sahne Sınıflandırması (Global Classification)
    2. Nesne Tespiti (Dense Object Detection)
    3. Anlamsal Bölütleme (Dense Semantic Segmentation)
    4. Re-ID Görsel Görünüş Gömmeleri (128D Embedding)
    üreten uçtan uca derin sinir ağı.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_scene_classes: int = 4,
        num_object_classes: int = 4,
        num_seg_classes: int = 5,
        reid_dim: int = 128
    ):
        super().__init__()
        self.num_scene_classes = num_scene_classes
        self.num_object_classes = num_object_classes
        self.num_seg_classes = num_seg_classes
        self.reid_dim = reid_dim

        # -------------------------------------------------------------
        # 1. Paylaşımlı Çok Ölçekli Evrişimli Omurga (Shared Backbone)
        # -------------------------------------------------------------
        # Kademe 1: H/2 x W/2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Kademe 2: H/4 x W/4
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Kademe 3: H/8 x W/8
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Kademe 4: H/16 x W/16
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------------------
        # 2. Görev Başlığı 1: Sahne Sınıflandırma Başlığı (Scene Cls Head)
        # -------------------------------------------------------------
        self.scene_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_scene_classes)
        )

        # -------------------------------------------------------------
        # 3. Görev Başlığı 2: Nesne Tespiti Başlığı (Dense Detection Head - H/8)
        # -------------------------------------------------------------
        self.det_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.det_cls = nn.Conv2d(128, num_object_classes, kernel_size=1)
        self.det_box = nn.Conv2d(128, 4, kernel_size=1)          # (l, t, r, b) mesafe regresyonu
        self.det_obj = nn.Conv2d(128, 1, kernel_size=1)          # Nesnellik (Objectness)

        # -------------------------------------------------------------
        # 4. Görev Başlığı 3: Anlamsal Bölütleme Başlığı (Dense Seg Head)
        # -------------------------------------------------------------
        self.seg_fuse = nn.Sequential(
            nn.Conv2d(64 + 128 + 256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.seg_logits = nn.Conv2d(64, num_seg_classes, kernel_size=1)

        # -------------------------------------------------------------
        # 5. Görev Başlığı 4: Re-ID Görsel Görünüş Gömmesi (128D Embedding)
        # -------------------------------------------------------------
        self.reid_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, reid_dim)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Girdi: (B, 3, H, W)
        Çıktılar:
            - scene_logits: (B, num_scene_classes)
            - det_cls: (B, num_object_classes, H/8, W/8)
            - det_box: (B, 4, H/8, W/8)
            - det_obj: (B, 1, H/8, W/8)
            - seg_logits: (B, num_seg_classes, H, W)
            - reid_embedding: (B, reid_dim) L2-normalize
        """
        b, _, orig_h, orig_w = x.shape

        # Omurga İleri Besleme
        c1 = self.stem(x)          # (B, 32, H/2, W/2)
        c2 = self.layer1(c1)       # (B, 64, H/4, W/4)
        c3 = self.layer2(c2)       # (B, 128, H/8, W/8)
        c4 = self.layer3(c3)       # (B, 256, H/16, W/16)

        # 1. Sahne Sınıflandırma
        scene_logits = self.scene_head(c4)

        # 2. Nesne Tespiti (C3 üzerinden)
        det_feat = self.det_conv(c3)
        det_cls_out = self.det_cls(det_feat)
        det_box_out = self.det_box(det_feat)
        det_obj_out = torch.sigmoid(self.det_obj(det_feat))

        # 3. Anlamsal Bölütleme (Çok Ölçekli Birleştirme H/4 çözünürlüğünde)
        c3_up = F.interpolate(c3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        c4_up = F.interpolate(c4, size=c2.shape[2:], mode="bilinear", align_corners=False)
        seg_fused = self.seg_fuse(torch.cat([c2, c3_up, c4_up], dim=1))
        seg_out = self.seg_logits(seg_fused)
        # Orijinal (H, W) boyutuna yukarı örnekleme
        seg_logits = F.interpolate(seg_out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        # 4. Re-ID Embedding
        reid_raw = self.reid_head(c4)
        reid_emb = F.normalize(reid_raw, p=2, dim=1)

        return {
            "scene_logits": scene_logits,
            "det_cls": det_cls_out,
            "det_box": det_box_out,
            "det_obj": det_obj_out,
            "seg_logits": seg_logits,
            "reid_embedding": reid_emb,
            "backbone_features": [c1, c2, c3, c4]
        }


class BelirsizlikAgirlikliKayip(nn.Module):
    """
    Homoscedastic Belirsizlik Ağırlıklı Çoklu Görev Kaybı (Kendall et al., CVPR 2018).
    Görev kayıplarını manuel sabit katsayılar (lambda) yerine öğrenilebilir
    belirsizlik parametreleri (log_var = ln(sigma^2)) ile dinamik dengeler:

    L_total = 0.5 * exp(-s_cls) * L_cls + 0.5 * exp(-s_det) * L_det + 0.5 * exp(-s_seg) * L_seg + 0.5 * (s_cls + s_det + s_seg)
    """

    def __init__(self, gorev_sayisi: int = 3):
        super().__init__()
        # s = log(sigma^2) parametresi başlat (0.0 -> sigma = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(gorev_sayisi))

    def forward(self, kayip_listesi: List[torch.Tensor]) -> Tuple[torch.Tensor, List[float]]:
        toplam_kayip = 0.0
        agirliklar = []

        for i, l in enumerate(kayip_listesi):
            s = self.log_vars[i]
            precision = torch.exp(-s)
            gorev_kaybi = 0.5 * precision * l + 0.5 * s
            toplam_kayip = toplam_kayip + gorev_kaybi
            agirliklar.append(float(precision.detach().cpu().item()))

        return toplam_kayip, agirliklar
