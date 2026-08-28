"""
Edge ve Gömülü Sistemler İçin Standart ve Derinlik Ayrışımlı (Depthwise Separable) CNN Mimarileri.
"""

from typing import Tuple
import torch
import torch.nn as nn


class DerinlikAyrisimliKonvolusyon(nn.Module):
    """MobileNet V1/V2 mimarilerindeki Depthwise + Pointwise ayrışımlı konvolüsyon bloğu."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 1. Aşama: Depthwise Konvolüsyon (Her kanala bağımsız 3x3 uzamsal filtre, groups=in_channels)
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU6(inplace=True)

        # 2. Aşama: Pointwise Konvolüsyon (Kanallar arası 1x1 doğrusal kombinasyon)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class StandartCNN(nn.Module):
    """Geleneksel ağır standart konvolüsyon ve tam bağlantılı (Dense) katmanlı temel model."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Blok 1: 3 -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            # Blok 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            # Blok 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 16x16 -> 8x8
        )

        # Ağır Klasik Sınıflandırıcı (128 * 8 * 8 = 8192 giriş)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class TinyVisionCNN(nn.Module):
    """Gömülü cihazlar ve Edge AI için Depthwise Separable Conv ve Global Average Pooling tabanlı hafif CNN."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        # Stem Katmanı (Hızlı uzamsal küçültme: 64x64 -> 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # Derinlik Ayrışımlı Bloklar
        self.stage1 = DerinlikAyrisimliKonvolusyon(16, 32, stride=1)   # 32x32
        self.stage2 = DerinlikAyrisimliKonvolusyon(32, 64, stride=2)   # 32x32 -> 16x16
        self.stage3 = DerinlikAyrisimliKonvolusyon(64, 128, stride=2)  # 16x16 -> 8x8
        self.stage4 = DerinlikAyrisimliKonvolusyon(128, 128, stride=1) # 8x8

        # Parametresiz Global Ortalama Havuzlama (Global Average Pooling) + Hafif Kafa
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
