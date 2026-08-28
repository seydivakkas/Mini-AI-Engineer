"""U-Net Anlamsal Bölütleme (Semantic Segmentation) Mimarisi.

Bu modül; Ronneberger et al. (2015) tarafından geliştirilen ve medikal/endüstriyel
bölütlemede standart kabul edilen U-Net mimarisini (Encoder, Decoder, Skip Connections)
PyTorch nn.Module olarak modüler ve ölçeklenebilir şekilde uygular.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CifteEvrisim(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 bloğu."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class AsagiOrnekleme(nn.Module):
    """MaxPool2D ile mekansal boyutu yarıya indiren ve Çifte Evrişim uygulayan Encoder bloğu."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            CifteEvrisim(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class YukariOrnekleme(nn.Module):
    """Yukarı evrişim (Upsample) ve Skip Connection birleştirme uygulayan Decoder bloğu."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = CifteEvrisim(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = CifteEvrisim(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: Decoder'dan gelen düşük çözünürlüklü özellik tensörü
        x2: Encoder'dan gelen Skip Connection yüksek çözünürlüklü tensör
        """
        x1 = self.up(x1)

        # Mekansal boyut uyuşmazlığı durumunda padding (Borders handling)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Skip Connection ile birleştir (Concatenate along Channel axis)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """End-to-End U-Net Anlamsal Bölütleme Ağı."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        kanal_tabani: int = 32,
        bilinear: bool = False,
    ) -> None:
        """
        Args:
            in_channels: Giriş görsel kanal sayısı (örn. 3 for RGB, 1 for Grayscale).
            num_classes: Bölütlenecek hedef sınıf sayısı (Background + Nesneler).
            kanal_tabani: Başlangıç evrişim kanal kapasitesi (32 veya 64).
            bilinear: True ise Bilinear Interpolation, False ise ConvTranspose2d kullanır.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        c = kanal_tabani
        # Daralan Yol (Encoder)
        self.inc = CifteEvrisim(in_channels, c)
        self.down1 = AsagiOrnekleme(c, c * 2)
        self.down2 = AsagiOrnekleme(c * 2, c * 4)
        self.down3 = AsagiOrnekleme(c * 4, c * 8)
        factor = 2 if bilinear else 1
        self.down4 = AsagiOrnekleme(c * 8, (c * 16) // factor)

        # Genişleyen Yol (Decoder + Skip Connections)
        self.up1 = YukariOrnekleme(c * 16, (c * 8) // factor, bilinear)
        self.up2 = YukariOrnekleme(c * 8, (c * 4) // factor, bilinear)
        self.up3 = YukariOrnekleme(c * 4, (c * 2) // factor, bilinear)
        self.up4 = YukariOrnekleme(c * 2, c, bilinear)

        # Çıkış 1x1 Evrişimi
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (Contracting Path)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder (Expansive Path with Skip Connections)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Logits (N, num_classes, H, W)
        logits = self.outc(x)
        return logits
