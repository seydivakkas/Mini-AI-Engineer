"""
Uretim Seviyesi PyTorch Modeli - UretimVisionNet
===============================================
Endüstriyel sınıflandırma ve görsel öznitelik çıkarımı için tasarlanmış,
ONNX ve INT8 kuantizasyona tam uyumlu Residual CNN mimarisi.
"""

from typing import Tuple
import torch
import torch.nn as nn


class ResidualBlok(nn.Module):
    """
    ONNX düğüm kaynaştırmaya (Operator Fusion) tam uyumlu Residual Blok.
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> Add -> ReLU
    """

    def __init__(self, kanallar: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(kanallar, kanallar, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(kanallar)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(kanallar, kanallar, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(kanallar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        artik = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + artik
        out = self.relu(out)
        return out


class UretimVisionNet(nn.Module):
    """
    Üretim Ortamı Görsel Sınıflandırıcı Mimarisi.
    Girdi: (B, C, H, W) -> Çıktı: (B, num_classes) Lojitleri
    """

    def __init__(self, girdi_kanali: int = 3, sinif_sayisi: int = 10, taban_kanal: int = 32) -> None:
        super().__init__()
        self.girdi_kanali = girdi_kanali
        self.sinif_sayisi = sinif_sayisi

        # Kök Katman (Stem)
        self.kok = nn.Sequential(
            nn.Conv2d(girdi_kanali, taban_kanal, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2
        )

        # Evrişim & Rezidüel Aşamaları
        self.asama1 = nn.Sequential(
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1, bias=False),  # H/4, W/4
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(inplace=True),
            ResidualBlok(taban_kanal * 2)
        )

        self.asama2 = nn.Sequential(
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1, bias=False),  # H/8, W/8
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(inplace=True),
            ResidualBlok(taban_kanal * 4)
        )

        # Havuzlama ve Sınıflandırma Başlığı
        self.kuresel_havuz = nn.AdaptiveAvgPool2d((1, 1))
        self.siniflandirici = nn.Sequential(
            nn.Flatten(),
            nn.Linear(taban_kanal * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, sinif_sayisi)
        )

        self._agirliklari_ilklendir()

    def _agirliklari_ilklendir(self) -> None:
        """Kaiming Normal ilklendirme ile sayısal kararlılık sağlar."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri yayılım fonksiyonu."""
        x = self.kok(x)
        x = self.asama1(x)
        x = self.asama2(x)
        x = self.kuresel_havuz(x)
        lojitter = self.siniflandirici(x)
        return lojitter
