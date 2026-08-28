"""PyTorch Evrişimli Sinir Ağı (CNN) Model Mimarisi.

Bu modül; görsel sınıflandırma için PyTorch nn.Module tabanlı,
modüler Conv2D + BatchNorm2D + ReLU + MaxPool2D ve Dropout katmanlarından
oluşan derin öğrenme mimarisini ve Kaiming/He ağırlık ilklendirmesini tanımlar.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class ConvBlok(nn.Module):
    """Evrişim, Batch Normalization, ReLU Aktivasyonu ve Havuzlama içeren temel blok."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,  # BatchNorm kullanıldığı için bias gereksizdir
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Girdi tensörünü bloktan geçirir."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class PyTorchVisionCNN(nn.Module):
    """Görsel sınıflandırma için 3 bloklu PyTorch CNN Modeli."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        dropout_rate: float = 0.3,
        input_size: Tuple[int, int] = (64, 64),
    ) -> None:
        """PyTorchVisionCNN mimarisini ilklendirir.

        Args:
            in_channels: Giriş kanal sayısı (RGB=3, Gri=1).
            num_classes: Sınıflandırılacak hedef sınıf sayısı (>= 2).
            dropout_rate: Dropout olasılık oranı ([0.0, 0.9]).
            input_size: Giriş görsel uzamsal boyutu (H, W).
        """
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes en az 2 olmalıdır, alınan: {num_classes}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate [0.0, 1.0) aralığında olmalıdır, alınan: {dropout_rate}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size

        # Evrişimsel Özellik Çıkarıcı (Feature Extractor)
        self.blok1 = ConvBlok(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.blok2 = ConvBlok(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.blok3 = ConvBlok(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # 3 havuzlama katmanından sonra uzamsal boyut: H / 8, W / 8
        feat_h = input_size[0] // 8
        feat_w = input_size[1] // 8
        flatten_dim = 128 * feat_h * feat_w

        # Sınıflandırıcı Başlığı (Classification Head)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

        # Ağırlıkları Kaiming/He yöntemiyle ilklendir
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Katman ağırlıklarını Kaiming He Normalizasyonu ile ilklendirir."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş (Forward Pass).

        Args:
            x: (B, C, H, W) şekline sahip PyTorch tensörü.

        Returns:
            torch.Tensor: (B, num_classes) şekline sahip işlenmemiş logitler (Logits).
        """
        x = self.blok1(x)
        x = self.blok2(x)
        x = self.blok3(x)
        logits = self.classifier(x)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Modelin toplam, eğitilebilir ve dondurulmuş parametre sayılarını döndürür."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
        }
