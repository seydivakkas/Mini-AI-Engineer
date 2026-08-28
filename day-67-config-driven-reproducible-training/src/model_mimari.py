"""
Konfigurasyona Duyarli Moduler Sinir Agi (Config-Driven VisionNet)
=================================================================
Pydantic ModelKonfigurasyonu nesnesi ile dinamik olarak sekillenen,
Kaiming/Xavier ilklendirmeli ve deterministik calisan konvolusyonel model.
"""

from typing import Dict, Any
import hashlib
import torch
import torch.nn as nn
from src.konfigurasyon_semasi import ModelKonfigurasyonu


class ResidualBlok(nn.Module):
    """Konfigurasyon uyumlu artık (Residual) katman."""

    def __init__(self, kanallar: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(kanallar, kanallar, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(kanallar)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(kanallar, kanallar, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(kanallar)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        artik = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + artik)
        return out


class ModulerVisionNet(nn.Module):
    """
    Konfigurasyondan parametre alan moduler gorsel siniflandirma agi.
    """

    def __init__(self, cfg: ModelKonfigurasyonu) -> None:
        super().__init__()
        self.cfg = cfg

        c_in = cfg.girdi_kanali
        c_base = cfg.taban_kanal
        num_classes = cfg.sinif_sayisi
        dropout = cfg.dropout_orani

        # Kok Evrisim
        self.kok = nn.Sequential(
            nn.Conv2d(c_in, c_base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Reziduel Bloklar
        self.blok1 = nn.Sequential(
            nn.Conv2d(c_base, c_base * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_base * 2),
            nn.ReLU(inplace=True),
            ResidualBlok(c_base * 2, dropout=dropout)
        )

        self.kuresel_havuz = nn.AdaptiveAvgPool2d((1, 1))
        self.siniflandirici = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_base * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

        self._ilklendir()

    def _ilklendir(self) -> None:
        """Deterministik ve kararlı Kaiming ağırlık ilklendirmesi."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kok(x)
        x = self.blok1(x)
        x = self.kuresel_havuz(x)
        lojitter = self.siniflandirici(x)
        return lojitter

    def agirlik_hashi_al(self) -> str:
        """Tum agirlik tensörlerinin SHA256 ozetini cikararak determinizm denetler."""
        hasher = hashlib.sha256()
        for name, param in sorted(self.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()
