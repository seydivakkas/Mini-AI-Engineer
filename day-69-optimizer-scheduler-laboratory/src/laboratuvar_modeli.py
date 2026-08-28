"""
Laboratuvar Sinir Agi ve Parametre Grubu Ayrıştırıcısı
=====================================================
Deneyler icin kullanilan moduler konvolusyonel sinir agi ve
Weight Decay'in bias ve normalizasyon katmanlarina uygulanmasini engelleyen
endustri standardi parametre grubu ayristiricisi (Parameter Group Splitting).
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn


class DeneySinirAgi(nn.Module):
    """Laboratuvar deneyleri için optimize edilmiş kompakt Residual VisionNet."""

    def __init__(self, girdi_kanali: int = 3, sinif_sayisi: int = 10, taban_kanal: int = 32) -> None:
        super().__init__()
        self.giris = nn.Sequential(
            nn.Conv2d(girdi_kanali, taban_kanal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(inplace=True)
        )

        self.blok1 = nn.Sequential(
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(inplace=True)
        )

        self.blok2 = nn.Sequential(
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(inplace=True)
        )

        self.havuz = nn.AdaptiveAvgPool2d((1, 1))
        self.siniflandirici = nn.Linear(taban_kanal * 4, sinif_sayisi)

        self._ilklendir()

    def _ilklendir(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.giris(x)
        x = self.blok1(x)
        x = self.blok2(x)
        x = self.havuz(x)
        x = torch.flatten(x, 1)
        return self.siniflandirici(x)


def parametre_gruplari_ayristir(
    model: nn.Module,
    weight_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Model parametrelerini 2 gruba ayırır:
    1. Ağırlıklar (Conv2D, Linear weights) -> Weight Decay uygulanır.
    2. Bias'lar ve Normalizasyon katmanları (BatchNorm/LayerNorm) -> Weight Decay = 0.0.
    """
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D tensörler (bias veya norm ölçek parametreleri)
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
