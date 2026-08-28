"""
Üretim Modeli ve Şema Doğrulama Sözleşmesi (Model Signature)
------------------------------------------------------------
Üretim ortamına girecek modellerin girdi/çıktı tensör boyut ve veri tipi sözleşmeleri.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class UretimVisionModeli(nn.Module):
    """
    Üretime hazır 3 Aşamalı Konvolüsyonel Görme Modeli
    """
    def __init__(self, giris_kanali: int = 3, sinif_sayisi: int = 10, taban_kanal: int = 32):
        super().__init__()
        self.giris_kanali = giris_kanali
        self.sinif_sayisi = sinif_sayisi
        self.taban_kanal = taban_kanal

        self.omurga = nn.Sequential(
            nn.Conv2d(giris_kanali, taban_kanal, kernel_size=3, padding=1),
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(),
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(),
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.kafa = nn.Linear(taban_kanal * 4, sinif_sayisi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.sema_dogrula(x)
        x = self.omurga(x).flatten(1)
        return self.kafa(x)

    def sema_dogrula(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Girdi tensörü 4 boyutlu [B, C, H, W] olmalıdır. Alınan: {x.shape}")
        if x.size(1) != self.giris_kanali:
            raise ValueError(f"Kanal sayısı {self.giris_kanali} olmalı. Alınan: {x.size(1)}")

    @classmethod
    def model_semasi(cls) -> Dict[str, Any]:
        return {
            "inputs": [{"name": "gorsel_tensor", "type": "tensor", "shape": [-1, 3, 32, 32], "dtype": "float32"}],
            "outputs": [{"name": "sinif_logitleri", "type": "tensor", "shape": [-1, 10], "dtype": "float32"}]
        }

    @classmethod
    def egit(
        cls,
        model: nn.Module,
        loader: DataLoader,
        epok_sayisi: int = 6,
        lr: float = 2e-3,
        cihaz: str = "cpu"
    ) -> None:
        model = model.to(cihaz).train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epok_sayisi):
            for x, y in loader:
                x, y = x.to(cihaz), y.to(cihaz)
                opt.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                opt.step()
