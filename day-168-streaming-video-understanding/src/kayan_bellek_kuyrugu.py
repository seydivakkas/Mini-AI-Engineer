"""
Kayan Bellek Kuyruğu (Sliding Window Memory Queue / Ring Buffer) Modülü (Day 168 - FAZ 9).
Sürekli canlı video akışını sabit kapasiteli (FIFO) dairesel tamponda tutar.
"""

from typing import List, Any, Optional
from collections import deque
import torch


class KayanBellekKuyrugu:
    """Canlı video akış karelerini tutan FIFO Dairesel Bellek Tamponu."""

    def __init__(self, maks_kapasite: int = 16):
        self.maks_kapasite = maks_kapasite
        self.tampon = deque(maxlen=maks_kapasite)

    def kare_ekle(self, kare_tensor: torch.Tensor, zaman_damgasi: float, metadata: Optional[dict] = None):
        """Yeni bir video karesini zaman damgasıyla kuyruğa ekler."""
        eleman = {
            "kare": kare_tensor,
            "zaman_damgasi": zaman_damgasi,
            "metadata": metadata or {},
        }
        self.tampon.append(eleman)

    def aktif_pencereyi_getir(self) -> List[dict]:
        """Tampondaki tüm güncel kareleri zaman sırasıyla döner."""
        return list(self.tampon)

    def tensor_yigini_olustur(self) -> torch.Tensor:
        """Tampondaki kareleri [1, T, N, D] veya [1, T, C, H, W] tensor yığınına çevirir."""
        if not self.tampon:
            raise ValueError("Kayan bellek kuyruğu henüz boş!")
        kareler = [e["kare"] for e in self.tampon]
        return torch.stack(kareler, dim=0).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.tampon)
