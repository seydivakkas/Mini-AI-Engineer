"""
Sentetik Gorsel Veri Seti ve Donusum Entegratoru (Vision Dataset)
================================================================
NumPy tabanli sentetik RGB gorseller ureten ve Albumentations / torchvision
donusum zincirlerini sorunsuz baglayan yuksek verimli PyTorch Dataset sinifi.
"""

from typing import Optional, Callable, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset


class SentetikGorselVeriSeti(Dataset):
    """
    Belirtilen ornek adedi ve cozunurlukte sentetik gorseller ureten Dataset sinifi.
    """

    def __init__(
        self,
        ornek_sayisi: int = 1000,
        gorsel_boyutu: Tuple[int, int] = (64, 64),
        sinif_sayisi: int = 10,
        tohum: int = 42,
        donusum: Optional[Callable[[np.ndarray], torch.Tensor]] = None
    ) -> None:
        self.ornek_sayisi = ornek_sayisi
        self.gorsel_boyutu = gorsel_boyutu
        self.sinif_sayisi = sinif_sayisi
        self.donusum = donusum

        # Deterministik sentetik veri havuzu ilklendirmesi
        rng = np.random.RandomState(tohum)
        H, W = gorsel_boyutu

        # HWC uint8 dizisi
        self.veriler = rng.randint(0, 256, size=(ornek_sayisi, H, W, 3), dtype=np.uint8)
        self.etiketler = rng.randint(0, sinif_sayisi, size=(ornek_sayisi,), dtype=np.int64)

    def __len__(self) -> int:
        return self.ornek_sayisi

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        gorsel_np = self.veriler[index]
        etiket = int(self.etiketler[index])

        if self.donusum is not None:
            gorsel_tensor = self.donusum(gorsel_np)
        else:
            # Standart CHW FloatTensor donusumu
            gorsel_tensor = torch.from_numpy(gorsel_np.transpose(2, 0, 1)).float() / 255.0

        return gorsel_tensor, etiket
