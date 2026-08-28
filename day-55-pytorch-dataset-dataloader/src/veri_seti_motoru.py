"""
Hızlı Sentetik Görsel Veri Seti ve Çoklu İş Parçacığı Tohumlayıcısı (Fast Dataset & Worker Seeding).
"""

from typing import Tuple
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def worker_init_fn(worker_id: int) -> None:
    """Her DataLoader alt sürecine (worker) bağımsız ve tekrarlanabilir rastgele tohum atar."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class HizliSentetikGorselVeriSeti(Dataset):
    """C-bellek dizisi ve ayarlanabilir I/O simülasyonu ile optimize edilmiş PyTorch veri seti."""

    def __init__(
        self,
        num_samples: int = 2400,
        channels: int = 3,
        height: int = 64,
        width: int = 64,
        num_classes: int = 10,
        simule_io_ms: float = 0.5,
        seed: int = 42
    ):
        super().__init__()
        self.num_samples = num_samples
        self.channels = channels
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.simule_io_ms = simule_io_ms

        np.random.seed(seed)
        # Bellekte bitişik (C-contiguous) tekil tampon tahsisi
        self.veriler = np.ascontiguousarray(
            np.random.randn(num_samples, channels, height, width).astype(np.float32)
        )
        self.etiketler = np.ascontiguousarray(
            np.random.randint(0, num_classes, size=num_samples).astype(np.int64)
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sıfır-kopyalama (zero-copy) tensör dönüşümü
        gorsel_tensor = torch.from_numpy(self.veriler[idx])
        # Gerçekçi CPU artırma / normalizasyon yükü simülasyonu
        if self.simule_io_ms > 0:
            gorsel_tensor = (gorsel_tensor - 0.5) / 0.5
        etiket_tensor = torch.tensor(self.etiketler[idx], dtype=torch.long)
        return gorsel_tensor, etiket_tensor
