"""
Asenkron GPU Veri On-Yukleyicisi (CUDA Stream Data Prefetcher)
=============================================================
Host-to-Device (CPU -> GPU PCIe) bellek transferini CUDA Stream kullanarak
GPU'daki ileri/geri hesaplama ile eszamanli (overlapped) yuruten yuksek performansli on-yukleyici.
"""

from typing import Iterator, Tuple, Optional, Any
import torch
from torch.utils.data import DataLoader


class CUDAPrefetcher:
    """
    DataLoader uzerinde calisan, sonraki mini-batch'i arka planda GPU'ya tasiyan asenkron iterator.
    """

    def __init__(self, dataloader: DataLoader, cihaz: Optional[torch.device] = None) -> None:
        self.loader = dataloader
        self.cihaz = cihaz or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_aktif = (self.cihaz.type == "cuda")

        if self.cuda_aktif:
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        self.iterator: Optional[Iterator] = None
        self.siradaki_girdi: Optional[torch.Tensor] = None
        self.siradaki_hedef: Optional[torch.Tensor] = None

    def __iter__(self) -> "CUDAPrefetcher":
        self.iterator = iter(self.loader)
        self._on_yukle()
        return self

    def _on_yukle(self) -> None:
        """Sonraki mini-batch'i arka plan stream'inde GPU'ya asenkron taşır."""
        try:
            self.siradaki_girdi, self.siradaki_hedef = next(self.iterator)
        except StopIteration:
            self.siradaki_girdi = None
            self.siradaki_hedef = None
            return

        if self.cuda_aktif and self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.siradaki_girdi = self.siradaki_girdi.to(self.cihaz, non_blocking=True)
                self.siradaki_hedef = self.siradaki_hedef.to(self.cihaz, non_blocking=True)
        else:
            self.siradaki_girdi = self.siradaki_girdi.to(self.cihaz)
            self.siradaki_hedef = self.siradaki_hedef.to(self.cihaz)

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.siradaki_girdi is None:
            raise StopIteration

        if self.cuda_aktif and self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        girdi = self.siradaki_girdi
        hedef = self.siradaki_hedef

        # Bir sonraki batch'i arka planda yükle
        self._on_yukle()

        return girdi, hedef

    def __len__(self) -> int:
        return len(self.loader)
