"""
Sıfırdan Görsel Yama Gömülme Katmanı (Patch Embedding Layer)
------------------------------------------------------------
Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words" makalesindeki gibi,
2D görseli örtüşmeyen PxP yamalara bölen ve doğrusal olarak D boyutuna projeksiyonlayan modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple
import torch
import torch.nn as nn


class YamaGomulmeKatmani(nn.Module):
    """
    2D Görseli Düzleştirilmiş Vektör Dizisine (Patch Sequence) Dönüştüren Katman:
    x ∈ ℝ^(B × C × H × W) ──> x_p ∈ ℝ^(B × N × D)
    burada N = (H/P) * (W/P) toplam yama sayısıdır.
    """
    def __init__(
        self,
        gorsel_boyutu: int = 32,
        yama_boyutu: int = 4,
        giris_kanali: int = 3,
        gomulme_boyutu: int = 64
    ):
        super().__init__()
        assert gorsel_boyutu % yama_boyutu == 0, (
            f"Görsel boyutu ({gorsel_boyutu}) yama boyutuna ({yama_boyutu}) tam bölünmelidir!"
        )

        self.gorsel_boyutu = gorsel_boyutu
        self.yama_boyutu = yama_boyutu
        self.giris_kanali = giris_kanali
        self.gomulme_boyutu = gomulme_boyutu

        self.grid_h = gorsel_boyutu // yama_boyutu
        self.grid_w = gorsel_boyutu // yama_boyutu
        self.toplam_yama_sayisi = self.grid_h * self.grid_w

        # Conv2d projeksiyonu: kernel=P, stride=P örtüşmeyen yamalar üretir
        self.projeksiyon = nn.Conv2d(
            in_channels=giris_kanali,
            out_channels=gomulme_boyutu,
            kernel_size=yama_boyutu,
            stride=yama_boyutu
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, C, H, W)
        Çıktı: (Batch, N, Gomulme_Boyutu)
        """
        b, c, h, w = x.shape
        assert h == self.gorsel_boyutu and w == self.gorsel_boyutu, (
            f"Girdi görsel boyutu ({h}x{w}) beklenen ({self.gorsel_boyutu}x{self.gorsel_boyutu}) ile uyuşmuyor!"
        )

        # (B, C, H, W) -> (B, D, grid_h, grid_w)
        x_proj = self.projeksiyon(x)

        # (B, D, grid_h, grid_w) -> (B, D, N) -> (B, N, D)
        x_flat = x_proj.flatten(2).transpose(1, 2)
        return x_flat
