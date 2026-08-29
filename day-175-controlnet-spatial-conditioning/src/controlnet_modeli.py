"""
ControlNet Modeli (Trainable Clone + Zero-Convolutions) Modülü (Day 175 - FAZ 9).
Dondurulmuş ana UNet'e kenar, derinlik ve poz rehberliğini sıfır-konvolüsyonlarla bağlar.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .zero_convolution import ZeroConv2d


class ControlNetModeli(nn.Module):
    """Mekansal Kontrol İpuçlarını (Canny/Depth/Pose) Difüzyona Enjekte Eden ControlNet."""

    def __init__(
        self,
        in_channels: int = 4,
        hint_channels: int = 3,
        base_channels: int = 64,
        context_dim: int = 256,
    ):
        super().__init__()
        # 1. Koşul Giriş Kökü (Conditioning Hint Embedding: Canny / Depth / Pose)
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
        )

        # 2. Giriş Sıfır-Konvolüsyonu
        self.zero_conv_in = ZeroConv2d(in_channels, base_channels)

        # 3. Eğitilebilir Down-Encoder Blokları
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.mid_block = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)

        # 4. Çıkış Sıfır-Konvolüsyonları (Ana Dondurulmuş UNet'e eklenecek köprüler)
        self.zero_convs = nn.ModuleList([
            ZeroConv2d(base_channels * 2, base_channels * 2),
            ZeroConv2d(base_channels * 4, base_channels * 4),
            ZeroConv2d(base_channels * 4, base_channels * 4),
        ])

    def forward(
        self,
        z_t: torch.Tensor,
        hint: torch.Tensor,
        control_weight: float = 1.0,
    ) -> List[torch.Tensor]:
        """
        z_t: [B, 4, H, W] (Gürültülü VAE Tensörü)
        hint: [B, 3, H, W] (Mekansal Koşul Görüntüsü: Canny / Depth / Pose)
        Döner: control_residuals Listesi [res1, res2, res3]
        """
        # Koşul ipucunu işle ve z_t ile topla
        hint_emb = self.input_hint_block(hint)
        x = self.zero_conv_in(z_t + hint_emb)

        # Down 1
        d1 = F.gelu(self.down1(x))
        out1 = self.zero_convs[0](d1) * control_weight

        # Down 2
        d2 = F.gelu(self.down2(d1))
        out2 = self.zero_convs[1](d2) * control_weight

        # Mid Block
        mid = F.gelu(self.mid_block(d2))
        out_mid = self.zero_convs[2](mid) * control_weight

        return [out1, out2, out_mid]
