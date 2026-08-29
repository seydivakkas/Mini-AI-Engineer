"""
DiT (Diffusion Transformer) Model Mimarisi (Day 177 - FAZ 9).
Patchify, 2D Positional Embeddings, N adet DiTBlock ve Unpatchify çıkış başlığı.
"""

from typing import Dict, Any
import math
import torch
import torch.nn as nn
from .dit_blok import DiTBlock
from .adaln_zero import modulate


class TimestepEmbedder(nn.Module):
    """Sinüzoidal Zaman Adımı Gömücüsü."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)


class DiffusionTransformer(nn.Module):
    """
    Sora & Flux omurgası olan Saf Diffusion Transformer (DiT).
    Girdi: [B, C, H, W] -> Patchify -> [B, N, D] -> DiT Blokları -> Unpatchify -> [B, C, H, W]
    """

    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        cond_size: int = 128,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (input_size // patch_size) ** 2

        # 1. Patchify Projeksiyonu
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # 2. Koşul Gömücüsü (Zaman adımı t)
        self.t_embedder = TimestepEmbedder(cond_size)

        # 3. Ardışık DiT Blokları
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, cond_size, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # 4. Son Katman: adaLN ve Doğrusal Unpatchify Başlığı
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, 2 * hidden_size, bias=True)
        )
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)

        # Başlangıç ağırlıkları sıfırlaması
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, patch_size**2 * out_channels] -> [B, out_channels, H, W]
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, H, W]
        t: [B] (Zaman adımı)
        """
        # Patchify ve Positional Embedding
        x = self.x_embedder(x).flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed

        # Koşullandırma
        c = self.t_embedder(t)

        # DiT Blokları
        for block in self.blocks:
            x = block(x, c)

        # Çıkış adaLN ve Unpatchify
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.final_linear(x)
        out = self.unpatchify(x)
        return out

    @classmethod
    def ornek_dit_karsilastirma_raporu(cls) -> Dict[str, Any]:
        """DiT model varyantları ve UNet karşılaştırma metrikleri."""
        return {
            "model_varyantlari": [
                {"model": "DiT-S/2 (Small)", "param_m": 33, "gflops": 18.2, "fid": 10.5, "aciklama": "Mobil & Kenar Cihazlar"},
                {"model": "DiT-B/2 (Base)", "param_m": 130, "gflops": 74.0, "fid": 4.8, "aciklama": "Hızlı Prototipleme"},
                {"model": "DiT-L/2 (Large)", "param_m": 458, "gflops": 260.0, "fid": 3.1, "aciklama": "Yüksek Kalite Üretim"},
                {"model": "DiT-XL/2 (X-Large)", "param_m": 675, "gflops": 384.0, "fid": 2.27, "aciklama": "Sora / SD3 / Flux Temeli (State-of-the-Art)"},
            ],
            "patch_boyut_analizi": [
                {"patch": "p=8", "token_sayisi": 16, "hiz_kat": "8.5x Hızlı", "kalite": "Düşük Detay (FID=18.4)"},
                {"patch": "p=4", "token_sayisi": 64, "hiz_kat": "3.2x Hızlı", "kalite": "Orta Detay (FID=7.2)"},
                {"patch": "p=2", "token_sayisi": 256, "hiz_kat": "1.0x (Referans)", "kalite": "Kusursuz Detay (FID=2.27)"},
            ],
            "unet_vs_dit_avantaji": "Konvolüsyonel İndüktif Önyargı Yok (Sıfır İnductive Bias), Transformer Ölçeklenme Yasası (Scaling Laws: Hesaplama Arttıkça FID Doğrusal İyileşir)",
        }
