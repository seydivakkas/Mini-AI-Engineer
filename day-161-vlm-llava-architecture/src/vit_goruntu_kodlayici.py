"""
Vision Transformer (ViT) Görüntü Kodlayıcı Modülü (Day 161 - FAZ 9).
Görüntüyü 14x14 patch'lere bölerek d_vision boyutunda görsel token embedding'leri üretir.
"""

import torch
import torch.nn as nn


class ViTGoruntuKodlayici(nn.Module):
    """Vision Transformer (CLIP-ViT-L/14 tarzı) Görüntü Kodlayıcısı."""

    def __init__(
        self,
        goruntu_boyutu: int = 224,
        patch_boyutu: int = 14,
        in_kanallar: int = 3,
        d_vision: int = 768,
        katman_sayisi: int = 4,
        kafa_sayisi: int = 8,
    ):
        super().__init__()
        self.goruntu_boyutu = goruntu_boyutu
        self.patch_boyutu = patch_boyutu
        self.d_vision = d_vision
        self.num_patches = (goruntu_boyutu // patch_boyutu) ** 2  # (224/14)^2 = 256 patch

        # Patch Embedding: 2D Konvolüsyon ile patch vektörleştirme
        self.patch_proj = nn.Conv2d(
            in_channels=in_kanallar,
            out_channels=d_vision,
            kernel_size=patch_boyutu,
            stride=patch_boyutu,
        )

        # Pozisyonel Kodlama ve CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_vision))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_vision) * 0.02)

        # Transformer Encoder Blokları
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_vision,
            nhead=kafa_sayisi,
            dim_feedforward=d_vision * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=katman_sayisi)
        self.ln_post = nn.LayerNorm(d_vision)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, 3, 224, 224)
        Çıktı: (Batch, Num_Patches, d_vision) -> CLS token hariç patch tokenları (256 adet)
        """
        B = x.shape[0]

        # Patch Proj: (B, d_vision, H/P, W/P) -> (B, num_patches, d_vision)
        x = self.patch_proj(x).flatten(2).transpose(1, 2)

        # CLS Token Ekle
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 257, d_vision)

        # Pozisyonel Embedding Ekle
        x = x + self.pos_embed

        # Transformer İleri Geçiş
        x = self.transformer(x)
        x = self.ln_post(x)

        # LLaVA mimarisinde CLS token atılır, sadece patch tokenları (256 token) LLM'e gider
        patch_tokens = x[:, 1:, :]  # (B, 256, d_vision)
        return patch_tokens
