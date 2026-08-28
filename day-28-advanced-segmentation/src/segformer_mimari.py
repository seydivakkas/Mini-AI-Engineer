"""
SegFormer: Basit ve Verimli Vision Transformer Tabanlı Anlamsal Bölütleme Mimarisi.
(Xie et al., NeurIPS 2021 - Mix Transformer Encoder + All-MLP Decoder)
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MekansalIndirgemeliDikkat(nn.Module):
    """
    Efficient Spatial Reduction Attention (SRA):
    Key ve Value dizilerini R oranıyla alt-örnekleyerek O(N^2) karmaşıklığını O(N^2 / R) seviyesine düşürür.
    """

    def __init__(self, d_model: int, num_heads: int = 4, sr_ratio: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.proj = nn.Linear(d_model, d_model)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out


class MixFFN(nn.Module):
    """
    Mix-FFN: Konumsal kodlama (Positional Encoding) yerine
    3x3 Derinlemesine Evrişim (Depthwise Conv) kullanarak keyfi çözünürlüklere uyum sağlar.
    """

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = self.fc1(x)
        # 3x3 Conv uzamsal boyut üzerinde
        x = x.permute(0, 2, 1).reshape(b, -1, h, w)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.fc2(x)
        return x


class TransformerBlogu(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, sr_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MekansalIndirgemeliDikkat(d_model, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MixFFN(d_model, d_model * mlp_ratio)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


class SegFormerModeli(nn.Module):
    """
    SegFormer Mimarisi:
    Çok Ölçekli Hiyerarşik Transformer Encoder + All-MLP Hafif Decoder.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 5, embed_dims: List[int] = [32, 64, 128, 256]):
        super().__init__()
        self.embed_dims = embed_dims

        # 4 Kademeli Patch Birleştirme (Overlapping Patch Merging)
        self.patch_embed1 = nn.Conv2d(in_channels, embed_dims[0], kernel_size=7, stride=4, padding=3)
        self.patch_embed2 = nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1)
        self.patch_embed3 = nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1)
        self.patch_embed4 = nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1)

        # Transformer Blokları
        self.block1 = TransformerBlogu(embed_dims[0], num_heads=1, sr_ratio=4)
        self.block2 = TransformerBlogu(embed_dims[1], num_heads=2, sr_ratio=2)
        self.block3 = TransformerBlogu(embed_dims[2], num_heads=4, sr_ratio=1)
        self.block4 = TransformerBlogu(embed_dims[3], num_heads=8, sr_ratio=1)

        # All-MLP Decoder Katmanları
        decoder_dim = 128
        self.linear_c1 = nn.Linear(embed_dims[0], decoder_dim)
        self.linear_c2 = nn.Linear(embed_dims[1], decoder_dim)
        self.linear_c3 = nn.Linear(embed_dims[2], decoder_dim)
        self.linear_c4 = nn.Linear(embed_dims[3], decoder_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (B, 3, H, W)
        Çıktı: (B, num_classes, H, W)
        """
        b, _, orig_h, orig_w = x.shape

        # Kademe 1: H/4 x W/4
        f1 = self.patch_embed1(x)
        _, c1, h1, w1 = f1.shape
        f1 = self.block1(f1.flatten(2).permute(0, 2, 1), h1, w1).permute(0, 2, 1).reshape(b, c1, h1, w1)

        # Kademe 2: H/8 x W/8
        f2 = self.patch_embed2(f1)
        _, c2, h2, w2 = f2.shape
        f2 = self.block2(f2.flatten(2).permute(0, 2, 1), h2, w2).permute(0, 2, 1).reshape(b, c2, h2, w2)

        # Kademe 3: H/16 x W/16
        f3 = self.patch_embed3(f2)
        _, c3, h3, w3 = f3.shape
        f3 = self.block3(f3.flatten(2).permute(0, 2, 1), h3, w3).permute(0, 2, 1).reshape(b, c3, h3, w3)

        # Kademe 4: H/32 x W/32
        f4 = self.patch_embed4(f3)
        _, c4, h4, w4 = f4.shape
        f4 = self.block4(f4.flatten(2).permute(0, 2, 1), h4, w4).permute(0, 2, 1).reshape(b, c4, h4, w4)

        # All-MLP Decoder
        _f1 = self.linear_c1(f1.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, -1, h1, w1)
        _f2 = self.linear_c2(f2.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, -1, h2, w2)
        _f3 = self.linear_c3(f3.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, -1, h3, w3)
        _f4 = self.linear_c4(f4.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, -1, h4, w4)

        # H1 x W1 (H/4 x W/4) çözünürlüğüne yukarı örnekleme
        _f2 = F.interpolate(_f2, size=(h1, w1), mode="bilinear", align_corners=False)
        _f3 = F.interpolate(_f3, size=(h1, w1), mode="bilinear", align_corners=False)
        _f4 = F.interpolate(_f4, size=(h1, w1), mode="bilinear", align_corners=False)

        # Özellik Birleştirme & Sınıflandırma
        fused = self.linear_fuse(torch.cat([_f1, _f2, _f3, _f4], dim=1))
        fused = self.dropout(fused)
        logits = self.linear_pred(fused)

        # Orijinal (H, W) boyutuna nihai yukarı örnekleme
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return logits
