"""
Dondurulmuş Katmanlarla L2-Normalize Embedding Çıkarıcı (Frozen Backbone Feature Extractor).
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MiniVisionTransformerOmurga(nn.Module):
    """Görsel semantik temsil çıkarımı için hafif Vision Transformer (ViT) omurgası."""

    def __init__(self, in_channels: int = 3, img_size: int = 32, patch_size: int = 4, embed_dim: int = 256, depth: int = 3, num_heads: int = 4):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # (B, C, H, W) -> (B, Embed, N_patches_h, N_patches_w) -> (B, N_patches, Embed)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        # [CLS] tokeni semantik embedding olarak döndürülür
        return x[:, 0]


class ResNetOzellikOmurgasi(nn.Module):
    """Görsel semantik temsil çıkarımı için konvolüsyonel ResNet-tarzı omurga."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.giris = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.katman1 = self._blok_olustur(64, 128, stride=2)
        self.katman2 = self._blok_olustur(128, 256, stride=2)
        self.katman3 = self._blok_olustur(256, feature_dim, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def _blok_olustur(self, in_c: int, out_c: int, stride: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.giris(x)
        x = self.katman1(x)
        x = self.katman2(x)
        x = self.katman3(x)
        x = self.gap(x)
        return torch.flatten(x, 1)


class OmurgaModelFabrikasi:
    """Farklı transfer learning omurgalarını oluşturan fabrika sınıfı."""

    @staticmethod
    def uret(model_turu: str = "resnet", **kwargs) -> nn.Module:
        if model_turu.lower() == "resnet":
            return ResNetOzellikOmurgasi(feature_dim=kwargs.get("feature_dim", 512))
        elif model_turu.lower() in ["vit", "minivit"]:
            return MiniVisionTransformerOmurga(
                embed_dim=kwargs.get("embed_dim", 256),
                depth=kwargs.get("depth", 3),
                num_heads=kwargs.get("num_heads", 4)
            )
        else:
            raise ValueError(f"Bilinmeyen model türü: {model_turu}")


class DondurulmusEmbeddingEkstraktoru(nn.Module):
    """Ağırlıkları dondurulmuş bir omurgadan L2-normalize özellik vektörleri (embeddings) çıkaran motor."""

    def __init__(self, backbone: nn.Module, normalize: bool = True, device: Optional[str] = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone.to(self.device)
        self.normalize = normalize

        # 1. Tüm katmanların gradyanlarını dondur (Freeze Parameters)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş yapar ve isteğe bağlı L2-normalizasyonu uygular."""
        ham_vektor = self.backbone(x)
        if self.normalize:
            return F.normalize(ham_vektor, p=2, dim=-1, eps=1e-12)
        return ham_vektor

    def cikart(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """DataLoader üzerindeki tüm veri setini toplu (batched) olarak işleyip NumPy embedding matrisi üretir."""
        embedding_list = []
        etiket_list = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                emb = self.forward(inputs)
                embedding_list.append(emb.cpu().numpy())
                etiket_list.append(targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets))

        tum_embeddingler = np.concatenate(embedding_list, axis=0).astype(np.float32)
        tum_etiketler = np.concatenate(etiket_list, axis=0).astype(np.int64)
        return tum_embeddingler, tum_etiketler
