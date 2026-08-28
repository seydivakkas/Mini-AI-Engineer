"""
Sıfırdan Mini Vision Transformer (MiniViT) Mimarisi
---------------------------------------------------
Görsel Yama Gömülme (Patch Embedding), Öğrenilebilir [CLS] Token, Pozisyonel Gömülmeler,
L-Katmanlı Pre-LN Transformer Encoder Bloğu ve MLP Sınıflandırma Kafası içeren tam model.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_gocume import YamaGomulmeKatmani


class OzelLayerNorm(nn.Module):
    """Sıfırdan Katman Normalizasyonu"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ortalama = x.mean(dim=-1, keepdim=True)
        varyans = x.var(dim=-1, keepdim=True, unbiased=False)
        return ((x - ortalama) / torch.sqrt(varyans + self.eps)) * self.gamma + self.beta


class CokKafaliOzDikkat(nn.Module):
    """Çok Kafalı Öz Dikkat (MHSA)"""
    def __init__(self, model_boyutu: int = 64, kafa_sayisi: int = 4, dropout_orani: float = 0.0):
        super().__init__()
        assert model_boyutu % kafa_sayisi == 0
        self.model_boyutu = model_boyutu
        self.kafa_sayisi = kafa_sayisi
        self.d_k = model_boyutu // kafa_sayisi

        self.w_q = nn.Linear(model_boyutu, model_boyutu, bias=True)
        self.w_k = nn.Linear(model_boyutu, model_boyutu, bias=True)
        self.w_v = nn.Linear(model_boyutu, model_boyutu, bias=True)
        self.w_o = nn.Linear(model_boyutu, model_boyutu, bias=True)
        self.dropout = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = x.size()
        q = self.w_q(x).view(b, n, self.kafa_sayisi, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.kafa_sayisi, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.kafa_sayisi, self.d_k).transpose(1, 2)

        skorlar = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        dikkat_haritalari = F.softmax(skorlar, dim=-1)
        agirliklar = self.dropout(dikkat_haritalari) if self.dropout is not None else dikkat_haritalari

        out = torch.matmul(agirliklar, v).transpose(1, 2).contiguous().view(b, n, self.model_boyutu)
        return self.w_o(out), dikkat_haritalari


class BeslemeliIleriAg(nn.Module):
    """Position-wise FFN (MLP)"""
    def __init__(self, model_boyutu: int = 64, genisleme: int = 4, dropout_orani: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(model_boyutu, model_boyutu * genisleme)
        self.akt = nn.GELU()
        self.fc2 = nn.Linear(model_boyutu * genisleme, model_boyutu)
        self.drop = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.akt(self.fc1(x)))))


class TransformerEncoderBlogu(nn.Module):
    """Pre-LN Transformer Encoder Bloğu"""
    def __init__(self, model_boyutu: int = 64, kafa_sayisi: int = 4, genisleme: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = OzelLayerNorm(model_boyutu)
        self.dikkat = CokKafaliOzDikkat(model_boyutu, kafa_sayisi, dropout)
        self.ln2 = OzelLayerNorm(model_boyutu)
        self.ffn = BeslemeliIleriAg(model_boyutu, genisleme, dropout)
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, att_map = self.dikkat(self.ln1(x))
        x = x + self.drop(attn_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.drop(ffn_out)
        return x, att_map


class MiniVisionTransformer(nn.Module):
    """
    Sıfırdan Mini Vision Transformer (MiniViT)
    """
    def __init__(
        self,
        gorsel_boyutu: int = 32,
        yama_boyutu: int = 4,
        giris_kanali: int = 3,
        sinif_sayisi: int = 10,
        gomulme_boyutu: int = 64,
        derinlik: int = 4,
        kafa_sayisi: int = 4,
        mlp_orani: int = 4,
        dropout_orani: float = 0.1
    ):
        super().__init__()
        self.gorsel_boyutu = gorsel_boyutu
        self.yama_boyutu = yama_boyutu
        self.gomulme_boyutu = gomulme_boyutu
        self.sinif_sayisi = sinif_sayisi

        # 1. Yama Gömülme Katmanı
        self.patch_embed = YamaGomulmeKatmani(
            gorsel_boyutu=gorsel_boyutu,
            yama_boyutu=yama_boyutu,
            giris_kanali=giris_kanali,
            gomulme_boyutu=gomulme_boyutu
        )
        num_patches = self.patch_embed.toplam_yama_sayisi

        # 2. [CLS] Sınıflandırma Token'ı (Öğrenilebilir)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, gomulme_boyutu))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 3. 1D Pozisyonel Gömülmeler (1 + N yama için)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, gomulme_boyutu))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else nn.Identity()

        # 4. L Katmanlı Transformer Encoder Yığını
        self.bloklar = nn.ModuleList([
            TransformerEncoderBlogu(
                model_boyutu=gomulme_boyutu,
                kafa_sayisi=kafa_sayisi,
                genisleme=mlp_orani,
                dropout=dropout_orani
            )
            for _ in range(derinlik)
        ])

        # 5. Çıkış Normalizasyonu ve MLP Sınıflandırma Kafası
        self.norm = OzelLayerNorm(gomulme_boyutu)
        self.head = nn.Linear(gomulme_boyutu, sinif_sayisi)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Öznitelik çıkarımı: Girdi -> Yamalar + [CLS] + PE -> Encoder Blokları -> Norm
        """
        b = x.shape[0]
        # (B, C, H, W) -> (B, N, D)
        x_patches = self.patch_embed(x)

        # [CLS] token'ı batch boyutuna genişlet ve başa ekle: (B, 1 + N, D)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x_patches), dim=1)

        # Pozisyonel kodlama ekle
        x = self.pos_drop(x + self.pos_embed)

        dikkat_listesi = []
        for blok in self.bloklar:
            x, att = blok(x)
            dikkat_listesi.append(att)

        x = self.norm(x)
        return x, dikkat_listesi

    def forward(
        self,
        x: torch.Tensor,
        dikkat_haritalarini_don: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Uçtan Uca İleri Geçiş.
        Çıktı: Logitler (Batch, Sınıf_Sayısı), [Opsiyonel Dikkat Haritaları]
        """
        x_features, dikkat_listesi = self.forward_features(x)
        
        # [CLS] token temsilini (0. indeks) sınıflandırma kafasına ilet
        cls_temsili = x_features[:, 0]
        logitler = self.head(cls_temsili)

        if dikkat_haritalarini_don:
            return logitler, dikkat_listesi
        return logitler
