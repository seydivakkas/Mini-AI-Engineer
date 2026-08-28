"""
Mini Vision Transformer (MiniViT) Mimarisi
------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class YamaGomulmeKatmani(nn.Module):
    def __init__(self, gorsel_boyutu: int = 32, yama_boyutu: int = 4, giris_kanali: int = 3, gomulme_boyutu: int = 64):
        super().__init__()
        assert gorsel_boyutu % yama_boyutu == 0
        self.gorsel_boyutu = gorsel_boyutu
        self.yama_boyutu = yama_boyutu
        self.grid_h = gorsel_boyutu // yama_boyutu
        self.toplam_yama_sayisi = self.grid_h * self.grid_h

        self.projeksiyon = nn.Conv2d(
            in_channels=giris_kanali,
            out_channels=gomulme_boyutu,
            kernel_size=yama_boyutu,
            stride=yama_boyutu
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projeksiyon(x).flatten(2).transpose(1, 2)


class OzelLayerNorm(nn.Module):
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
    def __init__(self, model_boyutu: int = 64, genisleme: int = 4, dropout_orani: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(model_boyutu, model_boyutu * genisleme)
        self.akt = nn.GELU()
        self.fc2 = nn.Linear(model_boyutu * genisleme, model_boyutu)
        self.drop = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.akt(self.fc1(x)))))


class TransformerEncoderBlogu(nn.Module):
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
        dropout_orani: float = 0.0
    ):
        super().__init__()
        self.gorsel_boyutu = gorsel_boyutu
        self.yama_boyutu = yama_boyutu
        self.gomulme_boyutu = gomulme_boyutu
        self.sinif_sayisi = sinif_sayisi

        self.patch_embed = YamaGomulmeKatmani(gorsel_boyutu, yama_boyutu, giris_kanali, gomulme_boyutu)
        num_patches = self.patch_embed.toplam_yama_sayisi

        self.cls_token = nn.Parameter(torch.zeros(1, 1, gomulme_boyutu))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, gomulme_boyutu))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout_orani) if dropout_orani > 0.0 else nn.Identity()

        self.bloklar = nn.ModuleList([
            TransformerEncoderBlogu(
                model_boyutu=gomulme_boyutu,
                kafa_sayisi=kafa_sayisi,
                genisleme=mlp_orani,
                dropout=dropout_orani
            )
            for _ in range(derinlik)
        ])

        self.norm = OzelLayerNorm(gomulme_boyutu)
        self.head = nn.Linear(gomulme_boyutu, sinif_sayisi)

    def forward(self, x: torch.Tensor, dikkat_haritalarini_don: bool = False):
        b = x.shape[0]
        x_patches = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = self.pos_drop(torch.cat((cls_tokens, x_patches), dim=1) + self.pos_embed)

        dikkat_listesi = []
        for blok in self.bloklar:
            x, att = blok(x)
            dikkat_listesi.append(att)

        cls_temsili = self.norm(x)[:, 0]
        logitler = self.head(cls_temsili)

        if dikkat_haritalarini_don:
            return logitler, dikkat_listesi
        return logitler
