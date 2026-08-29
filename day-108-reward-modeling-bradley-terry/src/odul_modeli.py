"""
Skaler Ödül Modeli (Reward Model) Mimarisi (Day 108).
Transformer omurgası, Son Token (Last-Token/EOS) Havuzlaması ve 1D Skaler Ödül Projeksiyon Başlığı.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlok(nn.Module):
    """Standart Transformer Bloğu."""

    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)

        skor = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            skor = skor + mask
        attn = F.softmax(skor, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)

        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class OdulModeli(nn.Module):
    """
    LLM Alignment için Skaler Ödül Modeli (Reward Model).
    Son anlamlı token'ın (EOS) gizli durumunu 1D skaler bir puana (r in R) dönüştürür.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        self.bloklar = nn.ModuleList([
            TransformerBlok(dim=dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

        # 1D Skaler Ödül Başlığı (Scalar Score Head)
        self.score_head = nn.Linear(dim, 1, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, S]
        Çıktı: Skaler ödül tensörü [B]
        """
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        # Nedensel Dikkat Maskesi
        mask = torch.triu(torch.full((S, S), float("-inf"), device=input_ids.device), diagonal=1)

        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        for blok in self.bloklar:
            x = blok(x, mask=mask)

        h = self.norm(x)  # [B, S, Dim]

        # Son geçerli token (Non-padding EOS token) indeksini bul
        maske_gecerli = (input_ids != self.pad_token_id).long()
        son_indeksler = maske_gecerli.sum(dim=1) - 1
        son_indeksler = son_indeksler.clamp(min=0)  # [B]

        # [B, Dim] son token durumlarını topla
        h_son = h[torch.arange(B, device=input_ids.device), son_indeksler]

        # Skaler Ödül Puanı: [B, 1] -> [B]
        odul = self.score_head(h_son).squeeze(-1)
        return odul

    def ciftli_odul_hesapla(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tercih edilen (chosen) ve reddedilen (rejected) dizilerin ödüllerini hesaplar.
        """
        r_w = self.forward(chosen_ids)
        r_l = self.forward(rejected_ids)
        return r_w, r_l
