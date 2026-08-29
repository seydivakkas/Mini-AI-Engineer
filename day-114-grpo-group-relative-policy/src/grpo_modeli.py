"""
GRPO Dil Modeli Mimarisi ve Grup Örnekleme Motoru (Day 114).
Causal Transformer mimarisi ve tek bir prompt için G adet paralel çıktı (group rollout) üretebilen örnekleyici.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grpo_kaybi import hesapla_token_bazli_logprob


class TransformerBlok(nn.Module):
    """Causal Maskeli Transformer Bloğu."""

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


class GRPODilModeli(nn.Module):
    """GRPO Eğitimi için Causal Dil Modeli."""

    def __init__(
        self,
        vocab_size: int = 1000,
        dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.bloklar = nn.ModuleList([
            TransformerBlok(dim=dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        mask = torch.triu(torch.full((S, S), float("-inf"), device=input_ids.device), diagonal=1)

        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        for blok in self.bloklar:
            x = blok(x, mask=mask)
        x = self.norm(x)
        return self.lm_head(x)

    def token_logprob_hesapla(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Verilen dizi için her token adımındaki log-olasılıkları [B, S] döndürür."""
        logits = self.forward(input_ids)
        return hesapla_token_bazli_logprob(logits, input_ids)

    @torch.no_grad()
    def grup_ornekle(
        self,
        prompt_ids: torch.Tensor,
        group_size: int = 8,
        max_new_tokens: int = 12,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Tek bir prompt [1, S_p] için G adet paralel yanıt üretir.
        Döndürür: [G, S_p + max_new_tokens]
        """
        S_p = prompt_ids.shape[1]
        # Prompt'u G adet kopyala: [G, S_p]
        mevcut_diziler = prompt_ids.expand(group_size, S_p).clone()

        for _ in range(max_new_tokens):
            logits = self.forward(mevcut_diziler)[:, -1, :] / max(1e-4, temperature)
            probs = F.softmax(logits, dim=-1)
            sonraki_tokenlar = torch.multinomial(probs, num_samples=1)  # [G, 1]
            mevcut_diziler = torch.cat([mevcut_diziler, sonraki_tokenlar], dim=-1)

        return mevcut_diziler
