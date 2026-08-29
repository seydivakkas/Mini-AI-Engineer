"""
PPO Actor (Politika) ve Critic (Değer Ağı) Modelleri (Day 109).
Otoregresif üretim, token bazlı log-olasılık hesabı ve durum değeri V(s) kestirimi.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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


class ActorPolicy(nn.Module):
    """
    PPO Aktör (Politika - LLM) Modeli.
    Metin üretir ve üretilen her token'ın log-olasılığını hesaplar.
    """

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
        return self.lm_head(x)  # [B, S, vocab_size]

    def uret_ve_logprob_al(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Otoregresif token üretir ve yanıt token'larının log-olasılıklarını döndürür.
        Çıktı: (tam_dizi [B, S_prompt + T], yanit_ids [B, T], logprobs [B, T])
        """
        B = prompt_ids.shape[0]
        curr_ids = prompt_ids.clone()
        yanit_tokens = []
        yanit_logprobs = []

        for _ in range(max_new_tokens):
            logits = self.forward(curr_ids)[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            next_token = dist.sample()  # [B]
            log_prob = dist.log_prob(next_token)  # [B]

            yanit_tokens.append(next_token.unsqueeze(1))
            yanit_logprobs.append(log_prob.unsqueeze(1))

            curr_ids = torch.cat([curr_ids, next_token.unsqueeze(1)], dim=1)

        yanit_ids = torch.cat(yanit_tokens, dim=1)        # [B, T]
        logprobs = torch.cat(yanit_logprobs, dim=1)        # [B, T]
        return curr_ids, yanit_ids, logprobs

    def logprob_degerlendir(
        self,
        tam_dizi: torch.Tensor,
        yanit_baslangic_idx: int,
    ) -> torch.Tensor:
        """Mevcut tam dizi içindeki yanıt token'larının log-olasılıklarını hesaplar."""
        logits = self.forward(tam_dizi)  # [B, S, V]
        # Shift logits to align with next tokens
        shift_logits = logits[:, yanit_baslangic_idx - 1 : -1, :]  # [B, T, V]
        target_tokens = tam_dizi[:, yanit_baslangic_idx:]           # [B, T]

        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        logprobs = log_probs_all.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
        return logprobs


class CriticValueNetwork(nn.Module):
    """
    PPO Eleştirmen (Critic / Değer Ağı) Modeli.
    Her token adımı için beklenen indirgenmiş kümülatif ödülü V(s_t) tahmin eder.
    """

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
        # 1D Durum Değeri Çıktı Başlığı
        self.value_head = nn.Linear(dim, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, yanit_baslangic_idx: int) -> torch.Tensor:
        """
        input_ids: [B, S]
        Çıktı: Yanıt adımları için değer tahminleri V(s_t) -> [B, T]
        """
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        mask = torch.triu(torch.full((S, S), float("-inf"), device=input_ids.device), diagonal=1)

        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        for blok in self.bloklar:
            x = blok(x, mask=mask)
        h = self.norm(x)  # [B, S, Dim]

        values_all = self.value_head(h).squeeze(-1)  # [B, S]
        # Yalnızca yanıt adımlarındaki durum değerlerini al
        return values_all[:, yanit_baslangic_idx:]   # [B, T]
