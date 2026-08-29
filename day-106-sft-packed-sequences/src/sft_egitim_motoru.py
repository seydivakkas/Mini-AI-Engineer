"""
SFT Eğitim Motoru ve Blok-Diyagonal Transformer Modülü (Day 106).
Prompt Maskeleme (-100), SFT Cross-Entropy Kaybı ve Blok-Diyagonal Dikkatli İleri Geçiş.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_paketleyici import PaketlenmisDizi, olustur_blok_diyagonal_maske


class SFTTransformerBlok(nn.Module):
    """Blok-diyagonal maske destekli Transformer Bloğu."""

    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm1(x)

        qkv = self.qkv_proj(h).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)

        skor = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            skor = skor + attn_mask

        attn = F.softmax(skor, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)

        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class SFTEgitimMotoru(nn.Module):
    """
    Instruction Supervised Fine-Tuning (SFT) Modeli.
    Prompt maskeleme (ignore_index=-100) ile sadece asistan yanıtlarında gradyan hesaplar.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        self.bloklar = nn.ModuleList([
            SFTTransformerBlok(dim=dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Ağırlık bağlama (Weight Tying)
        self.lm_head.weight = self.tok_embed.weight

        # SFT Kayıp Fonksiyonu (-100 etiketlerini yok sayar)
        self.kayip_fonksiyonu = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input_ids: [B, S]
        labels:    [B, S] (Prompt token'ları -100)
        """
        B, S = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        x = self.tok_embed(input_ids) + self.pos_embed(position_ids)

        for blok in self.bloklar:
            x = blok(x, attn_mask=attn_mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, S, vocab_size]

        kayip = None
        if labels is not None:
            # Shifted Cross Entropy: logits[:, :-1] -> labels[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            kayip = self.kayip_fonksiyonu(shift_logits, shift_labels)

        return logits, kayip

    def egitim_adimi_paketlenmis(self, paket: PaketlenmisDizi) -> float:
        """Tek bir paketlenmiş dizi üzerinde eğitim adımı (Forward + Backward) koşturur."""
        self.train()
        device = next(self.parameters()).device

        inp = paket.input_ids.unsqueeze(0).to(device)
        lbl = paket.labels.unsqueeze(0).to(device)
        pos = paket.position_ids.unsqueeze(0).to(device)

        maske = olustur_blok_diyagonal_maske(
            paket.ornek_uzunluklari,
            toplam_uzunluk=inp.shape[1],
            device=device,
        )

        _, loss = self(input_ids=inp, labels=lbl, position_ids=pos, attn_mask=maske)
        return float(loss.item()) if loss is not None else 0.0
