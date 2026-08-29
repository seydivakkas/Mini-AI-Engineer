"""
Öğretmen (Teacher) ve Öğrenci (Student) Dil Modelleri Mimarisi (Day 119).
Geniş parametreli Öğretmen modelden hafif, 10x daha küçük Öğrenci modele mimari yapı.
"""

from typing import Tuple
import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    """Transformer tabanlı dil modeli bloğu."""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        pos = torch.arange(0, S, device=x.device).unsqueeze(0).expand(B, S)
        emb = self.token_embed(x) + self.pos_embed(pos)

        # Causal Attention Mask
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        h = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.lm_head(h)
        return logits

    def toplam_parametre(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ogretmen_model_uret(vocab_size: int = 1000) -> TransformerLM:
    """Büyük Öğretmen Model (Teacher): d_model=256, 4 katman."""
    return TransformerLM(vocab_size=vocab_size, d_model=256, n_heads=8, num_layers=4)


def ogrenci_model_uret(vocab_size: int = 1000) -> TransformerLM:
    """Kompakt Öğrenci Model (Student): d_model=64, 2 katman (10x daha hafif)."""
    return TransformerLM(vocab_size=vocab_size, d_model=64, n_heads=2, num_layers=2)
