"""
Çift Akışlı Token Birleştirici (Dual-Stream Token Combiner) Modülü (Day 171 - FAZ 9).
Kullanıcı ses tokenlarını (RVQ) ve metin tokenlarını ortak gömme uzayında birleştirir.
"""

from typing import Tuple
import torch
import torch.nn as nn


class CiftAkisliTokenBirlestirici(nn.Module):
    """Metin ve 8 Kademeli RVQ Ses Tokenlarını Birleştiren Gömme Katmanı."""

    def __init__(
        self,
        text_vocab_size: int = 1000,
        audio_codebook_size: int = 1024,
        num_quantizers: int = 8,
        d_model: int = 256,
    ):
        super().__init__()
        self.text_vocab_size = text_vocab_size
        self.num_quantizers = num_quantizers
        self.d_model = d_model

        # 1. Metin Gömme Tablosu
        self.text_embed = nn.Embedding(text_vocab_size, d_model)

        # 2. RVQ Ses Gömme Tabloları (Her kuantalayıcı katmanı için ayrı embedding)
        self.audio_embeds = nn.ModuleList([
            nn.Embedding(audio_codebook_size, d_model)
            for _ in range(num_quantizers)
        ])

    def ses_tokenlarini_gom(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        audio_tokens: [B, Num_Q=8, T]
        Döner: [B, T, d_model] -> 8 katmanın gömmelerinin toplamı
        """
        B, Num_Q, T = audio_tokens.shape
        ses_vektoru = 0.0
        for q in range(Num_Q):
            q_token = audio_tokens[:, q, :]  # [B, T]
            ses_vektoru = ses_vektoru + self.audio_embeds[q](q_token)
        return ses_vektoru

    def metin_tokenlarini_gom(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """text_tokens: [B, S] -> [B, S, d_model]"""
        return self.text_embed(text_tokens)
