"""
Hafif Görsel Dil Modeli (Lightweight VLM) Modülü (Day 163 - FAZ 9).
ViT Patch Kodlayıcı + MLP Projektör + Causal Decoder LLM birleşimi.
"""

import torch
import torch.nn as nn


class HafifVLM(nn.Module):
    """Görsel SFT Eğitimi için Tam Donanımlı VLM Modeli."""

    def __init__(
        self,
        d_vision: int = 768,
        d_text: int = 512,
        vocab_size: int = 1000,
        num_patches: int = 256,
        katman_sayisi: int = 3,
        kafa_sayisi: int = 8,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.d_vision = d_vision
        self.d_text = d_text
        self.vocab_size = vocab_size

        # 1. Vision Patch Encoder (Conv2d 14x14)
        self.patch_proj = nn.Conv2d(3, d_vision, kernel_size=14, stride=14)

        # 2. MLP Projector (2 Katmanlı GELU)
        self.mlp_projector = nn.Sequential(
            nn.Linear(d_vision, d_text),
            nn.GELU(),
            nn.Linear(d_text, d_text),
        )

        # 3. Metin Embedding ve Causal Decoder LLM
        self.text_embedding = nn.Embedding(vocab_size, d_text)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_text,
            nhead=kafa_sayisi,
            dim_feedforward=d_text * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=katman_sayisi)
        self.lm_head = nn.Linear(d_text, vocab_size)

    def forward(self, goruntu: torch.Tensor, metin_token_idleri: torch.Tensor) -> torch.Tensor:
        """
        goruntu: (Batch, 3, 224, 224)
        metin_token_idleri: (Batch, Text_Seq_Len)
        """
        B = goruntu.shape[0]

        # Patch Çıkarımı: (B, d_vision, 16, 16) -> (B, 256, d_vision)
        vis_patches = self.patch_proj(goruntu).flatten(2).transpose(1, 2)

        # MLP Hizalama: (B, 256, d_text)
        projected_vis = self.mlp_projector(vis_patches)

        # Metin Embedding: (B, Text_Seq_Len, d_text)
        text_emb = self.text_embedding(metin_token_idleri)

        # Füzyon: (B, 256 + Text_Seq_Len, d_text)
        fused = torch.cat([projected_vis, text_emb], dim=1)

        # Causal Mask
        seq_len = fused.shape[1]
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=fused.device), diagonal=1)

        # Decoder İleri Geçiş
        h = self.decoder(tgt=fused, memory=fused, tgt_mask=causal_mask)
        logits = self.lm_head(h)
        return logits
