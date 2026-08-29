"""
LLaVA Uçtan Uca Görsel Dil Modeli (End-to-End VLM) Modülü (Day 161 - FAZ 9).
ViT Encoder + MLP Projector + Oto-Regresif LLM birleşimini yürütür.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .vit_goruntu_kodlayici import ViTGoruntuKodlayici
from .mlp_projektor import MLPProjektor


class BasitLLMKodlayici(nn.Module):
    """VLM için hafif oto-regresif Decoder LLM simülatörü."""

    def __init__(self, vocab_size: int = 1000, d_text: int = 512, katman_sayisi: int = 4, kafa_sayisi: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_text = d_text
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

    def forward(self, fused_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, Seq_Len, d_text) -> [Visual Tokens + Text Tokens]
        Çıktı: Logits (Batch, Seq_Len, vocab_size)
        """
        # Causal mask oluştur
        seq_len = fused_embeddings.shape[1]
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=fused_embeddings.device), diagonal=1)

        # Memory yerine dummy context ile oto-regresif çözümleme
        h = self.decoder(tgt=fused_embeddings, memory=fused_embeddings, tgt_mask=causal_mask)
        logits = self.lm_head(h)
        return logits


class LLaVAVLMModeli(nn.Module):
    """LLaVA: Large Language and Vision Assistant Mimari Modeli."""

    def __init__(
        self,
        goruntu_boyutu: int = 224,
        patch_boyutu: int = 14,
        d_vision: int = 768,
        d_text: int = 512,
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.vision_encoder = ViTGoruntuKodlayici(
            goruntu_boyutu=goruntu_boyutu,
            patch_boyutu=patch_boyutu,
            d_vision=d_vision,
        )
        self.mlp_projector = MLPProjektor(d_vision=d_vision, d_text=d_text)
        self.llm = BasitLLMKodlayici(vocab_size=vocab_size, d_text=d_text)

    def forward(self, goruntu: torch.Tensor, metin_token_idleri: torch.Tensor) -> torch.Tensor:
        """
        goruntu: (Batch, 3, 224, 224)
        metin_token_idleri: (Batch, Text_Seq_Len)
        """
        # 1. ViT ile Görsel Patch Tokenları Çıkar: (B, 256, d_vision)
        visual_tokens = self.vision_encoder(goruntu)

        # 2. MLP Projektör ile LLM Uzayına Dönüştür: (B, 256, d_text)
        projected_visual_tokens = self.mlp_projector(visual_tokens)

        # 3. Metin Tokenlarını Embedding Yap: (B, Text_Seq_Len, d_text)
        text_embeddings = self.llm.text_embedding(metin_token_idleri)

        # 4. Multimodal Füzyon: [Visual Tokens + Text Tokens] -> (B, 256 + Text_Seq_Len, d_text)
        fused_embeddings = torch.cat([projected_visual_tokens, text_embeddings], dim=1)

        # 5. LLM İleri Geçiş ve Logit Üretimi
        logits = self.llm(fused_embeddings)
        return logits
