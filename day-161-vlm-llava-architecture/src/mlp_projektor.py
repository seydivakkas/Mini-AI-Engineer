"""
MLP Hizalama Projektörü (Cross-Modal Projection Layer) Modülü (Day 161 - FAZ 9).
Görsel token boyutunu (d_vision) LLM metin embedding uzayına (d_text) dönüştürür.
"""

import torch
import torch.nn as nn


class MLPProjektor(nn.Module):
    """LLaVA 2 Katmanlı GELU MLP Projektörü."""

    def __init__(self, d_vision: int = 768, d_text: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_vision, d_text),
            nn.GELU(),
            nn.Linear(d_text, d_text),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, Num_Patches, d_vision)
        Çıktı: (Batch, Num_Patches, d_text) -> LLM ile doğrudan birleşebilen görsel tokenlar
        """
        return self.mlp(visual_tokens)
