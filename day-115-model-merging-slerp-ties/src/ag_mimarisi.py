"""
Model Birleştirme Testleri için Uzman Ağ Mimarisi (Day 115).
Matematik, Kodlama ve Mantıksal Akıl Yürütme alanlarında uzmanlaşabilen MLP/Transformer mimarisi.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class UzmanModel(nn.Module):
    """Farklı alanlarda uzmanlaştırılabilen çok katmanlı sinir ağı."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(x))
        h = self.norm(h)
        h = F.silu(self.fc2(h)) + h  # Residual bağlantı
        return self.fc3(h)
