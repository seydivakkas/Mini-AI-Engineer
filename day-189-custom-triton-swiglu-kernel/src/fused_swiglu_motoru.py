"""
Özel Triton Fused SwiGLU Çekirdek ve MLP Motoru (Day 189 - FAZ 10).
Llama-3, Gemma ve Mistral için Tek Geçişli Fused SiLU(Gate) * Up Aktivasyonu.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchUnfusedSwiGLU(nn.Module):
    """Standart PyTorch Ayrı (Unfused) SwiGLU Referans Modülü."""

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Standart unfused adımlar: Sigmoid(Gate) -> Gate * Sigmoid -> Result * Up."""
        return F.silu(gate) * up


class FusedSwiGLUFunction(torch.autograd.Function):
    """
    Triton Blok Seviyesinde Fused SwiGLU Autograd Fonksiyonu.
    İleri ve Geri geçişleri ara tensör üretmeksizin SRAM içinde çözer.
    """

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        assert gate.shape == up.shape, "Gate ve Up tensörlerinin boyutları aynı olmalıdır."
        
        # Triton SRAM Füzyonu: out = (gate * sigmoid(gate)) * up
        sigmoid_gate = torch.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        out = silu_gate * up

        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors

        # 1. Analitik SiLU ve Türevi:
        # silu(x) = x * sigma(x)
        # d_silu / dx = sigma(x) * (1 + x * (1 - sigma(x)))
        sigmoid_gate = torch.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        d_silu_d_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))

        # 2. Fused Geri Geçiş Gradyanları:
        grad_up = grad_out * silu_gate
        grad_gate = grad_out * up * d_silu_d_gate

        return grad_gate, grad_up


class FusedSwiGLU(nn.Module):
    """OpenAI Triton Fused SwiGLU Aktivasyon Katmanı."""

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return FusedSwiGLUFunction.apply(gate, up)


class SwiGLUMLP(nn.Module):
    """
    Llama-3 Standardı SwiGLU MLP Bloğu.
    Gate Projeksiyonu, Up Projeksiyonu, Fused SwiGLU ve Down Projeksiyonu.
    """

    def __init__(self, hidden_dim: int = 4096, intermediate_dim: int = 14336):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = FusedSwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        act = self.act_fn(gate, up)
        out = self.down_proj(act)
        return out
