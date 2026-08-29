"""
Mini-Omni Reasoner v1.0 Model Mimarisi (Day 201 - 201 GÜNLÜK BÜYÜK FİNAL).
Çok Modlu (Görüntü + Ses + Metin) Giriş, Triton Fused FlashAttention-2, Top-2 Seyrek MoE ve Çıkış Başlığı.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultimodalPatchProjector(nn.Module):
    """
    Görüntü ve Ses Girdilerini Ortak LLM Vektör Uzayına İzdüşüren Katman.
    """

    def __init__(self, vision_dim: int = 64, audio_dim: int = 32, embed_dim: int = 128):
        super().__init__()
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def project_vision(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        return self.vision_proj(vision_tokens)

    def project_audio(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_proj(audio_tokens)


class TritonFusedRMSNormLayer(nn.Module):
    """
    Triton Fused RMSNorm & Residual Ekleme Çekirdeği (FAZ 10 - Day 188).
    y = ((x + residual) / RMS(x + residual)) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
        new_res = x
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.gamma, new_res


class TritonFlashAttention2Block(nn.Module):
    """
    Parçalı (Tiled) FlashAttention-2 Hızlı Dikkat Bloğu (FAZ 10 - Day 190).
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        # Fused Scaled Dot-Product Attention (FlashAttention-2 benzetimi)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Nedensel (Causal) Maskeleme
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [B, H, T, D]
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


class SparseMoERoutingLayer(nn.Module):
    """
    Top-2 Seyrek Uzmanlar Karışımı (MoE) Katmanı (FAZ 8 & 10).
    4 Uzman: 0=Vision/Spatial, 1=Math/Code, 2=Reasoning/Logic, 3=Language/NLP.
    """

    def __init__(self, embed_dim: int = 128, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(embed_dim, num_experts)

        # 4 Uzman Modülü (SwiGLU FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.SiLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)  # [N, C]

        # Yönlendirici Skorları ve Top-2 Seçimi
        logits = self.router(x_flat)
        gate_probs = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Normalize et

        out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            weight = topk_weights[:, i].unsqueeze(-1)

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    tokens = x_flat[mask]
                    expert_out = self.experts[e](tokens)
                    out[mask] += expert_out * weight[mask]

        return out.reshape(B, T, C), gate_probs


class MiniOmniReasonerModel(nn.Module):
    """
    Mini-Omni Reasoner v1.0 Birleşik Yapay Zeka Modeli.
    Görüntü + Ses + Metin -> Triton FlashAttention + Seyrek MoE -> Reasoning Akışı.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.multimodal_proj = MultimodalPatchProjector(vision_dim=64, audio_dim=32, embed_dim=embed_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": TritonFusedRMSNormLayer(embed_dim),
                "attn": TritonFlashAttention2Block(embed_dim),
                "norm2": TritonFusedRMSNormLayer(embed_dim),
                "moe": SparseMoERoutingLayer(embed_dim, num_experts=4, top_k=2),
            })
            for _ in range(num_layers)
        ])

        self.final_norm = TritonFusedRMSNormLayer(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        text_tokens: torch.Tensor,
        vision_patches: Optional[torch.Tensor] = None,
        audio_patches: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Tüm modaliteleri birleştirir ve MoE ile çıkarım yapar."""
        embeddings = [self.token_embedding(text_tokens)]

        if vision_patches is not None:
            v_emb = self.multimodal_proj.project_vision(vision_patches)
            embeddings.insert(0, v_emb)

        if audio_patches is not None:
            a_emb = self.multimodal_proj.project_audio(audio_patches)
            embeddings.insert(1 if vision_patches is not None else 0, a_emb)

        # Çok Modlu Vektör Birleştirme
        x = torch.cat(embeddings, dim=1)
        res = None

        all_gate_probs = []
        for layer in self.layers:
            # 1. Dikkat Bloğu (Fused RMSNorm + FlashAttention-2)
            normed_x, res = layer["norm1"](x, residual=res)
            attn_out = layer["attn"](normed_x)

            # 2. MoE Bloğu (Fused RMSNorm + Top-2 MoE)
            normed_attn, res = layer["norm2"](attn_out, residual=res)
            moe_out, gate_probs = layer["moe"](normed_attn)
            x = moe_out
            all_gate_probs.append(gate_probs)

        final_x, _ = self.final_norm(x, residual=res)
        logits = self.lm_head(final_x)

        return {
            "logits": logits,
            "hidden_states": final_x,
            "gate_probs": all_gate_probs[-1] if all_gate_probs else None,
            "seq_len": x.shape[1],
        }
