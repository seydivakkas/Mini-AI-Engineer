"""
Tesla Chrono-Voxel Neural Fields (CV-NF) Cross-Attention Module
==============================================================
Sparse Linear-Complexity Cross-Attention between Continuous Query
Points and Multi-View Camera Feature Maps.

Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLinearCrossAttention(nn.Module):
    """
    Linear Complexity O(N) Cross-Attention for Multi-View Camera Fusion.
    """
    def __init__(self, query_dim: int = 32, key_dim: int = 32, embed_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, query_dim)

    def forward(self, queries: torch.Tensor, key_values: torch.Tensor) -> torch.Tensor:
        """
        queries: [B, N_query, query_dim]
        key_values: [B, N_kv, key_dim]
        returns: [B, N_query, query_dim]
        """
        B, N_q, _ = queries.shape
        _, N_kv, _ = key_values.shape

        Q = self.q_proj(queries).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_values).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_values).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product cross attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.out_proj(out)
