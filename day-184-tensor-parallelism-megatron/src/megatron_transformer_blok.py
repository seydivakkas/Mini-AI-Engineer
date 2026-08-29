"""
Megatron-LM Paralel Transformer Bloğu ve Doğrulayıcı Modülü (Day 184 - FAZ 10).
Column-Parallel MLP, Paralel Self-Attention ve Katman Başına 2 All-Reduce Mimarisi.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import torch
import torch.nn as nn
from .megatron_tp_motoru import ColumnParallelLinear, RowParallelLinear


class MegatronMLP(nn.Module):
    """
    Megatron-LM MLP Bloğu (Column + Row Parallelism Fused).
    1. Sütun Paralel Genişleme: hidden_size -> 4 * hidden_size (ColumnParallelLinear)
    2. Ara İletişimsiz Yerel Aktivasyon: GeLU (İletişim GEREKTİRMEZ!)
    3. Satır Paralel Daralma: 4 * hidden_size -> hidden_size (RowParallelLinear)
    - Toplam İletişim: Tüm MLP bloğu için YALNIZCA 1 All-Reduce!
    """

    def __init__(
        self,
        hidden_size: int = 512,
        ffn_hidden_size: int = 2048,
        tp_world_size: int = 2,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank

        self.dense_h_to_4h = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=ffn_hidden_size,
            tp_world_size=tp_world_size,
            tp_rank=tp_rank,
            bias=True,
        )
        self.activation_fn = nn.GELU()
        self.dense_4h_to_h = RowParallelLinear(
            in_features=ffn_hidden_size,
            out_features=hidden_size,
            tp_world_size=tp_world_size,
            tp_rank=tp_rank,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        all_rank_row_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Sütun paralel projeksiyon (İletişimsiz)
        intermediate = self.dense_h_to_4h(x)
        # Yerel aktivasyon (İletişimsiz)
        activated = self.activation_fn(intermediate)
        # Satır paralel projeksiyon (1 All-Reduce)
        output = self.dense_4h_to_h(activated, all_rank_outputs=all_rank_row_outputs)
        return output


class MegatronSelfAttention(nn.Module):
    """
    Megatron-LM Paralel Çok Başlıklı Dikkat Mekanizması (Multi-Head Self-Attention).
    - Başlıklar (Heads) K GPU arasında eşit bölünür: num_heads / K başlık / GPU
    - Q, K, V Projeksiyonları: ColumnParallelLinear (İletişimsiz)
    - Yerel Scaled Dot-Product Attention: İletişimsiz!
    - Çıktı Projeksiyonu (Out Projection): RowParallelLinear (Yalnızca 1 All-Reduce!)
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        tp_world_size: int = 2,
        tp_rank: int = 0,
    ):
        super().__init__()
        assert num_heads % tp_world_size == 0, f"num_heads ({num_heads}) tp_world_size'a ({tp_world_size}) tam bölünmelidir."
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank

        self.num_heads_per_partition = num_heads // tp_world_size
        self.head_dim = hidden_size // num_heads

        # Q, K, V projeksiyonları tek bir ColumnParallelLinear içinde birleştirilir
        self.query_key_value = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=3 * hidden_size,
            tp_world_size=tp_world_size,
            tp_rank=tp_rank,
            bias=True,
        )

        # Çıktı Projeksiyonu (RowParallelLinear)
        self.dense = RowParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            tp_world_size=tp_world_size,
            tp_rank=tp_rank,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_dense_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # 1. QKV Projeksiyonu [B, S, 3 * (H/K) * head_dim] (İletişimsiz)
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.reshape(B, S, self.num_heads_per_partition, 3 * self.head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # [B, num_heads_per_partition, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. Yerel Scaled Dot-Product Attention (İletişimsiz)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_probs, v)

        # [B, S, (H/K) * head_dim]
        context = context.transpose(1, 2).contiguous().reshape(B, S, self.num_heads_per_partition * self.head_dim)

        # 3. Çıktı Projeksiyonu (1 All-Reduce)
        output = self.dense(context, all_rank_outputs=all_rank_dense_outputs)
        return output


class MegatronTransformerKatmani(nn.Module):
    """
    Megatron-LM Tam Transformer Katmanı.
    - Pre-LN Attention + Residual (1 All-Reduce)
    - Pre-LN MLP + Residual (1 All-Reduce)
    - Katman Başına TOPLAM 2 All-Reduce!
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        ffn_hidden_size: int = 2048,
        tp_world_size: int = 2,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = MegatronSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            tp_world_size=tp_world_size,
            tp_rank=tp_rank,
        )
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = MegatronMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            tp_world_size=tp_world_size,
            tp_rank=tp_rank,
        )

    def forward(
        self,
        x: torch.Tensor,
        all_rank_attn_outputs: Optional[List[torch.Tensor]] = None,
        all_rank_mlp_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # 1. Attention Alt-Katmanı (1 All-Reduce)
        norm_x = self.input_layernorm(x)
        attn_out = self.attention(norm_x, all_rank_dense_outputs=all_rank_attn_outputs)
        h = x + attn_out

        # 2. MLP Alt-Katmanı (1 All-Reduce)
        norm_h = self.post_attention_layernorm(h)
        mlp_out = self.mlp(norm_h, all_rank_row_outputs=all_rank_mlp_outputs)
        out = h + mlp_out

        return out


class TPDogrulayici:
    """Tek GPU ile Megatron Tensor Parallelism Matematiksel Eşdeğerlik Doğrulayıcısı."""

    @classmethod
    def mlp_esdegerlik_dogrula(
        cls,
        hidden_size: int = 256,
        ffn_hidden_size: int = 1024,
        tp_world_size: int = 2,
    ) -> Dict[str, Any]:
        """Standart tek GPU MLP ile Megatron TP MLP'nin çıktısını kıyaslar."""
        torch.manual_seed(42)
        # Standart MLP
        w1_full = torch.randn(ffn_hidden_size, hidden_size)
        b1_full = torch.randn(ffn_hidden_size)
        w2_full = torch.randn(hidden_size, ffn_hidden_size)
        b2_full = torch.randn(hidden_size)

        x = torch.randn(4, hidden_size)

        # Standart İleri Geçiş
        h_std = nn.functional.linear(x, w1_full, b1_full)
        h_act = nn.functional.gelu(h_std)
        out_std = nn.functional.linear(h_act, w2_full, b2_full)

        # Megatron TP Simülasyonu
        split_ffn = ffn_hidden_size // tp_world_size
        tp_rank_outputs = []

        for r in range(tp_world_size):
            # Rank r'nin dilimleri
            w1_r = w1_full[r * split_ffn:(r + 1) * split_ffn, :]
            b1_r = b1_full[r * split_ffn:(r + 1) * split_ffn]
            w2_r = w2_full[:, r * split_ffn:(r + 1) * split_ffn]

            # 1. Sütun paralel
            h_r = nn.functional.linear(x, w1_r, b1_r)
            h_r_act = nn.functional.gelu(h_r)
            # 2. Satır paralel kısmi çarpım
            out_r_partial = nn.functional.linear(h_r_act, w2_r, None)
            tp_rank_outputs.append(out_r_partial)

        # All-Reduce (Toplama) + Bias
        out_tp = torch.stack(tp_rank_outputs).sum(dim=0) + b2_full

        fark = (out_std - out_tp).abs().max().item()
        eslesiyor = fark < 1e-3

        return {
            "tp_world_size": tp_world_size,
            "maksimum_mutlak_hata": fark,
            "matematiksel_olarak_eslesiyor": eslesiyor,
            "katman_basi_all_reduce_sayisi": 1,
        }
