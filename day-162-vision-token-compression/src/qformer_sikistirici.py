"""
BLIP-2 Q-Former (Querying Transformer) Token Sıkıştırma Modülü (Day 162 - FAZ 9).
Öğrenilebilir N_query sorgu vektörü ve Çapraz Dikkat (Cross-Attention) ile 256 tokenı 32 tokena sıkıştırır.
"""

import torch
import torch.nn as nn


class QFormerSikistirici(nn.Module):
    """BLIP-2 Tarzı Q-Former Token Sıkıştırma Katmanı."""

    def __init__(
        self,
        num_query_tokens: int = 32,
        d_vision: int = 768,
        d_model: int = 512,
        kafa_sayisi: int = 8,
        katman_sayisi: int = 2,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.d_model = d_model

        # Öğrenilebilir Query Token'ları (32 adet x 512d)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, d_model) * 0.02)

        # Görsel tokenları d_model boyutuna projeksiyon
        self.visual_proj = nn.Linear(d_vision, d_model)

        # Çapraz Dikkat (Cross-Attention) ve Kendi Kendine Dikkat (Self-Attention) Blokları
        self.katmanlar = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(embed_dim=d_model, num_heads=kafa_sayisi, batch_first=True),
                "cross_attn": nn.MultiheadAttention(embed_dim=d_model, num_heads=kafa_sayisi, batch_first=True),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model),
                "ln3": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                )
            })
            for _ in range(katman_sayisi)
        ])

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Girdi: visual_tokens -> (Batch, 256, 768)
        Çıktı: compressed_tokens -> (Batch, 32, 512)
        """
        B = visual_tokens.shape[0]

        # Görsel tokenları d_model uzayına yansıt (Bellek / Key & Value)
        kv_visual = self.visual_proj(visual_tokens)  # (B, 256, 512)

        # Query tokenlarını batch boyutuna çoğalt
        queries = self.query_tokens.expand(B, -1, -1)  # (B, 32, 512)

        # Q-Former Blokları İleri Geçiş
        for katman in self.katmanlar:
            # 1. Self-Attention (Query'ler kendi arasında konuşur)
            sa_out, _ = katman["self_attn"](queries, queries, queries)
            queries = katman["ln1"](queries + sa_out)

            # 2. Cross-Attention (Query'ler görsel tokenlardan bilgi çeker)
            ca_out, _ = katman["cross_attn"](query=queries, key=kv_visual, value=kv_visual)
            queries = katman["ln2"](queries + ca_out)

            # 3. Feed-Forward
            mlp_out = katman["mlp"](queries)
            queries = katman["ln3"](queries + mlp_out)

        return queries  # (B, 32, 512) -> %87.5 Sıkıştırma!
