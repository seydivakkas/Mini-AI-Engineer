"""
DeepSeek MLA vs GQA vs MHA Kıyaslama ve Karşılaştırma Laboratuvarı (Day 103).
KV Cache ayak izi, bellek sıkıştırma oranları ve çıkarım gecikmesini analiz eder.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn

from .mla_katmani import MultiHeadLatentAttention
from .latent_kv_cache import LatentKVCache


class MLALaboratuvari:
    """MLA, GQA ve MHA mimarilerini bellek ve çıkarım performansı açısından karşılaştırır."""

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 16,
        head_dim: int = 32,
        kv_latent_dim: int = 128,
        q_latent_dim: int = 256,
        rope_dim: int = 32,
        katman_sayisi: int = 32,
        cihaz: Optional[torch.device] = None,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.q_latent_dim = q_latent_dim
        self.rope_dim = rope_dim
        self.katman_sayisi = katman_sayisi
        self.cihaz = cihaz or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def modelleri_olustur(self) -> Dict[str, nn.Module]:
        """Karşılaştırma modellerini oluşturur."""
        return {
            "DeepSeek MLA (d_c=128, d_R=32)": MultiHeadLatentAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                kv_latent_dim=self.kv_latent_dim,
                q_latent_dim=self.q_latent_dim,
                rope_dim=self.rope_dim,
            ).to(self.cihaz).eval(),
        }

    def kv_cache_bellek_karsilastirmasi(
        self,
        batch_size: int = 16,
        dizi_uzunluklari: List[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768],
    ) -> Dict[str, List[float]]:
        """
        MHA, GQA ve MLA için 32 katmanlı modelde KV Cache bellek tüketimini (MB - FP16) hesaplar.
        """
        sonuclar = {"MHA (16 KV Kafa)": [], "GQA (4 KV Kafa)": [], "DeepSeek MLA": []}

        for S in dizi_uzunluklari:
            # 1. MHA: 2 * L * B * H_q * S * d_h * 2 bayt
            mha_bayt = 2 * self.katman_sayisi * batch_size * self.num_heads * S * self.head_dim * 2
            sonuclar["MHA (16 KV Kafa)"].append(round(mha_bayt / (1024 * 1024), 2))

            # 2. GQA-4: 2 * L * B * (H_q / 4) * S * d_h * 2 bayt
            gqa_kv_heads = max(1, self.num_heads // 4)
            gqa_bayt = 2 * self.katman_sayisi * batch_size * gqa_kv_heads * S * self.head_dim * 2
            sonuclar["GQA (4 KV Kafa)"].append(round(gqa_bayt / (1024 * 1024), 2))

            # 3. DeepSeek MLA: L * B * S * (d_c + d_R) * 2 bayt
            mla_bayt = self.katman_sayisi * batch_size * S * (self.kv_latent_dim + self.rope_dim) * 2
            sonuclar["DeepSeek MLA"].append(round(mla_bayt / (1024 * 1024), 2))

        return sonuclar

    def gecikme_ve_throughput_olc(
        self,
        batch_size: int = 8,
        seq_len: int = 512,
        iterasyon: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """MLA modelinin çıkarım hızını ölçer."""
        model = self.modelleri_olustur()["DeepSeek MLA (d_c=128, d_R=32)"]
        girdi = torch.randn(batch_size, seq_len, self.dim, device=self.cihaz)

        # Isınma
        with torch.no_grad():
            for _ in range(5):
                _ = model(girdi)

        gecikmeler = []
        for _ in range(iterasyon):
            if self.cihaz.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(girdi)
            if self.cihaz.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            gecikmeler.append((t1 - t0) * 1000.0)

        p50 = float(np.percentile(gecikmeler, 50))
        p90 = float(np.percentile(gecikmeler, 90))
        throughput_tps = (batch_size * seq_len * 1000.0) / max(p50, 1e-3)
        param_sayisi = sum(p.numel() for p in model.parameters())

        return {
            "DeepSeek MLA": {
                "p50_ms": round(p50, 2),
                "p90_ms": round(p90, 2),
                "throughput_tps": round(throughput_tps, 1),
                "parametre_sayisi": param_sayisi,
            }
        }
