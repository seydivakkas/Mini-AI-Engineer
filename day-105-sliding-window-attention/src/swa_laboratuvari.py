"""
Sliding Window Attention (SWA) Kıyaslama ve Karşılaştırma Laboratuvarı (Day 105).
Full Attention vs SWA bellek tavanı, FLOPs tasarrufu ve etkin alıcı alan (receptive field) analizi.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn

from .sliding_window_attention import SlidingWindowAttention
from .rolling_buffer_cache import RollingBufferCache


class SWALaboratuvari:
    """Full Attention ile Mistral SWA mimarisini karşılaştıran benchmark motoru."""

    def __init__(
        self,
        dim: int = 512,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        window_size: int = 512,
        katman_sayisi: int = 32,
        cihaz: Optional[torch.device] = None,
    ):
        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.katman_sayisi = katman_sayisi
        self.cihaz = cihaz or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def modelleri_olustur(self) -> Dict[str, nn.Module]:
        """SWA modellerini oluşturur."""
        return {
            f"Mistral SWA (W={self.window_size})": SlidingWindowAttention(
                dim=self.dim,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                window_size=self.window_size,
            ).to(self.cihaz).eval(),
        }

    def kv_cache_bellek_karsilastirmasi(
        self,
        batch_size: int = 16,
        dizi_uzunluklari: List[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768],
    ) -> Dict[str, List[float]]:
        """
        Full Attention (Sınırsız Büyüyen) vs SWA (Sabit W Boyutlu) KV Cache belleğini (MB - FP16) hesaplar.
        """
        sonuclar = {"Full Causal Attention (O(S))": [], f"Mistral SWA (W={self.window_size}) (O(W))": []}
        head_dim = self.dim // self.num_q_heads  # 64

        for S in dizi_uzunluklari:
            # 1. Full Attention: 2 * L * B * H_kv * S * d_h * 2 bayt
            full_bayt = 2 * self.katman_sayisi * batch_size * self.num_kv_heads * S * head_dim * 2
            sonuclar["Full Causal Attention (O(S))"].append(round(full_bayt / (1024 * 1024), 2))

            # 2. SWA: 2 * L * B * H_kv * min(S, W) * d_h * 2 bayt
            swa_tokens = min(S, self.window_size)
            swa_bayt = 2 * self.katman_sayisi * batch_size * self.num_kv_heads * swa_tokens * head_dim * 2
            sonuclar[f"Mistral SWA (W={self.window_size}) (O(W))"].append(round(swa_bayt / (1024 * 1024), 2))

        return sonuclar

    def etkin_alici_alan_hesabi(self) -> Dict[str, Any]:
        """
        Transformer katmanları istiflendikçe (stacking) etkin alıcı alanın (Effective Receptive Field)
        nasıl katman bazında genişlediğini hesaplar.
        Formül: Katman_l alıcı alanı = l * W.
        """
        katman_alici_alanlari = []
        for l in range(1, self.katman_sayisi + 1):
            katman_alici_alanlari.append(l * self.window_size)

        return {
            "toplam_katman": self.katman_sayisi,
            "pencere_boyutu": self.window_size,
            "maksimum_alici_alan": self.katman_sayisi * self.window_size,
            "katman_bazli_alanlar": katman_alici_alanlari,
        }

    def gecikme_ve_throughput_olc(
        self,
        batch_size: int = 8,
        seq_len: int = 512,
        iterasyon: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """SWA modelinin çıkarım hızını ölçer."""
        model = self.modelleri_olustur()[f"Mistral SWA (W={self.window_size})"]
        girdi = torch.randn(batch_size, seq_len, self.dim, device=self.cihaz)

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
            "Mistral SWA": {
                "p50_ms": round(p50, 2),
                "p90_ms": round(p90, 2),
                "throughput_tps": round(throughput_tps, 1),
                "parametre_sayisi": param_sayisi,
            }
        }
