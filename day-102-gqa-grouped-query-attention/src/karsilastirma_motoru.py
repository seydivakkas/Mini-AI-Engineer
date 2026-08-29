"""
GQA vs MHA vs MQA Kıyaslama ve Karşılaştırma Motoru (Day 102).
Farklı bağlam uzunlukları ve batch boyutlarında KV Cache bellek tüketimi ve çıkarım gecikmesini analiz eder.
"""

import time
from typing import Dict, Any, List
import numpy as np
import torch

from .dikkat_mimarileri import GroupedQueryAttention, AttentionTuru
from .kv_cache import KVCache


class GQALaboratuvari:
    """MHA, MQA ve GQA mimarilerini sistematik olarak karşılaştıran benchmark motoru."""

    def __init__(
        self,
        dim: int = 512,
        num_q_heads: int = 32,
        katman_sayisi: int = 32,
        cihaz: Optional[torch.device] = None,
    ):
        self.dim = dim
        self.num_q_heads = num_q_heads
        self.katman_sayisi = katman_sayisi
        self.cihaz = cihaz or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def modelleri_olustur(self) -> Dict[str, GroupedQueryAttention]:
        """MHA, MQA ve GQA modellerini dinamik başlık oranlarıyla oluşturur."""
        gqa_kv = max(1, self.num_q_heads // 4)
        return {
            f"MHA ({self.num_q_heads} Q, {self.num_q_heads} KV)": GroupedQueryAttention(
                dim=self.dim, num_q_heads=self.num_q_heads, num_kv_heads=self.num_q_heads
            ).to(self.cihaz).eval(),
            f"GQA ({self.num_q_heads} Q, {gqa_kv} KV)": GroupedQueryAttention(
                dim=self.dim, num_q_heads=self.num_q_heads, num_kv_heads=gqa_kv
            ).to(self.cihaz).eval(),
            f"MQA ({self.num_q_heads} Q, 1 KV)": GroupedQueryAttention(
                dim=self.dim, num_q_heads=self.num_q_heads, num_kv_heads=1
            ).to(self.cihaz).eval(),
        }

    def kv_cache_bellek_analizi(
        self,
        batch_size: int = 16,
        dizi_uzunluklari: List[int] = [512, 1024, 2048, 4096, 8192],
    ) -> Dict[str, List[float]]:
        """Farklı dizi uzunluklarında teorik 32-katman KV Cache bellek tüketimini (MB) hesaplar."""
        sonuclar = {"MHA": [], "GQA": [], "MQA": []}
        head_dim = self.dim // self.num_q_heads  # 16

        for S in dizi_uzunluklari:
            # 2 * Layers * B * H_kv * S * HeadDim * 2 Bayt (FP16) / 1024^2
            bayt_basi = 2 * self.katman_sayisi * batch_size * S * head_dim * 2 / (1024 * 1024)
            sonuclar["MHA"].append(round(bayt_basi * 32, 2))
            sonuclar["GQA"].append(round(bayt_basi * 8, 2))
            sonuclar["MQA"].append(round(bayt_basi * 1, 2))

        return sonuclar

    def gecikme_ve_throughput_olc(
        self,
        batch_size: int = 8,
        seq_len: int = 512,
        iterasyon: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """Her mimari varyantı için ileri geçiş süresi ve token işleme kapasitesini ölçer."""
        modeller = self.modelleri_olustur()
        girdi = torch.randn(batch_size, seq_len, self.dim, device=self.cihaz)
        rapor = {}

        for isim, model in modeller.items():
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

            # Parametre Sayısı
            param_sayisi = sum(p.numel() for p in model.parameters())

            rapor[isim] = {
                "p50_ms": round(p50, 2),
                "p90_ms": round(p90, 2),
                "throughput_tps": round(throughput_tps, 1),
                "parametre_sayisi": param_sayisi,
                "mimari_turu": model.mimari_turu.value,
            }

        return rapor
