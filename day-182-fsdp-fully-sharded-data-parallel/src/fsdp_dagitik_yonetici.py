"""
FSDP Dağıtık Yönetici ve Bellek Analizcisi Modülü (Day 182 - FAZ 10).
Çok Katmanlı FSDP İleri/Geri İcra Yöneticisi ve DDP vs ZeRO-1/2/3 VRAM Analitiği.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from .fsdp_sharding_motoru import FSDPKatmanSarmalayici, ShardingLevel


class FSDPModelYoneticisi:
    """
    Çok Katmanlı Ağlar İçin FSDP Dağıtık Model Yöneticisi.
    Katmanları bağımsız FSDP bloklarına böler ve N rank arasında Zero-Redundancy eğitimi yürütür.
    """

    def __init__(
        self,
        layers: List[nn.Module],
        world_size: int = 4,
        rank: int = 0,
        sharding_level: ShardingLevel = ShardingLevel.FULL_SHARD,
        lr: float = 0.01,
    ):
        self.world_size = world_size
        self.rank = rank
        self.sharding_level = sharding_level
        self.lr = lr

        # Her katmanı FSDP sarmalayıcı içine al
        self.fsdp_layers = []
        for layer in layers:
            sarmalayici = FSDPKatmanSarmalayici(
                module=layer,
                world_size=world_size,
                rank=rank,
                sharding_level=sharding_level,
            )
            self.fsdp_layers.append(sarmalayici)

        # Yerel Shard Optimizer'ı (Sadece bu rank'in 1/N parametrelerini günceller)
        local_params = [
            l.local_shard for l in self.fsdp_layers if l.local_shard is not None
        ]
        if local_params:
            self.optimizer = torch.optim.AdamW(local_params, lr=lr)
        else:
            self.optimizer = None

    def ileri_gecis(
        self,
        x: torch.Tensor,
        all_rank_shards_per_layer: Optional[List[List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Sıralı FSDP İleri Geçişi:
        Her katman sırası geldiğinde All-Gather yapar, çıktıyı hesaplar ve anında belleği boşaltır.
        """
        curr = x
        for idx, layer in enumerate(self.fsdp_layers):
            layer_shards = all_rank_shards_per_layer[idx] if all_rank_shards_per_layer else None
            curr = layer(curr, all_rank_shards=layer_shards)
        return curr

    def get_toplam_bellek_raporu(self) -> Dict[str, Any]:
        """Tüm model için FSDP bellek tasarruf özeti."""
        toplam_param = sum(l.total_param_numel for l in self.fsdp_layers)
        toplam_shard = sum(l.shard_numel for l in self.fsdp_layers)

        bytes_per_elem = 4
        tam_mb = (toplam_param * bytes_per_elem) / (1024 * 1024)
        shard_mb = (toplam_shard * bytes_per_elem) / (1024 * 1024)

        return {
            "toplam_katman_sayisi": len(self.fsdp_layers),
            "toplam_model_parametre": toplam_param,
            "rank_basina_shard_parametre": toplam_shard,
            "tam_model_vram_mb": round(tam_mb, 3),
            "fsdp_rank_vram_mb": round(shard_mb, 3),
            "vram_tasarruf_kati": f"{self.world_size:.1f}x daha az VRAM",
            "sharding_level": self.sharding_level.value,
        }


class FSDPBellekAnalizcisi:
    """DDP vs ZeRO-1 vs ZeRO-2 vs FSDP/ZeRO-3 Karşılaştırmalı Bellek Tüketim Analizcisi."""

    @classmethod
    def statik_bellek_hesapla_gb(
        cls,
        param_milyar: float,
        world_size: int = 8,
        mixed_precision: bool = True,
    ) -> Dict[str, float]:
        """
        Model parametresi (Billion) ve GPU sayısına göre yöntemlerin statik VRAM gereksinimini hesaplar (GB).
        - Parametreler: 2 bayt (FP16) veya 4 bayt (FP32)
        - Gradyanlar: 2 bayt (FP16)
        - AdamW Optimizer: 12 bayt (Master FP32 ağırlık 4B + momentum 4B + variance 4B)
        - Toplam Temel Yük: 16 bayt / parametre (FP16/FP32 karma hassasiyette)
        """
        P = param_milyar * 1e9  # Toplam parametre sayısı
        bytes_param = 2.0 if mixed_precision else 4.0
        bytes_grad = 2.0 if mixed_precision else 4.0
        bytes_opt = 12.0  # FP32 Master Weight (4B) + Momentum (4B) + Variance (4B)

        N = float(world_size)
        to_gb = 1.0 / (1024 ** 3)

        # 1. DDP (ZeRO-0): Param (P*2) + Grad (P*2) + Opt (P*12) = 16P bayt
        ddp_bytes = P * (bytes_param + bytes_grad + bytes_opt)

        # 2. ZeRO-1: Optimizer Sharding -> Param (P*2) + Grad (P*2) + Opt (P*12/N)
        zero1_bytes = P * (bytes_param + bytes_grad + (bytes_opt / N))

        # 3. ZeRO-2: Grad + Opt Sharding -> Param (P*2) + Grad (P*2/N) + Opt (P*12/N)
        zero2_bytes = P * (bytes_param + ((bytes_grad + bytes_opt) / N))

        # 4. FSDP / ZeRO-3: Param + Grad + Opt Sharding -> (P*16) / N
        fsdp_bytes = P * ((bytes_param + bytes_grad + bytes_opt) / N)

        return {
            "model_param_b": param_milyar,
            "gpu_sayisi": world_size,
            "ddp_gb": round(ddp_bytes * to_gb, 2),
            "zero1_gb": round(zero1_bytes * to_gb, 2),
            "zero2_gb": round(zero2_bytes * to_gb, 2),
            "fsdp_gb": round(fsdp_bytes * to_gb, 2),
            "fsdp_vram_tasarrufu_yuzde": round((1.0 - (fsdp_bytes / ddp_bytes)) * 100.0, 1),
        }

    @classmethod
    def buyuk_model_karsilastirma_tablosu(cls, world_size: int = 64) -> List[Dict[str, Any]]:
        """7B, 13B, 70B ve 175B modeller için 64 GPU kümesinde statik VRAM kıyaslama tablosu."""
        modeller = [
            {"ad": "Llama-2-7B", "param_b": 7.0},
            {"ad": "Llama-2-13B", "param_b": 13.0},
            {"ad": "Llama-3-70B", "param_b": 70.0},
            {"ad": "GPT-3-175B", "param_b": 175.0},
        ]

        sonuclar = []
        for m in modeller:
            res = cls.statik_bellek_hesapla_gb(m["param_b"], world_size=world_size)
            res["model_adi"] = m["ad"]
            sonuclar.append(res)
        return sonuclar
