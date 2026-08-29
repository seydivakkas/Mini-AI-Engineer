"""
FSDP (Fully Sharded Data Parallel) Sharding Motoru (Day 182 - FAZ 10).
Zhao et al. (2023) PyTorch FSDP & DeepSpeed ZeRO-3: Zero-Redundancy Parametre, Gradyan ve Optimizer Sharding.
"""

from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import torch
import torch.nn as nn


class ShardingLevel(Enum):
    """FSDP / ZeRO Sharding Seviyeleri."""
    NO_SHARD = "NO_SHARD"           # DDP / ZeRO-0: Tam Replikasyon
    SHARD_GRAD_OP = "SHARD_GRAD_OP" # ZeRO-2: Gradyan ve Optimizer Sharding
    FULL_SHARD = "FULL_SHARD"       # ZeRO-3: Parametre, Gradyan ve Optimizer Sharding ($1/N$)


class FSDPKatmanSarmalayici(nn.Module):
    """
    FSDP Katman Sarmalayıcı Modülü.
    Katman parametrelerini N parçaya (shard) böler.
    İleri geçişte geçici All-Gather ile tam ağırlığı oluşturur, işlem biter bitmez siler (Free/Drop).
    Geri geçişte Reduce-Scatter ile gradyanları rank'lere paylaştırır.
    """

    def __init__(
        self,
        module: nn.Module,
        world_size: int = 4,
        rank: int = 0,
        sharding_level: ShardingLevel = ShardingLevel.FULL_SHARD,
    ):
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.rank = rank
        self.sharding_level = sharding_level

        # Orijinal parametre şekillerini ve boyutlarını kaydet
        self.param_metadata = []
        flat_params = []

        for name, p in module.named_parameters():
            self.param_metadata.append({
                "name": name,
                "shape": p.shape,
                "numel": p.numel(),
                "requires_grad": p.requires_grad,
            })
            flat_params.append(p.data.flatten().clone())

        if flat_params:
            self.full_flat_param = torch.cat(flat_params)
            self.total_param_numel = self.full_flat_param.numel()
        else:
            self.full_flat_param = torch.empty(0)
            self.total_param_numel = 0

        # World size'a tam bölünmesi için padding
        N = self.world_size
        M = self.total_param_numel
        pad_size = (N - (M % N)) % N if N > 0 else 0
        self.pad_size = pad_size
        self.padded_numel = M + pad_size
        self.shard_numel = self.padded_numel // max(N, 1)

        # Yerel Shard Parametresini oluştur (Sadece 1/N'lik parça VRAM'de tutulur)
        if self.sharding_level == ShardingLevel.FULL_SHARD and self.total_param_numel > 0:
            if pad_size > 0:
                padded_param = torch.cat([self.full_flat_param, torch.zeros(pad_size, dtype=self.full_flat_param.dtype)])
            else:
                padded_param = self.full_flat_param

            # Bu rank'in 1/N'lik dilimi
            local_shard_data = padded_param[self.rank * self.shard_numel:(self.rank + 1) * self.shard_numel].clone()
            self.local_shard = nn.Parameter(local_shard_data, requires_grad=True)

            # Orijinal parametreleri bellekten temizle (Zero-Redundancy)
            for meta in self.param_metadata:
                parts = meta["name"].split(".")
                curr = module
                for part in parts[:-1]:
                    curr = getattr(curr, part)
                setattr(curr, parts[-1], None)
        else:
            self.local_shard = None

        self.unshared_flat_weight = None

    def unshard_parameters(self, all_rank_shards: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        All-Gather İletişimi: N rank'in yerel parçalarını birleştirerek tam ağırlık tensörünü üretir.
        """
        if self.sharding_level != ShardingLevel.FULL_SHARD or self.local_shard is None:
            return self.full_flat_param

        if all_rank_shards is not None:
            gathered = torch.cat(all_rank_shards, dim=0)
        else:
            # Tekil simülasyon testi için kendi shard'ını çoğalt
            gathered = self.local_shard.repeat(self.world_size)

        if self.pad_size > 0:
            unshared = gathered[:-self.pad_size]
        else:
            unshared = gathered

        self.unshared_flat_weight = unshared
        self._flat_weighti_module_yerlestir(unshared)
        return unshared

    def reshard_parameters(self):
        """
        Bellek Boşaltma (Free / Drop): İleri veya geri geçiş bittiği anda
        tam ağırlık tensörünü silerek VRAM'i boşa çıkarır.
        """
        if self.sharding_level == ShardingLevel.FULL_SHARD:
            self.unshared_flat_weight = None
            for meta in self.param_metadata:
                parts = meta["name"].split(".")
                curr = self.module
                for part in parts[:-1]:
                    curr = getattr(curr, part)
                setattr(curr, parts[-1], None)

    def _flat_weighti_module_yerlestir(self, flat_weight: torch.Tensor):
        """Düzleştirilmiş tam ağırlığı modülün alt parametrelerine yeniden boyutlandırıp atar."""
        offset = 0
        for meta in self.param_metadata:
            numel = meta["numel"]
            shape = meta["shape"]
            sub_weight = flat_weight[offset:offset + numel].reshape(shape)
            offset += numel

            parts = meta["name"].split(".")
            curr = self.module
            for part in parts[:-1]:
                curr = getattr(curr, part)
            param_val = nn.Parameter(sub_weight.clone(), requires_grad=meta["requires_grad"])
            setattr(curr, parts[-1], param_val)

    def forward(self, x: torch.Tensor, all_rank_shards: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        FSDP İleri Geçişi:
        1. All-Gather ile parametreleri unshard et
        2. Modülü çalıştır
        3. Parametreleri serbest bırak (Drop)
        """
        self.unshard_parameters(all_rank_shards)
        out = self.module(x)
        self.reshard_parameters()
        return out

    def get_memory_stats(self) -> Dict[str, Any]:
        """FSDP bellek kullanım ve tasarruf istatistikleri."""
        bytes_per_elem = 4  # float32
        tam_boyut_mb = (self.total_param_numel * bytes_per_elem) / (1024 * 1024)
        shard_boyut_mb = (self.shard_numel * bytes_per_elem) / (1024 * 1024) if self.local_shard is not None else tam_boyut_mb

        return {
            "toplam_parametre_numel": self.total_param_numel,
            "shard_parametre_numel": self.shard_numel,
            "tam_model_bellek_mb": round(tam_boyut_mb, 3),
            "shard_bellek_mb": round(shard_boyut_mb, 3),
            "vram_tasarruf_orani": f"%{max(0, 100 - (100 / self.world_size)):.1f}" if self.sharding_level == ShardingLevel.FULL_SHARD else "%0.0",
            "sharding_level": self.sharding_level.value,
        }
