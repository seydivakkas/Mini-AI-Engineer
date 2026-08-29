"""
Pipeline Parallelism (PP) Çekirdek Motoru ve P2P İletişim Kuyruğu (Day 185 - FAZ 10).
Aşama Bazlı Katman Bölümleme, Mikro-Batch Aktivasyon Önbelleklemesi ve P2P Transferi.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class P2PIletisimKuyrugu:
    """Komşu Pipeline Aşamaları Arasında Noktadan Noktaya (P2P) Tensör Transfer Kuyruğu."""

    def __init__(self, num_stages: int = 4):
        self.num_stages = num_stages
        # forward_buffers[p]: stage p -> stage p+1 aktivasyon transferi
        self.forward_buffers: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_stages)}
        # backward_buffers[p]: stage p -> stage p-1 gradyan transferi
        self.backward_buffers: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_stages)}

    def forward_gonder(self, from_stage: int, microbatch_id: int, tensor: torch.Tensor):
        """Aşama p'den p+1'e aktivasyon tensörü gönderir."""
        if from_stage < self.num_stages - 1:
            self.forward_buffers[from_stage].append({
                "microbatch_id": microbatch_id,
                "tensor": tensor.detach().clone(),
            })

    def forward_al(self, to_stage: int, microbatch_id: int) -> Optional[torch.Tensor]:
        """Aşama p'nin p-1'den gelen aktivasyon tensörünü çeker."""
        if to_stage == 0:
            return None
        from_stage = to_stage - 1
        buffer = self.forward_buffers[from_stage]
        for i, item in enumerate(buffer):
            if item["microbatch_id"] == microbatch_id:
                return buffer.pop(i)["tensor"]
        return None

    def backward_gonder(self, from_stage: int, microbatch_id: int, grad_tensor: torch.Tensor):
        """Aşama p'den p-1'e gradyan tensörü gönderir."""
        if from_stage > 0:
            self.backward_buffers[from_stage].append({
                "microbatch_id": microbatch_id,
                "grad_tensor": grad_tensor.detach().clone(),
            })

    def backward_al(self, to_stage: int, microbatch_id: int) -> Optional[torch.Tensor]:
        """Aşama p'nin p+1'den gelen gradyan tensörünü çeker."""
        if to_stage == self.num_stages - 1:
            return None
        from_stage = to_stage + 1
        buffer = self.backward_buffers[from_stage]
        for i, item in enumerate(buffer):
            if item["microbatch_id"] == microbatch_id:
                return buffer.pop(i)["grad_tensor"]
        return None


class PipelineStage(nn.Module):
    """
    Tek Bir Pipeline Aşaması (Pipeline Stage).
    Modelin belirli bir katman aralığını barındırır.
    İleri geçişte aktivasyonları önbelleğe alır, geri geçişte gradyanları hesaplar.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.stage_id = stage_id
        self.num_stages = num_stages

        # Mikro-batch aktivasyon önbelleği: {microbatch_id: input_tensor}
        self.activation_cache: Dict[int, torch.Tensor] = {}

    def forward_step(self, microbatch_id: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """Mikro-batch ileri geçişi ve aktivasyon önbellekleme."""
        input_var = input_tensor.detach().clone().requires_grad_(True)
        self.activation_cache[microbatch_id] = input_var

        output = self.layers(input_var)
        return output

    def backward_step(self, microbatch_id: int, output_grad: torch.Tensor) -> torch.Tensor:
        """Mikro-batch geri geçişi ve giriş gradyanı hesaplama."""
        input_var = self.activation_cache.pop(microbatch_id, None)
        assert input_var is not None, f"Aşama {self.stage_id} için mikro-batch {microbatch_id} önbellekte bulunamadı."

        output = self.layers(input_var)
        torch.autograd.backward(output, output_grad)

        input_grad = input_var.grad if input_var.grad is not None else torch.zeros_like(input_var)
        return input_grad

    def get_cached_activation_count(self) -> int:
        """O an VRAM'de bekleyen aktif mikro-batch aktivasyon sayısı."""
        return len(self.activation_cache)
