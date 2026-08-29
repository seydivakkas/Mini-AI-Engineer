"""
DeepSpeed ZeRO-Offload ve CPU AdamW Optimizer Motoru (Day 183 - FAZ 10).
Rajbhandari et al. (2021) ZeRO-Offload: Host CPU RAM & NVMe Üzerinde Optimizer Durumu Boşaltma.
"""

from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import math
import torch
import torch.nn as nn


class OffloadDevice(Enum):
    """Bellek Boşaltma Hedef Cihazları."""
    GPU = "GPU"
    CPU = "CPU"
    NVME = "NVME"


class ZeROOffloadYapilandirma:
    """ZeRO-Offload ve ZeRO-Infinity Konfigürasyon Sınıfı."""

    def __init__(
        self,
        stage: int = 3,
        offload_optimizer_device: OffloadDevice = OffloadDevice.CPU,
        offload_param_device: OffloadDevice = OffloadDevice.CPU,
        pin_memory: bool = True,
        buffer_count: int = 2,
    ):
        self.stage = stage
        self.offload_optimizer_device = offload_optimizer_device
        self.offload_param_device = offload_param_device
        self.pin_memory = pin_memory
        self.buffer_count = buffer_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zero_stage": self.stage,
            "offload_optimizer": self.offload_optimizer_device.value,
            "offload_param": self.offload_param_device.value,
            "pin_memory": self.pin_memory,
            "double_buffering": self.buffer_count >= 2,
        }


class CPUAdamWOptimizer:
    """
    Host CPU Üzerinde Çalışan ZeRO-Offload AdamW Optimizer Motoru.
    - Master Weights (FP32, 4B), Momentum (FP32, 4B) ve Variance (FP32, 4B) tamponlarını Host CPU RAM'inde saklar.
    - GPU VRAM'inden parametre başına 12 baytlık devasa AdamW yükünü tamamen boşaltır (%75 statik VRAM tasarrufu).
    - İleri-geri geçiş GPU'da tamamlandığında gradyanları PCIe üzerinden CPU'ya çeker, CPU'da günceller ve yeni ağırlığı aktarır.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # CPU RAM'de tutulan FP32 Master Durumları
        self.master_weights: List[torch.Tensor] = []
        self.exp_avg: List[torch.Tensor] = []      # Momentum (m)
        self.exp_avg_sq: List[torch.Tensor] = []   # Variance (v)

        for p in self.params:
            # CPU üzerinde 32-bit master ağırlık kopyası
            master_w = p.data.detach().clone().to(device="cpu", dtype=torch.float32)
            self.master_weights.append(master_w)
            self.exp_avg.append(torch.zeros_like(master_w))
            self.exp_avg_sq.append(torch.zeros_like(master_w))

    def step(self):
        """
        1. GPU'daki gradyanları CPU'ya taşı (Device-to-Host).
        2. CPU üzerinde FP32 AdamW güncellemesini hesapla.
        3. Güncellenmiş ağırlıkları hedef tensöre geri yaz (Host-to-Device).
        """
        self.step_count += 1
        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_count)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # 1. Gradyanı CPU'ya aktar (Device-to-Host PCIe transferi)
            grad_cpu = p.grad.data.to(device="cpu", dtype=torch.float32)

            master_w = self.master_weights[i]
            m = self.exp_avg[i]
            v = self.exp_avg_sq[i]

            # 2. Weight Decay (Decoupled AdamW)
            if self.weight_decay != 0.0:
                master_w.mul_(1.0 - self.lr * self.weight_decay)

            # 3. Momentum ve Variance Güncellemesi
            m.mul_(self.beta1).add_(grad_cpu, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad_cpu, grad_cpu, value=1.0 - self.beta2)

            # 4. Bias Düzeltmesi ve Parametre Güncellemesi
            step_size = self.lr / bias_correction1
            denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            master_w.addcdiv_(m, denom, value=-step_size)

            # 5. Güncellenmiş ağırlığı orijinal tensöre geri aktar (Host-to-Device)
            p.data.copy_(master_w.to(device=p.device, dtype=p.dtype))

    def zero_grad(self):
        """Tüm parametrelerin gradyanlarını sıfırlar."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def get_offload_memory_stats(self) -> Dict[str, Any]:
        """CPU RAM ve GPU VRAM tasarruf istatistikleri."""
        toplam_param = sum(p.numel() for p in self.params)
        gpu_tasarruf_mb = (toplam_param * 12) / (1024 * 1024)  # 12 bayt/param
        cpu_ram_mb = (toplam_param * 12) / (1024 * 1024)

        return {
            "toplam_parametre_sayisi": toplam_param,
            "gpu_vram_tasarrufu_mb": round(gpu_tasarruf_mb, 3),
            "cpu_ram_tuketimi_mb": round(cpu_ram_mb, 3),
            "vram_optimizer_azalmasi": "%100.0 (Optimizer GPU'dan tamamen silindi)",
        }
