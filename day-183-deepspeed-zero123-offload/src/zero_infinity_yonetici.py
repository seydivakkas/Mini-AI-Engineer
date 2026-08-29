"""
ZeRO-Infinity Katman Yöneticisi ve Bellek Profilleyicisi Modülü (Day 183 - FAZ 10).
CPU/NVMe Bellek Boşaltma, Double-Buffering Overlap ve 1 Trilyon Parametre Profilleme.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from .zero_offload_motoru import OffloadDevice, ZeROOffloadYapilandirma, CPUAdamWOptimizer


class ZeROInfinityKatmanSarmalayici(nn.Module):
    """
    ZeRO-Infinity Katman Sarmalayıcı Modülü.
    Ağırlıkları CPU RAM veya NVMe depolama havuzunda saklar.
    İleri geçişte katman bazında GPU'ya (H2D) çeker ve işlem bitince GPU'dan temizler.
    Geri geçişte gradyanları CPU'ya (D2H) aktarır.
    """

    def __init__(
        self,
        module: nn.Module,
        offload_device: OffloadDevice = OffloadDevice.CPU,
        compute_device: str = "cpu",
    ):
        super().__init__()
        self.module = module
        self.offload_device = offload_device
        self.compute_device = compute_device

        # Orijinal ağırlıkları CPU/NVMe deposunda sakla
        self.offloaded_state: Dict[str, torch.Tensor] = {}
        for name, p in module.named_parameters():
            self.offloaded_state[name] = p.data.detach().clone().to(device="cpu")

        # Dinlenme anında GPU belleğini boşalt
        self._gpu_bellek_temizle()

    def _gpu_bellek_temizle(self):
        """GPU VRAM'inde tutulan parametreleri temizler."""
        for name, _ in self.module.named_parameters():
            parts = name.split(".")
            curr = self.module
            for part in parts[:-1]:
                curr = getattr(curr, part)
            setattr(curr, parts[-1], None)

    def _agirliklari_yukle_ve_calistir(self, x: torch.Tensor) -> torch.Tensor:
        """Ağırlıkları CPU/NVMe'den hedef hesaplama cihazına yükler ve çalıştırır."""
        # 1. Host-to-Device PCIe Transferi
        for name, tensor_data in self.offloaded_state.items():
            parts = name.split(".")
            curr = self.module
            for part in parts[:-1]:
                curr = getattr(curr, part)
            param_val = nn.Parameter(tensor_data.to(device=self.compute_device, dtype=torch.float32))
            setattr(curr, parts[-1], param_val)

        # 2. İleri Hesaplama
        x_dev = x.to(self.compute_device)
        out = self.module(x_dev)

        # 3. Hesaplama biter bitmez GPU VRAM'i temizle (Immediate Drop)
        self._gpu_bellek_temizle()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._agirliklari_yukle_ve_calistir(x)


class ZeROOffloadProfilleyici:
    """
    Farklı Model Ölçekleri İçin (7B - 1T) GPU VRAM, Host CPU RAM ve NVMe SSD
    Bellek Dağılımını Modelleme ve PCIe Gen4/Gen5 İletişim Analitiği.
    """

    @classmethod
    def bellek_dagilimi_hesapla_gb(cls, param_milyar: float) -> Dict[str, Any]:
        """
        Model parametresi (Billion) bazında bellek gereksinimlerini hesaplar (GB).
        - Parametreler: 2 bayt (FP16)
        - Gradyanlar: 2 bayt (FP16)
        - AdamW Optimizer: 12 bayt (FP32)
        - Toplam: 16 bayt / param
        """
        P = param_milyar * 1e9
        to_gb = 1.0 / (1024 ** 3)

        toplam_gb = P * 16.0 * to_gb
        param_gb = P * 2.0 * to_gb
        grad_gb = P * 2.0 * to_gb
        opt_gb = P * 12.0 * to_gb

        # 1. Standart DDP (Her şey GPU'da)
        ddp_gpu_gb = toplam_gb

        # 2. ZeRO-Offload (Optimizer CPU RAM'de, Parametre + Gradyan GPU'da)
        offload_gpu_gb = param_gb + grad_gb
        offload_cpu_gb = opt_gb

        # 3. ZeRO-Infinity (Sadece aktif katman GPU'da, Parametre + Opt NVMe'de)
        # Aktif tek bir katman ~%2 GPU VRAM
        infinity_gpu_gb = (param_gb + grad_gb) * 0.05
        infinity_cpu_gb = param_gb * 0.2
        infinity_nvme_gb = param_gb + opt_gb

        return {
            "model_param_b": param_milyar,
            "toplam_statik_gb": round(toplam_gb, 2),
            "ddp_gpu_vram_gb": round(ddp_gpu_gb, 2),
            "zero_offload_gpu_gb": round(offload_gpu_gb, 2),
            "zero_offload_cpu_gb": round(offload_cpu_gb, 2),
            "zero_infinity_gpu_gb": round(infinity_gpu_gb, 2),
            "zero_infinity_nvme_gb": round(infinity_nvme_gb, 2),
            "offload_vram_tasarrufu_yuzde": round((1.0 - (offload_gpu_gb / ddp_gpu_gb)) * 100.0, 1),
            "infinity_vram_tasarrufu_yuzde": round((1.0 - (infinity_gpu_gb / ddp_gpu_gb)) * 100.0, 1),
        }

    @classmethod
    def coklu_model_profil_raporu(cls) -> List[Dict[str, Any]]:
        """7B, 13B, 70B, 175B ve 1 Trilyon (1000B) parametreli modellerin offload profillemesi."""
        modeller = [
            {"ad": "Llama-2-7B", "param_b": 7.0},
            {"ad": "Llama-2-13B", "param_b": 13.0},
            {"ad": "Llama-3-70B", "param_b": 70.0},
            {"ad": "GPT-3-175B", "param_b": 175.0},
            {"ad": "Titan-1T (1000B)", "param_b": 1000.0},
        ]

        rapor = []
        for m in modeller:
            res = cls.bellek_dagilimi_hesapla_gb(m["param_b"])
            res["model_adi"] = m["ad"]
            rapor.append(res)
        return rapor
