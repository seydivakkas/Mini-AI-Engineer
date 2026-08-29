"""
LoRA Enjektörü ve Ağırlık Yönetim Modülü (Day 176 - FAZ 9).
Cross-Attention projeksiyonlarına (to_q, to_k, to_v, to_out) otomatik LoRA enjeksiyonu yapar.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn
from .lora_katmani import LoRALinear


class LoRAEnjektoru:
    """Difüzyon Modellerine LoRA Katmanları Enjekte Eden ve Yöneten Motor."""

    @classmethod
    def lora_enjekte_et(
        cls,
        model: nn.Module,
        hedef_katman_isimleri: List[str] = ["to_q", "to_k", "to_v", "to_out"],
        r: int = 8,
        lora_alpha: float = 16.0,
    ) -> int:
        """
        Modeldeki hedef katmanları tarar ve LoRALinear katmanlarıyla değiştirir.
        Döner: Değiştirilen katman sayısı.
        """
        enjekte_edilen = 0
        for name, module in model.named_modules():
            for child_name, child_module in module.named_children():
                if any(target in child_name for target in hedef_katman_isimleri) and isinstance(child_module, nn.Linear):
                    lora_layer = LoRALinear(child_module, r=r, lora_alpha=lora_alpha)
                    setattr(module, child_name, lora_layer)
                    enjekte_edilen += 1
        return enjekte_edilen

    @classmethod
    def parametre_sayilarini_getir(cls, model: nn.Module) -> Dict[str, Any]:
        """Eğitilebilir ve dondurulmuş parametre istatistiklerini hesaplar."""
        toplam_param = sum(p.numel() for p in model.parameters())
        egitilebilir_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        oran = (egitilebilir_param / toplam_param) * 100 if toplam_param > 0 else 0.0

        return {
            "toplam_parametre": toplam_param,
            "egitilebilir_parametre": egitilebilir_param,
            "egitilebilir_oran_yuzde": round(oran, 3),
            "tasarruf_orani": f"~%{100 - oran:.2f} Bellek Tasarrufu",
        }
