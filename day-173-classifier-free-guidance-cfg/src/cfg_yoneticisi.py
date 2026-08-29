"""
Classifier-Free Guidance (CFG) Yöneticisi Modülü (Day 173 - FAZ 9).
Koşullu (Conditional) ve Koşulsuz (Unconditional) gürültü kestirimlerini w ölçeği ile birleştirir.
"""

from typing import Tuple
import torch
import torch.nn as nn


class CFGYoneticisi:
    """Classifier-Free Guidance ve Dinamik Eşikleme (Dynamic Thresholding) Motoru."""

    def __init__(self, varsayilan_guidance_scale: float = 7.5):
        self.varsayilan_guidance_scale = varsayilan_guidance_scale

    def yonlendirilmis_gurultu_hesapla(
        self,
        eps_uncond: torch.Tensor,
        eps_cond: torch.Tensor,
        guidance_scale: float = None,
        dinamik_esikleme: bool = False,
        esik_yuzdeligi: float = 0.995,
    ) -> torch.Tensor:
        """
        CFG Formülasyonu (Ho & Salimans, 2022):
        eps_tilde = eps_uncond + w * (eps_cond - eps_uncond)
        w=1.0: Saf koşullu model
        w>1.0: Metin istemi yönünde agresif ekstrapolasyon
        """
        w = guidance_scale if guidance_scale is not None else self.varsayilan_guidance_scale
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)

        if dinamik_esikleme:
            eps_guided = self._dinamik_esik_uygula(eps_guided, percentile=esik_yuzdeligi)

        return eps_guided

    def _dinamik_esik_uygula(self, tensor: torch.Tensor, percentile: float = 0.995) -> torch.Tensor:
        """Aşırı CFG değerlerinde oluşan renk patlamasını (Oversaturation) önler."""
        B = tensor.shape[0]
        tensor_flat = tensor.abs().view(B, -1)
        k = int((1.0 - percentile) * tensor_flat.shape[1])
        k = max(1, k)
        val, _ = torch.kthvalue(tensor_flat, tensor_flat.shape[1] - k, dim=1, keepdim=True)
        s = torch.clamp(val, min=1.0).view(B, 1, 1, 1)
        return torch.clamp(tensor, -s, s) / s
