"""
SimPO (Simple Preference Optimization) Motoru (Day 217 - FAZ 11).
Referanssız, Uzunluk-Normalize Edilmiş ve Marjin Destekli Tercih Optimizasyonu.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimPORewardCalculator:
    """
    SimPO Örtük Ödül Hesaplayıcısı:
    r_SimPO(x, y) = (β / |y|) * log π_θ(y | x)
    """

    @classmethod
    def odul_hesapla(
        cls,
        logp_y: torch.Tensor,
        uzunluk_y: int,
        beta: float = 2.0,
    ) -> torch.Tensor:
        """Uzunluk normalize edilmiş log-olasılık ödülünü hesaplar."""
        return (beta / max(1, uzunluk_y)) * logp_y


class SimPOLossObjective:
    """
    SimPO Kayıp Fonksiyonu (NeurIPS 2024):
    L_SimPO(θ) = -log σ( (β/|y_w|) log π(y_w|x) - (β/|y_l|) log π(y_l|x) - γ )
    """

    @classmethod
    def kayip_hesapla(
        cls,
        logp_w: torch.Tensor,
        logp_l: torch.Tensor,
        len_w: int,
        len_l: int,
        beta: float = 2.0,
        gamma_margin: float = 0.8,
    ) -> Tuple[torch.Tensor, float, float]:
        """Referanssız ve hedef marjinli SimPO kaybını hesaplar."""
        r_w = SimPORewardCalculator.odul_hesapla(logp_w, len_w, beta)
        r_l = SimPORewardCalculator.odul_hesapla(logp_l, len_l, beta)

        delta_r = r_w - r_l
        logit = delta_r - gamma_margin
        kayip = -F.logsigmoid(logit)

        return kayip, float(delta_r.item()), float(kayip.item())


class SimPOMemoryProfiler:
    """SimPO vs DPO VRAM ve Bellek Tasarrufu Profilleyicisi."""

    @classmethod
    def vram_tasarrufu_hesapla(cls, model_parametre_milyar: float = 7.0) -> Dict[str, float]:
        """Model boyutuna göre DPO ve SimPO VRAM gereksinimlerini hesaplar (LoRA/Post-Training)."""
        # 16-bit Ağırlık (14 GB for 7B)
        model_agirlik_gb = model_parametre_milyar * 2.0
        lora_ve_optimizer_gb = 4.4
        aktivasyon_kv_gb = 14.0

        # DPO: Policy Model + Frozen Reference Model + LoRA/Optimizer + KV
        dpo_vram = model_agirlik_gb + model_agirlik_gb + lora_ve_optimizer_gb + aktivasyon_kv_gb
        # SimPO: Yalnızca Policy Model (Sıfır Reference Model) + LoRA/Optimizer + KV
        simpo_vram = model_agirlik_gb + lora_ve_optimizer_gb + aktivasyon_kv_gb

        tasarruf_gb = dpo_vram - simpo_vram
        tasarruf_yuzde = (tasarruf_gb / dpo_vram) * 100.0

        return {
            "dpo_vram_gb": float(dpo_vram),
            "simpo_vram_gb": float(simpo_vram),
            "tasarruf_gb": float(tasarruf_gb),
            "tasarruf_yuzde": float(tasarruf_yuzde),
        }


class SimPOTrainer:
    """Referanssız SimPO Tercih Eğiticisi."""

    @classmethod
    def egitim_adimi(
        cls,
        prompt: str,
        chosen: str,
        rejected: str,
        beta: float = 2.0,
        gamma_margin: float = 0.8,
    ) -> Dict[str, Any]:
        """Tek bir SimPO optimizasyon adımı yürütür."""
        len_w = len(chosen.split())
        len_l = len(rejected.split())

        # Simüle tensör log-olasılıkları
        logp_w = torch.tensor(-15.2, requires_grad=True)
        logp_l = torch.tensor(-28.6, requires_grad=True)

        kayip_tensor, delta_r, kayip_val = SimPOLossObjective.kayip_hesapla(
            logp_w=logp_w,
            logp_l=logp_l,
            len_w=len_w,
            len_l=len_l,
            beta=beta,
            gamma_margin=gamma_margin,
        )

        kayip_tensor.backward()

        return {
            "prompt": prompt,
            "len_chosen": len_w,
            "len_rejected": len_l,
            "kayip": kayip_val,
            "ortuk_marjin": delta_r,
            "hedef_marjin": gamma_margin,
            "grad_w": float(logp_w.grad.item()) if logp_w.grad is not None else 0.0,
        }
