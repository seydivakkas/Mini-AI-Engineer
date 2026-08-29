"""
QLoRA ve NF4 Kuantizasyon Kıyaslama Laboratuvarı (Day 107).
Full Finetuning vs FP16 LoRA vs QLoRA bellek ölçeklenmesi, kuantizasyon sadakati ve Unsloth autograd analizi.
"""

import time
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn

from .nf4_kuantizasyon import NF4Kuantizator, DoubleQuantization
from .qlora_katmani import QLoRALinear


class QLoRALaboratuvari:
    """QLoRA mimarisini ve NF4 kuantizasyonunu analiz eden benchmark motoru."""

    def __init__(self, cihaz: torch.device = torch.device("cpu")):
        self.cihaz = cihaz

    def vram_olceklenme_analizi(self) -> Dict[str, Dict[str, float]]:
        """
        7B, 13B, 33B ve 70B modeller için Full FT vs FP16 LoRA vs QLoRA (NF4+DQ) VRAM ihtiyacını (GB) hesaplar.
        """
        modeller = {
            "7B Modeli": 7.0,
            "13B Modeli": 13.0,
            "33B Modeli": 33.0,
            "70B Modeli": 70.0,
        }

        sonuclar = {}
        for isim, milyar in modeller.items():
            # Full Finetuning: Model (2B) + Gradients (2B) + AdamW States (12B) = 16 Bayt/Parametre
            full_ft_gb = milyar * 16.0

            # FP16 LoRA: Model (2B) + Adapter params (~0.2% * 16B) = ~2.05 Bayt/Parametre + Aktivasyonlar
            lora_gb = milyar * 2.05 + 2.0

            # QLoRA: NF4 (0.5B) + Double Quant (0.016B) + Adapters + Paged Optimizer = ~0.55 Bayt/Parametre + Aktivasyonlar
            qlora_gb = milyar * 0.55 + 1.2

            sonuclar[isim] = {
                "Full Fine-Tuning (GB)": round(full_ft_gb, 1),
                "FP16 LoRA (GB)": round(lora_gb, 1),
                "QLoRA (NF4 + DQ) (GB)": round(qlora_gb, 1),
                "Tasarruf Orani (%)": round(((full_ft_gb - qlora_gb) / full_ft_gb) * 100.0, 1),
            }

        return sonuclar

    def kuantizasyon_sadakati_olc(
        self,
        dim_in: int = 1024,
        dim_out: int = 1024,
        block_size: int = 64,
    ) -> Dict[str, float]:
        """
        Normal dağılımlı gerçekçi LLM ağırlıkları üzerinde NF4 kuantizasyon sadakatini (MSE, Cosine Sim) ölçer.
        """
        torch.manual_seed(42)
        # LLM ağırlıkları genellikle N(0, 0.02) civarındadır
        w_orijinal = torch.randn(dim_out, dim_in, device=self.cihaz) * 0.02

        kuantizator = NF4Kuantizator(block_size=block_size, device=self.cihaz)
        q_idx, c1, sekil = kuantizator.kuantize_et(w_orijinal)
        w_deq = kuantizator.dekuantize_et(q_idx, c1, sekil)

        # 1. Hata Metrikleri
        mse = float(torch.mean((w_orijinal - w_deq) ** 2).item())
        snr = float(10 * torch.log10(torch.mean(w_orijinal ** 2) / torch.mean((w_orijinal - w_deq) ** 2)).item())

        # 2. Cosine Benzerliği
        w_orig_norm = torch.nn.functional.normalize(w_orijinal.flatten(), dim=0)
        w_deq_norm = torch.nn.functional.normalize(w_deq.flatten(), dim=0)
        cos_sim = float(torch.dot(w_orig_norm, w_deq_norm).item())

        return {
            "mse_kaybi": round(mse, 8),
            "snr_db": round(snr, 2),
            "kosinus_benzerligi": round(cos_sim, 6),
        }

    def autograd_ve_hiz_olc(
        self,
        in_features: int = 512,
        out_features: int = 512,
        batch_size: int = 8,
        seq_len: int = 128,
        iterasyon: int = 30,
    ) -> Dict[str, Any]:
        """QLoRALinear katmanının ileri ve geri geçiş performansını test eder."""
        qlora = QLoRALinear(
            in_features=in_features,
            out_features=out_features,
            r=16,
            lora_alpha=32,
            device=self.cihaz,
        )
        # Ağırlık ilklendir ve kuantize et
        w_init = torch.randn(out_features, in_features, device=self.cihaz) * 0.02
        qlora.agirliklari_yukle_ve_kuantize_et(w_init)

        x = torch.randn(batch_size, seq_len, in_features, device=self.cihaz, requires_grad=True)

        # Isınma
        for _ in range(5):
            out = qlora(x)
            loss = out.sum()
            loss.backward()

        # Süre Ölçümü
        if self.cihaz.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(iterasyon):
            out = qlora(x)
            loss = out.sum()
            loss.backward()

        if self.cihaz.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        toplam_sure_ms = ((t1 - t0) / iterasyon) * 1000.0
        return {
            "ortalama_adim_sure_ms": round(toplam_sure_ms, 2),
            "parametre_sayisi_ana": in_features * out_features,
            "parametre_sayisi_lora": (in_features * 16) + (out_features * 16),
            "egitilebilir_parametre_orani_yuzde": round(
                ((in_features * 16 + out_features * 16) / (in_features * out_features)) * 100.0, 2
            ),
        }
