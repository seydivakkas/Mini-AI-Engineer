"""
Sistematik Mimari Ablasyon Motoru (Day 100).
4 farklı mimari varyantı (Base, +RMSNorm, +SwiGLU, Full Modern v2) performans ve bellek açısından test eder.
"""

import time
from typing import Dict, Any, List
import numpy as np
import torch

from .konfigurasyon import ModernMiniViTConfig
from .model import ModernMiniViTForImageClassification


class AblasyonMotoru:
    """Modern mimari varyantlarını sistematik olarak kıyaslayan benchmark motoru."""

    def __init__(self, cihaz: Optional[torch.device] = None):
        self.cihaz = cihaz or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def varyantlari_olustur(self) -> Dict[str, ModernMiniViTForImageClassification]:
        """4 farklı mimari konfigürasyonunu inşa eder."""
        varyant_sozlugu = {}

        # 1. Baz Model (Baseline ViT)
        cfg_base = ModernMiniViTConfig(
            norm_turu="layernorm",
            ffn_turu="gelu",
            dikkat_turu="standard",
        )
        varyant_sozlugu["01_MiniViT_Base (LayerNorm + GELU)"] = ModernMiniViTForImageClassification(cfg_base)

        # 2. +RMSNorm
        cfg_rmsnorm = ModernMiniViTConfig(
            norm_turu="rmsnorm",
            ffn_turu="gelu",
            dikkat_turu="standard",
        )
        varyant_sozlugu["02_+RMSNorm (RMSNorm + GELU)"] = ModernMiniViTForImageClassification(cfg_rmsnorm)

        # 3. +SwiGLU
        cfg_swiglu = ModernMiniViTConfig(
            norm_turu="rmsnorm",
            ffn_turu="swiglu",
            dikkat_turu="standard",
        )
        varyant_sozlugu["03_+SwiGLU (RMSNorm + SwiGLU)"] = ModernMiniViTForImageClassification(cfg_swiglu)

        # 4. Tam Modern MiniViT-v2 (+SDPA / FlashAttention)
        cfg_modern = ModernMiniViTConfig(
            norm_turu="rmsnorm",
            ffn_turu="swiglu",
            dikkat_turu="sdpa",
        )
        varyant_sozlugu["04_Modern_MiniViT_v2 (Full Modern)"] = ModernMiniViTForImageClassification(cfg_modern)

        return varyant_sozlugu

    def varyanti_olc(
        self,
        model: ModernMiniViTForImageClassification,
        girdi: torch.Tensor,
        iterasyon: int = 50,
    ) -> Dict[str, Any]:
        """Tek bir model varyantının gecikme, bellek ve parametre profilini çıkarır."""
        model = model.to(self.cihaz).eval()
        girdi = girdi.to(self.cihaz)

        # Parametre sayısı
        param_sayisi = sum(p.numel() for p in model.parameters())

        # Isınma
        with torch.no_grad():
            for _ in range(10):
                _ = model(girdi)

        # Gecikme Ölçümü
        gecikmeler = []
        if self.cihaz.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.cihaz)

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

        # Tepe Bellek
        if self.cihaz.type == "cuda":
            tepe_bellek_mb = torch.cuda.max_memory_allocated(self.cihaz) / (1024 * 1024)
        else:
            # CPU simüle edilmiş aktivasyon belleği
            tepe_bellek_mb = float(param_sayisi * 4 / (1024 * 1024) + 1.5)

        p50 = float(np.percentile(gecikmeler, 50))
        p90 = float(np.percentile(gecikmeler, 90))
        p99 = float(np.percentile(gecikmeler, 99))
        throughput_fps = (girdi.shape[0] * 1000.0) / max(p50, 1e-3)

        return {
            "parametre_sayisi": param_sayisi,
            "p50_gecikme_ms": round(p50, 2),
            "p90_gecikme_ms": round(p90, 2),
            "p99_gecikme_ms": round(p99, 2),
            "tepe_bellek_mb": round(tepe_bellek_mb, 2),
            "throughput_fps": round(throughput_fps, 1),
            "gecikmeler": gecikmeler,
        }

    def tum_ablasyonu_calistir(
        self,
        batch_size: int = 16,
        iterasyon: int = 40,
    ) -> Dict[str, Dict[str, Any]]:
        """Tüm mimari varyantlarını sırayla test eder ve sonuçları toplar."""
        varyantlar = self.varyantlari_olustur()
        girdi = torch.randn(batch_size, 3, 32, 32)
        sonuclar = {}

        for isim, model in varyantlar.items():
            sonuc = self.varyanti_olc(model, girdi, iterasyon=iterasyon)
            sonuclar[isim] = sonuc

        return sonuclar
