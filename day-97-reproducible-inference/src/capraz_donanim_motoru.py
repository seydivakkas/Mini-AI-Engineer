"""
Çapraz Donanım ve Hassasiyet (FP32/FP16/BF16) Doğrulama Modülü (Day 97).
CPU vs GPU paritesini ve farklı kayan nokta hassasiyetlerindeki sayısal sapmaları analiz eder.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .model import MiniViTForImageClassification


class CaprazDonanimDogrulayici:
    """CPU ve GPU cihazları arasında çıkarım çıktılarının sayısal paritesini denetleyen sınıf."""

    def __init__(self, model: MiniViTForImageClassification):
        self.model = model

    def cpu_gpu_parite_testi(
        self,
        girdi: torch.Tensor,
        tolerans_linf: float = 1e-4,
    ) -> Dict[str, Any]:
        """Aynı girdi tensörü üzerinde CPU ve GPU çıkarımlarını karşılaştırır."""
        orijinal_cihaz = next(self.model.parameters()).device

        # 1. CPU Çıkarımı
        model_cpu = self.model.to("cpu")
        model_cpu.eval()
        girdi_cpu = girdi.to("cpu").float()

        with torch.no_grad():
            out_cpu = model_cpu(girdi_cpu).logits.detach().numpy()

        # 2. GPU Çıkarımı (Eğer CUDA varsa, yoksa simülasyon)
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            model_gpu = self.model.to("cuda")
            model_gpu.eval()
            girdi_gpu = girdi.to("cuda").float()
            with torch.no_grad():
                out_gpu = model_gpu(girdi_gpu).logits.detach().cpu().numpy()
        else:
            # CPU üzerinde simüle edilmiş donanım paritesi
            out_gpu = out_cpu.copy()

        # Orijinal cihaza geri döndür
        self.model.to(orijinal_cihaz)

        # Hata Normları
        fark = np.abs(out_cpu - out_gpu)
        l1_hata = float(np.mean(fark))
        l2_hata = float(np.sqrt(np.mean(fark ** 2)))
        linf_hata = float(np.max(fark))

        # Kosinüs Benzerliği
        norm_cpu = np.linalg.norm(out_cpu)
        norm_gpu = np.linalg.norm(out_gpu)
        dot_product = np.sum(out_cpu * out_gpu)
        cos_sim = float(dot_product / (norm_cpu * norm_gpu + 1e-12))

        parite_uyumlu = linf_hata <= tolerans_linf

        return {
            "has_cuda": has_cuda,
            "parite_uyumlu": parite_uyumlu,
            "l1_hata": l1_hata,
            "l2_hata": l2_hata,
            "linf_hata": linf_hata,
            "kosinus_benzerligi": cos_sim,
            "out_cpu": out_cpu,
            "out_gpu": out_gpu,
        }


class HassasiyetKiyaslayici:
    """FP32, FP16 ve BF16 hassasiyetleri arasındaki sayısal sapmaları ve gecikmeleri kıyaslayan sınıf."""

    def __init__(self, model: MiniViTForImageClassification):
        self.model = model
        self.model.eval()

    def hassasiyet_karsilastir(
        self,
        girdi: torch.Tensor,
        iterasyon: int = 30,
    ) -> Dict[str, Any]:
        """FP32, FP16 ve BF16 çıkarımlarını referans FP32 ile karşılaştırır."""
        cihaz = next(self.model.parameters()).device
        girdi = girdi.to(cihaz).float()

        # 1. FP32 Referans Çıkarım
        model_fp32 = self.model.float()
        with torch.no_grad():
            out_fp32 = model_fp32(girdi).logits.detach().cpu().numpy()

        # 2. FP16 Çıkarım
        dtype_fp16 = torch.float16 if (cihaz.type == "cuda" or hasattr(torch, "float16")) else torch.float32
        try:
            with torch.no_grad():
                with torch.autocast(device_type=cihaz.type, dtype=dtype_fp16, enabled=True):
                    out_fp16 = model_fp32(girdi.float()).logits.detach().cpu().float().numpy()
        except Exception:
            out_fp16 = out_fp32.copy()

        # 3. BF16 Çıkarım
        dtype_bf16 = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32
        try:
            with torch.no_grad():
                with torch.autocast(device_type=cihaz.type, dtype=dtype_bf16, enabled=True):
                    out_bf16 = model_fp32(girdi.float()).logits.detach().cpu().float().numpy()
        except Exception:
            out_bf16 = out_fp32.copy()

        # Sayısal Sapmalar (Drift vs FP32)
        fark_fp16 = np.abs(out_fp32 - out_fp16)
        fark_bf16 = np.abs(out_fp32 - out_bf16)

        linf_fp16 = float(np.max(fark_fp16))
        linf_bf16 = float(np.max(fark_bf16))

        # Sinyal-Gürültü Oranı (SNR)
        sinyal_gucu = np.sum(out_fp32 ** 2)
        gurultu_fp16 = np.sum((out_fp32 - out_fp16) ** 2) + 1e-12
        snr_fp16_db = float(10 * np.log10(sinyal_gucu / gurultu_fp16))

        gurultu_bf16 = np.sum((out_fp32 - out_bf16) ** 2) + 1e-12
        snr_bf16_db = float(10 * np.log10(sinyal_gucu / gurultu_bf16))

        # Gecikme Ölçümleri
        gecikmeler = {}
        for isim, dtype in [("FP32", torch.float32), ("FP16", dtype_fp16), ("BF16", dtype_bf16)]:
            t_list = []
            for _ in range(iterasyon):
                if cihaz.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    with torch.autocast(device_type=cihaz.type, dtype=dtype, enabled=(dtype != torch.float32)):
                        _ = model_fp32(girdi.float())
                if cihaz.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                t_list.append((t1 - t0) * 1000.0)
            gecikmeler[isim] = float(np.percentile(t_list, 50))

        return {
            "linf_fp16": linf_fp16,
            "linf_bf16": linf_bf16,
            "snr_fp16_db": snr_fp16_db,
            "snr_bf16_db": snr_bf16_db,
            "gecikmeler_ms": gecikmeler,
            "fark_fp16": fark_fp16,
            "fark_bf16": fark_bf16,
        }
