"""
Analitik FLOPs, MACs, Parametre Sayısı ve Çıkarım Gecikmesi Profilleme Motoru (FLOPs Profiling Engine).
"""

from typing import Dict, Any, List, Tuple
import time
import numpy as np
import torch
import torch.nn as nn


class FLOPsProfilMotoru:
    """Derin öğrenme modelleri için analitik FLOPs, parametre sayısı ve gecikme analizörü."""

    @classmethod
    def parametre_sayisi_hesapla(cls, model: nn.Module) -> Dict[str, Any]:
        """Modelin eğitilebilir, dondurulmuş ve toplam parametrelerini ile bellek boyutunu hesaplar."""
        toplam_param = sum(p.numel() for p in model.parameters())
        egitilebilir_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        boyut_kb = float(round((toplam_param * 4) / 1024.0, 2))
        boyut_mb = float(round(boyut_kb / 1024.0, 3))

        return {
            "toplam_param": toplam_param,
            "egitilebilir_param": egitilebilir_param,
            "boyut_kb": boyut_kb,
            "boyut_mb": boyut_mb
        }

    @classmethod
    def analitik_flops_hesapla(
        cls,
        model: nn.Module,
        girdi_sekli: Tuple[int, int, int, int] = (1, 3, 64, 64)
    ) -> Dict[str, Any]:
        """Forward kancaları (hooks) ile modelin katman bazında kesin MACs ve FLOPs değerlerini hesaplar."""
        katman_kayitlari: List[Dict[str, Any]] = []
        hooks = []

        def conv_hook(mod: nn.Conv2d, inp: Any, out: torch.Tensor):
            b, c_out, h_out, w_out = out.shape
            c_in = mod.in_channels
            kh, kw = mod.kernel_size
            groups = mod.groups

            # Depthwise vs Standard vs Pointwise MACs hesabı
            macs = int(h_out * w_out * (c_in // groups) * c_out * kh * kw)
            flops = macs * 2  # 1 çarpma + 1 toplama

            katman_kayitlari.append({
                "katman_tipi": "Conv2d (Depthwise)" if groups == c_in and groups > 1 else ("Conv2d (Pointwise)" if kh == 1 and kw == 1 else "Conv2d (Standart)"),
                "in_shape": list(inp[0].shape),
                "out_shape": list(out.shape),
                "params": sum(p.numel() for p in mod.parameters()),
                "macs": macs,
                "flops": flops
            })

        def linear_hook(mod: nn.Linear, inp: Any, out: torch.Tensor):
            in_feat = mod.in_features
            out_feat = mod.out_features
            macs = int(in_feat * out_feat)
            flops = macs * 2

            katman_kayitlari.append({
                "katman_tipi": "Linear (Dense)",
                "in_shape": list(inp[0].shape),
                "out_shape": list(out.shape),
                "params": sum(p.numel() for p in mod.parameters()),
                "macs": macs,
                "flops": flops
            })

        def bn_hook(mod: nn.BatchNorm2d, inp: Any, out: torch.Tensor):
            b, c, h, w = out.shape
            flops = int(b * c * h * w * 2)
            macs = flops // 2

            katman_kayitlari.append({
                "katman_tipi": "BatchNorm2d",
                "in_shape": list(inp[0].shape),
                "out_shape": list(out.shape),
                "params": sum(p.numel() for p in mod.parameters()),
                "macs": macs,
                "flops": flops
            })

        # Kancaları kaydetme
        for _, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, nn.BatchNorm2d):
                hooks.append(module.register_forward_hook(bn_hook))

        # Sahte girdi ile ileri geçiş (Forward Pass)
        model_durumu = model.training
        model.eval()
        dummy_input = torch.randn(*girdi_sekli)

        with torch.no_grad():
            _ = model(dummy_input)

        # Kancaları temizleme
        for h in hooks:
            h.remove()
        model.train(model_durumu)

        toplam_macs = sum(k["macs"] for k in katman_kayitlari)
        toplam_flops = sum(k["flops"] for k in katman_kayitlari)
        toplam_mflops = float(round(toplam_flops / 1e6, 3))
        toplam_gflops = float(round(toplam_flops / 1e9, 5))

        return {
            "toplam_macs": toplam_macs,
            "toplam_flops": toplam_flops,
            "toplam_mflops": toplam_mflops,
            "toplam_gflops": toplam_gflops,
            "katmanlar": katman_kayitlari
        }

    @classmethod
    def gecikme_olcum(
        cls,
        model: nn.Module,
        girdi_sekli: Tuple[int, int, int, int] = (1, 3, 64, 64),
        num_runs: int = 30
    ) -> Dict[str, float]:
        """Modelin çıkarım gecikmesini (Inference Latency) istatistiksel olarak milisaniye cinsinden ölçer."""
        model.eval()
        dummy_input = torch.randn(*girdi_sekli)

        # Isınma (Warmup)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)

        gecikmeler_ms: List[float] = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t_bitis = time.perf_counter()
                gecikmeler_ms.append((t_bitis - t0) * 1000.0)

        ort_gecikme = float(round(float(np.mean(gecikmeler_ms)), 2))
        p95_gecikme = float(round(float(np.percentile(gecikmeler_ms, 95)), 2))
        fps = float(round(1000.0 / max(ort_gecikme, 1e-4), 1))

        return {
            "ort_gecikme_ms": ort_gecikme,
            "p95_gecikme_ms": p95_gecikme,
            "fps": fps
        }

    @classmethod
    def karsilastirmali_profil(
        cls,
        standart_model: nn.Module,
        tiny_model: nn.Module,
        girdi_sekli: Tuple[int, int, int, int] = (1, 3, 64, 64)
    ) -> Dict[str, Any]:
        """Standart CNN ve TinyVisionCNN modellerini kapsamlı olarak kıyaslar."""
        # 1. Parametreler
        std_p = cls.parametre_sayisi_hesapla(standart_model)
        tiny_p = cls.parametre_sayisi_hesapla(tiny_model)

        # 2. FLOPs
        std_f = cls.analitik_flops_hesapla(standart_model, girdi_sekli)
        tiny_f = cls.analitik_flops_hesapla(tiny_model, girdi_sekli)

        # 3. Gecikme
        std_l = cls.gecikme_olcum(standart_model, girdi_sekli)
        tiny_l = cls.gecikme_olcum(tiny_model, girdi_sekli)

        # Hızlanma ve Tasarruf Oranları
        param_tasarrufu_yuzde = float(round((1.0 - (tiny_p["toplam_param"] / max(std_p["toplam_param"], 1))) * 100.0, 1))
        flops_tasarrufu_yuzde = float(round((1.0 - (tiny_f["toplam_flops"] / max(std_f["toplam_flops"], 1))) * 100.0, 1))
        flops_tasarruf_carpani = float(round(std_f["toplam_flops"] / max(tiny_f["toplam_flops"], 1), 2))
        param_tasarruf_carpani = float(round(std_p["toplam_param"] / max(tiny_p["toplam_param"], 1), 2))

        return {
            "standart": {
                "params": std_p,
                "flops": std_f,
                "latency": std_l
            },
            "tinyvision": {
                "params": tiny_p,
                "flops": tiny_f,
                "latency": tiny_l
            },
            "ozet": {
                "param_tasarrufu_yuzde": param_tasarrufu_yuzde,
                "flops_tasarrufu_yuzde": flops_tasarrufu_yuzde,
                "flops_tasarruf_carpani": flops_tasarruf_carpani,
                "param_tasarruf_carpani": param_tasarruf_carpani
            }
        }
