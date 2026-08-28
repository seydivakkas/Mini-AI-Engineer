"""
Model Karsilastirma ve Sayisal Esdegerlik Benchmark Motoru
==========================================================
PyTorch FP32, ONNX FP32 ve ONNX INT8 modellerini karsilastirir;
hizlanma (Speedup), gecikme (Latency), model boyutu ve sayisal esdegerlik analizlerini uretir.
"""

from typing import Dict, List, Any, Tuple
import os
import time
import numpy as np
import torch
import torch.nn as nn
from src.cikarim_motoru import ONNXInferenceEngine


class ModelBenchmarkKarsilastirici:
    """
    PyTorch ve ONNX modellerinin endustriyel karsilastirma motoru.
    """

    def __init__(
        self,
        pytorch_model: nn.Module,
        onnx_fp32_yolu: str,
        onnx_int8_yolu: str,
        is_parcacigi: int = 4
    ) -> None:
        self.pytorch_model = pytorch_model.eval()
        self.onnx_fp32_yolu = onnx_fp32_yolu
        self.onnx_int8_yolu = onnx_int8_yolu
        self.is_parcacigi = is_parcacigi

        self.motor_fp32 = ONNXInferenceEngine(onnx_fp32_yolu, is_parcacigi_sayisi=is_parcacigi)
        self.motor_int8 = ONNXInferenceEngine(onnx_int8_yolu, is_parcacigi_sayisi=is_parcacigi)

    def sayisal_esdegerlik_test_et(
        self,
        test_girdisi: np.ndarray
    ) -> Dict[str, Any]:
        """
        PyTorch, ONNX FP32 ve ONNX INT8 arasindaki sayisal farkliliklari (Cosine Sim, MAE, Max Diff) hesaplar.
        """
        tensör_girdi = torch.from_numpy(test_girdisi).float()

        with torch.no_grad():
            pytorch_cikti = self.pytorch_model(tensör_girdi).cpu().numpy()

        onnx_fp32_cikti = self.motor_fp32.tahmin_et(test_girdisi)
        onnx_int8_cikti = self.motor_int8.tahmin_et(test_girdisi)

        def _kosinus_benzerligi(a: np.ndarray, b: np.ndarray) -> float:
            a_flat = a.flatten()
            b_flat = b.flatten()
            norm_a = np.linalg.norm(a_flat)
            norm_b = np.linalg.norm(b_flat)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

        # PyTorch vs ONNX FP32
        fp32_maks_fark = float(np.max(np.abs(pytorch_cikti - onnx_fp32_cikti)))
        fp32_ort_fark = float(np.mean(np.abs(pytorch_cikti - onnx_fp32_cikti)))
        fp32_kosinus = _kosinus_benzerligi(pytorch_cikti, onnx_fp32_cikti)

        # PyTorch vs ONNX INT8
        int8_maks_fark = float(np.max(np.abs(pytorch_cikti - onnx_int8_cikti)))
        int8_ort_fark = float(np.mean(np.abs(pytorch_cikti - onnx_int8_cikti)))
        int8_kosinus = _kosinus_benzerligi(pytorch_cikti, onnx_int8_cikti)

        return {
            "fp32_maks_fark": round(fp32_maks_fark, 6),
            "fp32_ort_fark": round(fp32_ort_fark, 6),
            "fp32_kosinus_benzerligi": round(fp32_kosinus, 6),
            "int8_maks_fark": round(int8_maks_fark, 6),
            "int8_ort_fark": round(int8_ort_fark, 6),
            "int8_kosinus_benzerligi": round(int8_kosinus, 6),
            "pytorch_cikti": pytorch_cikti,
            "onnx_fp32_cikti": onnx_fp32_cikti,
            "onnx_int8_cikti": onnx_int8_cikti
        }

    def tam_benchmark_kos(
        self,
        ornek_girdi: np.ndarray,
        tekrar: int = 100
    ) -> Dict[str, Any]:
        """
        3 model varyantı uzerinde eszamanli ve adil gecikme/verimlilik testi kosar.
        """
        tensör_girdi = torch.from_numpy(ornek_girdi).float()

        # 1. PyTorch FP32 Gecikmesi
        # Isinma
        for _ in range(10):
            with torch.no_grad():
                _ = self.pytorch_model(tensör_girdi)

        pytorch_gecikmeler: List[float] = []
        for _ in range(tekrar):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.pytorch_model(tensör_girdi)
            t1 = time.perf_counter()
            pytorch_gecikmeler.append((t1 - t0) * 1000.0)

        arr_pt = np.array(pytorch_gecikmeler)
        pt_metrikler = {
            "ortalama_ms": round(float(np.mean(arr_pt)), 3),
            "p50_ms": round(float(np.percentile(arr_pt, 50)), 3),
            "p95_ms": round(float(np.percentile(arr_pt, 95)), 3),
            "p99_ms": round(float(np.percentile(arr_pt, 99)), 3),
            "fps": round(float(1000.0 / (np.mean(arr_pt) + 1e-9) * ornek_girdi.shape[0]), 2)
        }

        # 2. ONNX FP32 Gecikmesi
        ort_fp32_metrikler = self.motor_fp32.gecikme_olcumle(ornek_girdi, tekrar_sayisi=tekrar)

        # 3. ONNX INT8 Gecikmesi
        ort_int8_metrikler = self.motor_int8.gecikme_olcumle(ornek_girdi, tekrar_sayisi=tekrar)

        # Hizlanma Katsayilari (Speedup)
        hizlanma_fp32 = round(pt_metrikler["ortalama_ms"] / (ort_fp32_metrikler["ortalama_ms"] + 1e-9), 2)
        hizlanma_int8 = round(pt_metrikler["ortalama_ms"] / (ort_int8_metrikler["ortalama_ms"] + 1e-9), 2)

        # Model Boyutlari
        pt_param_boyut_mb = sum(p.numel() * p.element_size() for p in self.pytorch_model.parameters()) / (1024 * 1024)
        onnx_fp32_mb = os.path.getsize(self.onnx_fp32_yolu) / (1024 * 1024)
        onnx_int8_mb = os.path.getsize(self.onnx_int8_yolu) / (1024 * 1024)

        return {
            "pytorch_fp32": {
                "gecikme_ms": pt_metrikler["ortalama_ms"],
                "p50_ms": pt_metrikler["p50_ms"],
                "p95_ms": pt_metrikler["p95_ms"],
                "fps": pt_metrikler["fps"],
                "boyut_mb": round(pt_param_boyut_mb, 3),
                "speedup": 1.00
            },
            "onnx_fp32": {
                "gecikme_ms": ort_fp32_metrikler["ortalama_ms"],
                "p50_ms": ort_fp32_metrikler["p50_ms"],
                "p95_ms": ort_fp32_metrikler["p95_ms"],
                "fps": ort_fp32_metrikler["fps_throughput"],
                "boyut_mb": round(onnx_fp32_mb, 3),
                "speedup": hizlanma_fp32
            },
            "onnx_int8": {
                "gecikme_ms": ort_int8_metrikler["ortalama_ms"],
                "p50_ms": ort_int8_metrikler["p50_ms"],
                "p95_ms": ort_int8_metrikler["p95_ms"],
                "fps": ort_int8_metrikler["fps_throughput"],
                "boyut_mb": round(onnx_int8_mb, 3),
                "speedup": hizlanma_int8
            },
            "ornek_sayisi": ornek_girdi.shape[0],
            "tekrar": tekrar
        }
