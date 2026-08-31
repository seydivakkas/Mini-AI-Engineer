"""
Tesla Kuantizasyon Profilleyici Modülü
======================================
Bu modül; FP32 tensörlerin INT8 kuantizasyon hızını, bellek sıkıştırma oranını,
SQNR doğruluğunu ve Fused NPU çekirdek hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_fsd_npu_int8_kuantizasyon import TeslaFSDNPUQuantizer


class TeslaKuantizasyonProfilleyici:
    """
    NPU INT8 Kuantizasyon Performans Profilleyicisi.
    """
    def __init__(self, weight_elements: int = 50000, iterations: int = 100):
        self.weight_elements = weight_elements
        self.iterations = iterations

    def benchmark_quantization(self) -> Dict[str, Any]:
        quantizer = TeslaFSDNPUQuantizer()

        # Rastgele FSD HydraNet Ağırlık Tensörü (FP32)
        np.random.seed(42)
        weights_fp32 = np.random.normal(0, 0.5, self.weight_elements).astype(np.float32)

        gecikmeler_us: List[float] = []

        q_weights = None
        scale = 1.0
        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            q_weights, scale = quantizer.quantize_symmetric_int8(weights_fp32)
            deq_weights = quantizer.dequantize_int8(q_weights, scale)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        deq_final = quantizer.dequantize_int8(q_weights, scale)
        err_metrics = quantizer.compute_sqnr_and_error(weights_fp32, deq_final)

        # Bellek Karşılaştırması
        mem_fp32_kb = (self.weight_elements * 4) / 1024.0
        mem_int8_kb = (self.weight_elements * 1) / 1024.0
        tasarruf_yuzde = ((mem_fp32_kb - mem_int8_kb) / mem_fp32_kb) * 100.0

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "quant_step_ortalama_us": t_avg_us,
            "quant_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_kuantizasyon_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "mem_fp32_kb": mem_fp32_kb,
            "mem_int8_kb": mem_int8_kb,
            "tasarruf_yuzdesi": tasarruf_yuzde,
            "sqnr_db": err_metrics["sqnr_db"],
            "max_abs_err": err_metrics["max_abs_err"],
            "mae": err_metrics["mae"],
            "scale": scale,
            "sample_fp32": weights_fp32[:100],
            "sample_deq": deq_final[:100],
            "gecikmeler": gecikmeler_us[:200]
        }
