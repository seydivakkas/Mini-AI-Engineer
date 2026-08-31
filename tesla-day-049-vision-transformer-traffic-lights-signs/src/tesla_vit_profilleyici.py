"""
Tesla ViT Profilleyici Modülü
=============================
Bu modül; Vision Transformer (ViT) yama gömme, öz-dikkat hesaplama ve
trafik ışığı/levhası sınıflandırma gecikmelerini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_vision_transformer_trafik_algilayici import TeslaVisionTransformerTrafficDetector


class TeslaViTProfilleyici:
    """
    Vision Transformer Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_vit_detector(self) -> Dict[str, Any]:
        vit = TeslaVisionTransformerTrafficDetector(img_size=64, patch_size=8, embed_dim=32, num_heads=4)
        sample_img = np.random.uniform(0, 255, (64, 64, 3)).astype(np.float32)

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = vit.forward_vit_traffic_detector(sample_img)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "vit_step_ortalama_us": t_avg_us,
            "vit_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_vit_karesi": int(1e6 / max(t_avg_us, 1e-4)),
            "tl_state": ciktilar["traffic_light_state"],
            "tl_conf": ciktilar["traffic_light_confidence"],
            "countdown_sec": ciktilar["countdown_seconds"],
            "sign_name": ciktilar["traffic_sign"],
            "sign_conf": ciktilar["traffic_sign_confidence"],
            "attn_matrix": ciktilar["attention_matrix"],
            "patch_count": ciktilar["patch_count"],
            "gecikmeler": gecikmeler_us[:200]
        }
