"""
Tesla Dağıtık Eğitim Profilleyici Modülü
========================================
Bu modül; FP8 kuantalama, gradyan kırpma ve FSDP bellek dağıtım hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_dagitik_egitim_motoru import TeslaDojoDistributedTrainer


class TeslaEgitimProfilleyici:
    """
    Tesla Dojo Dağıtık Eğitim Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 50):
        self.iterations = iterations

    def benchmark_distributed_training(self) -> Dict[str, Any]:
        trainer = TeslaDojoDistributedTrainer(num_devices=8)

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            t_inst = TeslaDojoDistributedTrainer()
            t0 = time.perf_counter_ns()
            _ = t_inst.train_step_fsdp_fp8(hidden_dim=256)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        sim_res = trainer.train_step_fsdp_fp8(hidden_dim=512)

        # 30 adımlık eğitim kaybı eğrisi simülasyonu
        loss_curve = [0.85 * np.exp(-i / 10.0) + 0.12 + float(np.random.randn() * 0.01) for i in range(30)]

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "num_devices": trainer.num_devices,
            "fp32_mem_mb": sim_res["fp32_memory_mb"],
            "sharded_mem_mb": sim_res["sharded_memory_per_gpu_mb"],
            "mem_reduction": sim_res["memory_reduction_factor"],
            "initial_norm": sim_res["initial_grad_norm"],
            "clipped_norm": sim_res["clipped_grad_norm"],
            "final_loss": sim_res["training_loss"],
            "step_ortalama_us": t_avg_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_egitim_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "loss_curve": loss_curve,
            "gecikmeler": gecikmeler_us[:200]
        }
