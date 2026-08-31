"""
Tesla Damıtma Profilleyici Modülü
=================================
Bu modül; Bilgi Damıtma (Knowledge Distillation) kaybını, L1-Norm kanal
budama hızını ve FLOPs tasarruf oranını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_model_damitma_ve_budama import TeslaKnowledgeDistiller


class TeslaDamitmaProfilleyici:
    """
    Model Damıtma ve Budama Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_distillation_and_pruning(self) -> Dict[str, Any]:
        distiller = TeslaKnowledgeDistiller(temperature=4.0, alpha=0.7)

        # Devasa Öğretmen ve Kompakt Öğrenci Logitleri (10 Sınıf)
        np.random.seed(42)
        teacher_logits = np.array([8.5, 3.2, 1.1, 0.2, 0.1, -0.5, 0.0, 1.2, 0.4, 0.1])
        student_logits = np.array([5.1, 2.8, 1.3, 0.5, 0.2, -0.1, 0.1, 0.9, 0.3, 0.0])
        true_labels = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 64 Kanallı Conv2D Ağırlık Tensörü: (64, 32, 3, 3)
        conv_weights = np.random.normal(0, 0.1, (64, 32, 3, 3)).astype(np.float32)

        gecikmeler_us: List[float] = []
        loss_dict = None
        pruned_w = None
        mask = None
        sparsity = 0.0

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            loss_dict = distiller.compute_distillation_loss(teacher_logits, student_logits, true_labels)
            pruned_w, mask, sparsity = distiller.prune_channels_l1_norm(conv_weights, prune_ratio=0.3)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        p_T = distiller.compute_soft_probabilities(teacher_logits, 4.0)
        p_S = distiller.compute_soft_probabilities(student_logits, 4.0)

        return {
            "distill_step_ortalama_us": t_avg_us,
            "distill_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_damitma_adimi": int(1e6 / max(t_avg_us, 1e-4)),
            "total_loss": loss_dict["total_loss"],
            "loss_soft_kd": loss_dict["loss_soft_kd"],
            "loss_hard_ce": loss_dict["loss_hard_ce"],
            "kl_div": loss_dict["kl_divergence"],
            "sparsity_pct": sparsity * 100.0,
            "pruned_channels_count": int(np.sum(~mask)),
            "active_channels_count": int(np.sum(mask)),
            "teacher_probs": p_T,
            "student_probs": p_S,
            "gecikmeler": gecikmeler_us[:200]
        }
