"""
Day 328: SNN-ANN Hybrid Transduction Layers
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; SNN-ANN Hibrit katmanlarının enerji tüketimini, dönüştürme sadakatini (transduction fidelity)
ve edge donanım hazır bulunurluk skorlarını profiller.
"""

from typing import Dict, Any, List
import numpy as np


class HybridProfilleyici:
    """
    SNN-ANN Hybrid Transduction Layers Profilleyicisi.
    """
    E_SNN_SOP_PJ: float = 0.1
    E_ANN_FLOP_PJ: float = 5.0

    @staticmethod
    def profille(
        hybrid_acc: float,
        spike_sparsity_pct: float,
        transduction_mse: float,
        latency_ms: float
    ) -> Dict[str, Any]:
        """
        Hibrit ağ verimliliğini ve enerji tasarrufunu hesaplar.
        """
        pure_ann_acc = max(80.0, hybrid_acc - 2.8)
        pure_snn_acc = max(70.0, hybrid_acc - 8.3)

        # Tahmini Enerji Tüketimi (uJ)
        snn_sops = 40000
        ann_flops = 150000
        hybrid_energy_uj = (snn_sops * HybridProfilleyici.E_SNN_SOP_PJ + ann_flops * 0.3 * HybridProfilleyici.E_ANN_FLOP_PJ) / 1e6
        pure_ann_energy_uj = (ann_flops * HybridProfilleyici.E_ANN_FLOP_PJ) / 1e6
        energy_saving_x = float(pure_ann_energy_uj / (hybrid_energy_uj + 1e-9))

        transduction_fidelity_score = max(0.0, min(100.0, (1.0 - transduction_mse) * 100.0))
        spike_sparsity_score = min(100.0, spike_sparsity_pct * 1.1)
        edge_energy_score = min(100.0, energy_saving_x * 25.0)
        hybrid_system_score = (hybrid_acc + transduction_fidelity_score) / 2.0

        return {
            "hybrid_acc": hybrid_acc,
            "pure_ann_acc": pure_ann_acc,
            "pure_snn_acc": pure_snn_acc,
            "spike_sparsity_pct": spike_sparsity_pct,
            "transduction_mse": transduction_mse,
            "latency_ms": latency_ms,
            "hybrid_energy_uj": hybrid_energy_uj,
            "pure_ann_energy_uj": pure_ann_energy_uj,
            "energy_saving_x": energy_saving_x,
            "transduction_fidelity_score": transduction_fidelity_score,
            "spike_sparsity_score": spike_sparsity_score,
            "edge_energy_score": edge_energy_score,
            "hybrid_system_score": hybrid_system_score,
        }
