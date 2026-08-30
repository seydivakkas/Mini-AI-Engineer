"""
Day 321: Spiking Neural Networks (SNN) & Leaky Integrate-and-Fire (LIF) Neuron Mathematics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; SNN modelinin Sinaptik Operasyon Sayılarını (SOPs), spike seyreklik oranlarını (sparsity)
ve tahmini picoJoule donanım enerji tüketimini profillemek için kullanılır.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class SNNProfilleyici:
    """
    SNN Profilleyici ve Enerji/Operasyon Metrik Analizörü.

    Enerji Sabitleri (45nm CMOS / Neuromorphic Hardware):
        - SNN Synaptic Operation (Accumulate / Add): E_sop = 0.9 pJ
        - ANN Multiply-Accumulate (MAC / FLOP): E_mac = 4.6 pJ
    """
    E_SOP_PJ: float = 0.9  # PicoJoules per Accumulate Operation
    E_MAC_PJ: float = 4.6  # PicoJoules per MAC Operation

    @staticmethod
    def profille(
        model: nn.Module,
        info_dict: Dict[str, Any],
        time_steps: int
    ) -> Dict[str, Any]:
        """
        Modelin bir batch çıkarımındaki SOP, FLOP, seyreklik ve enerji metriklerini hesaplar.
        """
        input_spikes = info_dict["input_spikes"]  # (Batch, T, In)
        spikes1 = info_dict["spikes1"]            # (Batch, T, Hidden)
        spikes2 = info_dict["spikes2"]            # (Batch, T, Out)

        batch_size = input_spikes.shape[0]
        in_features = input_spikes.shape[2]
        hidden_features = spikes1.shape[2]
        out_features = spikes2.shape[2]

        # 1. Katman 1 Sinaptik Operasyonlar (SOPs_1)
        # Katman 1'e giren her 1-spike için hidden_features kadar toplama işlemi gerçekleşir
        sop_l1 = torch.sum(input_spikes).item() * hidden_features
        
        # 2. Katman 2 Sinaptik Operasyonlar (SOPs_2)
        sop_l2 = torch.sum(spikes1).item() * out_features
        
        total_sops = sop_l1 + sop_l2
        sops_per_sample = total_sops / batch_size

        # Karşılaştırma için Standart ANN FLOP Hesabı (Dense MACs)
        # ANN her zaman adımında tüm nöronlar için çarpım-toplam yapar
        ann_macs_l1 = batch_size * time_steps * in_features * hidden_features
        ann_macs_l2 = batch_size * time_steps * hidden_features * out_features
        total_ann_macs = ann_macs_l1 + ann_macs_l2
        ann_macs_per_sample = total_ann_macs / batch_size

        # Tahmini Enerji Tüketimleri (picoJoules / Örnek)
        snn_energy_pj = sops_per_sample * SNNProfilleyici.E_SOP_PJ
        ann_energy_pj = ann_macs_per_sample * SNNProfilleyici.E_MAC_PJ

        # Enerji Tasarrufu Oranı (Energy Efficiency Gain)
        energy_gain = (ann_energy_pj / (snn_energy_pj + 1e-9)) if snn_energy_pj > 0 else 1.0

        # Seyreklik Metrikleri
        total_possible_spikes = batch_size * time_steps * (in_features + hidden_features + out_features)
        total_actual_spikes = torch.sum(input_spikes).item() + torch.sum(spikes1).item() + torch.sum(spikes2).item()
        global_sparsity = 1.0 - (total_actual_spikes / total_possible_spikes)

        return {
            "batch_size": batch_size,
            "time_steps": time_steps,
            "total_sops": total_sops,
            "sops_per_sample": sops_per_sample,
            "ann_macs_per_sample": ann_macs_per_sample,
            "snn_energy_pj": snn_energy_pj,
            "ann_energy_pj": ann_energy_pj,
            "energy_gain_x": energy_gain,
            "global_sparsity": global_sparsity,
            "layer1_sparsity": info_dict.get("sparsity_layer1", 0.0),
            "layer2_sparsity": info_dict.get("sparsity_layer2", 0.0),
        }
