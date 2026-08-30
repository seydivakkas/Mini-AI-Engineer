"""
Day 324: Neuromorphic Hardware Mapping (Intel Loihi 2 & SynSense)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Intel Loihi 2 ve SynSense nöromorfik donanımları için Neuro-Core matris haritalamasını,
INT8 sabitleştirilmiş (fixed-point) ağırlık kuantizasyonunu, AER (Address Event Representation)
paket yönlendirmesini ve donanım eşleme simülatörünü içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AERPacket:
    """
    Address Event Representation (AER) Donanım Yönlendirme Paketi.
    """
    src_core: int
    dst_core: int
    neuron_id: int
    timestamp_us: float
    hop_distance: int


class LoihiNeuroCore:
    """
    Intel Loihi 2 / SynSense Nöromorfik Donanım Çekirdeği (Neuro-Core).

    Donanım Özellikleri:
        - Sabit Noktalı (Fixed-Point) INT8 Ağırlıklar: W_int in [-128, 127]
        - Sabit Noktalı (Fixed-Point) INT16 Zar Potansiyeli: V_int in [-32768, 32767]
        - Maksimum Nöron Kapasitesi: max_neurons (ör. 256)
        - Maksimum Sinaps Kapasitesi: max_synapses
    """
    def __init__(
        self,
        core_id: int,
        grid_x: int,
        grid_y: int,
        max_neurons: int = 256,
        max_synapses: int = 65536,
        weight_bits: int = 8
    ):
        self.core_id = core_id
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.max_neurons = max_neurons
        self.max_synapses = max_synapses
        self.weight_bits = weight_bits

        self.assigned_neurons: List[int] = []
        self.weights_int8: Optional[np.ndarray] = None
        self.scale_factor: float = 1.0

    def is_full(self, count: int = 1) -> bool:
        return (len(self.assigned_neurons) + count) > self.max_neurons

    def load_quantized_weights(self, weights_fp32: np.ndarray) -> float:
        """
        FP32 sinaptik ağırlıkları simetrik INT8 sabitleştirilmiş formata kuantize eder.
        W_int = clamp(round(W_fp32 * scale), -128, 127)
        """
        max_abs = np.max(np.abs(weights_fp32)) + 1e-9
        max_int = (2 ** (self.weight_bits - 1)) - 1  # 127 for 8-bit
        
        self.scale_factor = max_int / max_abs
        quantized = np.round(weights_fp32 * self.scale_factor)
        self.weights_int8 = np.clip(quantized, -max_int - 1, max_int).astype(np.int8)
        return self.scale_factor

    def dequantize_weights(self) -> np.ndarray:
        """
        INT8 ağırlıkları test doğrulaması için tekrar FP32 formatına döndürür.
        """
        if self.weights_int8 is None:
            return np.array([])
        return self.weights_int8.astype(np.float32) / (self.scale_factor + 1e-9)


class AERPacketRouter:
    """
    Address Event Representation (AER) Çekirdekler Arası Yönlendirme Simülatörü.
    """
    @staticmethod
    def calculate_manhattan_distance(core_a: LoihiNeuroCore, core_b: LoihiNeuroCore) -> int:
        """
        İki Neuro-Core arasındaki Manhattan Ağ Yönlendirme Mesafesini (Hop) hesaplar.
        Hop = |x1 - x2| + |y1 - y2|
        """
        return abs(core_a.grid_x - core_b.grid_x) + abs(core_a.grid_y - core_b.grid_y)

    @staticmethod
    def route_spike(
        src_core: LoihiNeuroCore,
        dst_core: LoihiNeuroCore,
        neuron_id: int,
        timestamp_us: float
    ) -> AERPacket:
        """
        AER Spike Paketini oluşturur ve yönlendirme gecikmesini hesaplar.
        """
        hops = AERPacketRouter.calculate_manhattan_distance(src_core, dst_core)
        return AERPacket(
            src_core=src_core.core_id,
            dst_core=dst_core.core_id,
            neuron_id=neuron_id,
            timestamp_us=timestamp_us,
            hop_distance=hops
        )


class NeuromorphicHardwareMapper:
    """
    PyTorch SNN Modelini Intel Loihi 2 / SynSense Mesh Çip Mimarisine Haritalayıcı.

    Haritalama Adımları:
        1. Katman Bölümleme (Layer Partitioning): Katmandaki nöron sayısı core limitini aşarsa bölünür.
        2. INT8 Sabitleştirilmiş Kuantizasyon.
        3. Mesh Ağı Yerleşimi ($M \times N$ Grid).
        4. Donanım Çıkarım Simülasyonu.
    """
    def __init__(
        self,
        mesh_rows: int = 4,
        mesh_cols: int = 4,
        max_neurons_per_core: int = 64
    ):
        self.mesh_rows = mesh_rows
        self.mesh_cols = mesh_cols
        self.max_neurons_per_core = max_neurons_per_core
        self.total_cores = mesh_rows * mesh_cols

        self.cores: List[LoihiNeuroCore] = []
        core_id = 0
        for r in range(mesh_rows):
            for c in range(mesh_cols):
                self.cores.append(LoihiNeuroCore(core_id=core_id, grid_x=c, grid_y=r, max_neurons=max_neurons_per_core))
                core_id += 1

    def map_snn_weights(self, weights_fp32: np.ndarray) -> Dict[str, Any]:
        """
        FP32 Sinaptik Ağırlık Matrisini (Out, In) donanım çekirdeklerine böler ve kuantize eder.
        """
        out_features, in_features = weights_fp32.shape
        
        # Kaç çekirdeğe bölüneceğini hesapla
        needed_cores = math.ceil(out_features / self.max_neurons_per_core)
        if needed_cores > self.total_cores:
            raise ValueError(f"Model {needed_cores} çekirdek gerektiriyor, mevcut donanımda ise sadece {self.total_cores} çekirdek var!")

        mapped_info = []
        current_core_idx = 0

        for i in range(0, out_features, self.max_neurons_per_core):
            chunk_w = weights_fp32[i : i + self.max_neurons_per_core, :]
            core = self.cores[current_core_idx]
            
            chunk_neurons = list(range(i, min(i + self.max_neurons_per_core, out_features)))
            core.assigned_neurons = chunk_neurons
            scale = core.load_quantized_weights(chunk_w)

            mapped_info.append({
                "core_id": core.core_id,
                "grid_pos": (core.grid_x, core.grid_y),
                "num_neurons": len(chunk_neurons),
                "scale_factor": scale,
            })
            current_core_idx += 1

        # Donanım Doluluk Oranı
        used_cores = current_core_idx
        core_utilization = (used_cores / self.total_cores) * 100.0

        # Kuantizasyon Gürültü Oranı (SQNR)
        dequantized_w = np.vstack([c.dequantize_weights() for c in self.cores[:used_cores]])
        signal_power = np.mean(weights_fp32 ** 2)
        noise_power = np.mean((weights_fp32 - dequantized_w) ** 2) + 1e-9
        sqnr_db = float(10.0 * np.log10(signal_power / noise_power))

        return {
            "used_cores": used_cores,
            "total_cores": self.total_cores,
            "core_utilization_pct": core_utilization,
            "mapped_cores_info": mapped_info,
            "sqnr_db": sqnr_db,
            "dequantized_weights": dequantized_w,
        }
