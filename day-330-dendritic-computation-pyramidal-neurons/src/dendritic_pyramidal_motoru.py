"""
Day 330: Dendritic Computation & Non-linear Pyramidal Branch Dynamics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Aktif Dendritik Dal NMDA Spike modelini, Çok Bölmeli (Multi-Compartment) Piramidal Nöron Mimarisini
ve Tek Nöron ile XOR Problemi Çözen Dendritik Sınıflandırıcıyı içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class DendriticBranch:
    """
    Aktif Nöron Dendritik Dal Modeli (NMDA Plateau Potential & Non-linear Integration).
    Girdi sinapslarını bölgesel olarak doğrusal olmayan eşikleme ile entegre eder.
    """
    def __init__(self, branch_id: int, num_synapses: int = 2, threshold: float = 1.2, plateau_gain: float = 2.5):
        self.branch_id = branch_id
        self.num_synapses = num_synapses
        self.threshold = threshold
        self.plateau_gain = plateau_gain
        self.weights = np.ones(num_synapses, dtype=np.float32)

    def compute_branch_potential(self, inputs: np.ndarray) -> Tuple[float, bool]:
        """
        Girdi: (Num_Synapses,) -> Çıktı: (Dendritik Dal Potansiyeli, NMDA Spike Var mı?)
        """
        linear_sum = float(np.dot(self.weights, inputs))
        
        # NMDA Doygunluk ve Plateau Potansiyeli (Sigmoidal non-linearity)
        if linear_sum >= self.threshold:
            # NMDA Dendritic Spike Ateşlendi
            is_nmda_spike = True
            v_branch = self.plateau_gain * (1.0 / (1.0 + np.exp(-2.0 * (linear_sum - self.threshold))))
        else:
            is_nmda_spike = False
            v_branch = linear_sum * 0.5

        return float(v_branch), is_nmda_spike


class MultiCompartmentPyramidalNeuron:
    """
    5-Bölmeli (5-Compartment) Piramidal Nöron Mimarisi.
    Bölmeler: Soma, Basal-1, Basal-2, Apical Trunk, Tuft
    """
    def __init__(self, v_rest: float = -70.0, v_th: float = -60.0, g_coupling: float = 0.7):
        self.v_rest = v_rest
        self.v_th = v_th
        self.g_coupling = g_coupling

        # Bölmelerin zarları [Soma, Basal1, Basal2, Apical, Tuft]
        self.v_compartments = np.full(5, v_rest, dtype=np.float32)
        
        self.basal1 = DendriticBranch(branch_id=1, num_synapses=2, threshold=1.2)
        self.basal2 = DendriticBranch(branch_id=2, num_synapses=2, threshold=1.2)

    def reset_neuron(self):
        """Zar potansiyellerini dinlenme durumuna sıfırlar."""
        self.v_compartments.fill(self.v_rest)

    def step_simulation(self, inputs_b1: np.ndarray, inputs_b2: np.ndarray) -> Tuple[float, bool, Dict[str, float]]:
        """
        Kablo teorisi adım simülasyonu. Soma aksiyon potansiyeli fırlatırsa (Soma Spike) True döner.
        """
        v_b1, spike_b1 = self.basal1.compute_branch_potential(inputs_b1)
        v_b2, spike_b2 = self.basal2.compute_branch_potential(inputs_b2)

        # Dendritik dal potansiyelleri zarlara aktarılır
        self.v_compartments[1] = self.v_rest + v_b1 * 10.0
        self.v_compartments[2] = self.v_rest + v_b2 * 10.0

        # Kablo Bağlantı Akımı: I_couple = g * (V_dend - V_soma)
        i_couple1 = self.g_coupling * (self.v_compartments[1] - self.v_compartments[0])
        i_couple2 = self.g_coupling * (self.v_compartments[2] - self.v_compartments[0])

        # Soma potansiyel güncellemesi
        self.v_compartments[0] += (i_couple1 + i_couple2) * 0.5

        is_soma_spike = False
        if self.v_compartments[0] >= self.v_th:
            is_soma_spike = True
            self.v_compartments[0] = self.v_rest  # Reset

        v_states = {
            "v_soma": float(self.v_compartments[0]),
            "v_basal1": float(self.v_compartments[1]),
            "v_basal2": float(self.v_compartments[2]),
        }
        return float(self.v_compartments[0]), is_soma_spike, v_states


class DendriticXORClassifier:
    """
    Tek Bir Piramidal Nöron ile Doğrusal Ayrıştırılamayan XOR Problemini Çözen Sınıflandırıcı.
    Point Nöronlar (LIF / McCulloch-Pitts) tek başlarına XOR çözemezken, 
    2 aktif dendritik dala sahip tek piramidal nöron XOR'u %100 başarıyla çözer!
    """
    def __init__(self):
        self.neuron = MultiCompartmentPyramidalNeuron(v_rest=-70.0, v_th=-60.0, g_coupling=0.7)

    def predict_xor(self, x1: float, x2: float) -> int:
        """
        XOR Girdisi: (x1, x2) in {(0,0), (0,1), (1,0), (1,1)}
        Çıktı: 0 veya 1
        """
        self.neuron.reset_neuron()
        not_x1 = 1.0 - x1
        not_x2 = 1.0 - x2

        inputs_b1 = np.array([x1, not_x2], dtype=np.float32)
        inputs_b2 = np.array([not_x1, x2], dtype=np.float32)

        # 5 Zaman Adımı Simülasyonu
        has_spiked = False
        for _ in range(5):
            _, is_spike, _ = self.neuron.step_simulation(inputs_b1, inputs_b2)
            if is_spike:
                has_spiked = True
                break

        return 1 if has_spiked else 0
