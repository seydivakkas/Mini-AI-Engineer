"""
Day 332: Optogenetic Stimulus Pattern Synthesis & Generative Inversion
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; ChR2 Opsin Fotoakım Kinetiğini, Nöral Dokuda Işık Uyarım Populasyonunu
ve Hedef Nöral Akıştan İdeal Işık Desenini Sentezleyen Üretken İnversiyon Çözücüsünü içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ChR2OpsinModel:
    """
    Channelrhodopsin-2 (ChR2) Fotoakım Kinetiği Modeli.
    Mavi ışık uyarımını (470nm, mW/mm^2) opsin fotoakımına (I_ChR2) dönüştürür.
    """
    def __init__(self, g_max: float = 0.4, i_sat: float = 2.0, e_chr2: float = 0.0):
        self.g_max = g_max
        self.i_sat = i_sat
        self.e_chr2 = e_chr2

    def compute_photocurrent(self, light_irradiance: float, v_membrane: float = -70.0) -> float:
        """
        Girdi: Işık Şiddeti (mW/mm^2) -> Çıktı: Fotoakım (pA)
        """
        if light_irradiance <= 0.0:
            return 0.0
        
        # Opsin açık kanal oranı O(I) = I / (I + I_sat)
        open_fraction = light_irradiance / (light_irradiance + self.i_sat)
        i_chr2 = self.g_max * open_fraction * (v_membrane - self.e_chr2)
        return float(i_chr2)


class OptogeneticNeuralPopulation:
    """
    ChR2 Opsin İfada Eden 25 Nöronluk Nöromorfik Doku Katmanı.
    """
    def __init__(self, num_neurons: int = 25, v_rest: float = -70.0, v_th: float = -50.0):
        self.num_neurons = num_neurons
        self.v_rest = v_rest
        self.v_th = v_th
        self.opsin = ChR2OpsinModel()
        self.v_mem = np.full(num_neurons, v_rest, dtype=np.float32)

    def simulate_step(self, light_pattern: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Girdi: (Num_Neurons,) Işık Şiddeti Dizisi -> Çıktı: (Spike Vektörü, Zar Potansiyelleri)
        """
        spikes = np.zeros(self.num_neurons, dtype=np.float32)
        
        for i in range(self.num_neurons):
            i_photo = self.opsin.compute_photocurrent(float(light_pattern[i]), self.v_mem[i])
            # Potansiyel güncellemesi (İnward depolarizing photocurrent)
            self.v_mem[i] += (- (self.v_mem[i] - self.v_rest) * 0.1 - i_photo * 0.25)
            
            if self.v_mem[i] >= self.v_th:
                spikes[i] = 1.0
                self.v_mem[i] = self.v_rest  # Reset

        return spikes, self.v_mem.copy()


class OptogeneticGenerativeInverter(nn.Module):
    """
    Türevlenebilir PyTorch Üretken İnversiyon Modeli.
    Hedef Nöral Spike Matrisinden (Target Spike Raster) En Uygun Mekansal Işık Desenini Sentezler.
    """
    def __init__(self, num_neurons: int = 25, time_steps: int = 20):
        super().__init__()
        self.num_neurons = num_neurons
        self.time_steps = time_steps
        
        # Optimize edilecek Işık Desen Parametresi (mW/mm^2)
        self.light_pattern_param = nn.Parameter(torch.rand(num_neurons, time_steps) * 1.5)

    def forward(self) -> torch.Tensor:
        """
        Işık Deseninden Nöral Aktivite Tahmini (Differentiable Surrogate Forward Model)
        """
        # Pozitif Işık Şiddeti (Softplus kısıtı)
        light_intensity = torch.nn.functional.softplus(self.light_pattern_param)
        
        # Aktivasyon tahmini: Işık şiddeti eşiği aştığında türevlenebilir sigmoid aktivitasyon
        predicted_activity = torch.sigmoid(4.0 * (light_intensity - 1.2))
        return predicted_activity

    def sentezle_isik_deseni(
        self,
        target_spike_raster: torch.Tensor,
        num_epochs: int = 30,
        lr: float = 0.05
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Hedef Spike Matrisine Göre Optimum Işık Desenini İnversiyon Yoluyla Sentezler.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        loss_history = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            pred_activity = self.forward()
            
            # Kayıp: MSE + L2 Düzenleme (Fototoksisiteyi önlemek için minimum ışık)
            l2_reg = 0.01 * torch.mean(self.light_pattern_param ** 2)
            loss = criterion(pred_activity, target_spike_raster) + l2_reg
            
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

        optimal_light = torch.nn.functional.softplus(self.light_pattern_param).detach().cpu().numpy()
        return optimal_light, loss_history
