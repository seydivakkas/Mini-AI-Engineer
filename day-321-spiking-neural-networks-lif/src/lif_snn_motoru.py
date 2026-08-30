"""
Day 321: Spiking Neural Networks (SNN) & Leaky Integrate-and-Fire (LIF) Neuron Mathematics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Leaky Integrate-and-Fire (LIF) nöron diferansiyel denklemlerini,
türevlenebilir surrogate gradient (temsili gradyan) mekanizmasını,
Poisson spike kodlama katmanını ve modüler SNN mimarisini içerir.
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Sürekli türevlenemeyen Heaviside Adım Fonksiyonu S = H(V - V_th) için
    Fast Sigmoid türevli Surrogate Gradient (Temsili Gradyan) mekanizması.

    İleri Geçiş (Forward):
        S[t] = 1 (V >= V_th) veya 0 (V < V_th)

    Geri Geçiş (Backward):
        dS/dV = k / (1 + k * |V - V_th|)^2
    """
    slope: float = 25.0

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, v_th: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.v_th = v_th
        return (input_tensor >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        v_th = ctx.v_th
        slope = FastSigmoidSurrogate.slope
        
        # Fast Sigmoid türevi: dS/dV
        grad_input = grad_output.clone()
        surrogate_grad = slope / ((1.0 + slope * torch.abs(input_tensor - v_th)) ** 2)
        return grad_input * surrogate_grad, None


class PoissonEncoder(nn.Module):
    """
    Sürekli girdi verisini (ör. [0, 1] normalize piksel değerleri veya sinyaller)
    Poisson olasılık dağılımına uygun zaman adımlı Spike dizilerine (0/1) dönüştürür.
    """
    def __init__(self, time_steps: int = 50):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, Features)
        Çıktı: (Batch, Time_Steps, Features)
        """
        x_clamped = torch.clamp(x, 0.0, 1.0)
        # Her zaman adımı için bağımsız Bernoulli/Poisson çekimi
        shape = (x_clamped.shape[0], self.time_steps) + x_clamped.shape[1:]
        rand_tensor = torch.rand(shape, device=x.device, dtype=x.dtype)
        x_expanded = x_clamped.unsqueeze(1).expand(shape)
        spikes = (rand_tensor < x_expanded).float()
        return spikes


class LIFNeuronCell(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Nöron Hücresi.

    Matematiksel Dinamikler:
        tau_m * dV/dt = -(V - V_rest) + R_m * I(t)
        Ayrık Zaman Güncellemesi (Euler):
            V[t] = beta * V[t-1] + (1 - beta) * (V_rest + R_m * I[t])
        Ateşleme (Spike):
            S[t] = H(V[t] - V_th)
        Sıfırlama (Reset):
            V[t] <- V_reset (Soft or Hard Reset)
    """
    def __init__(
        self,
        num_neurons: int,
        beta: float = 0.85,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        refractory_period: int = 2,
        reset_mechanism: str = "zero",
        surrogate_slope: float = 25.0,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.beta = beta
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.refractory_period = refractory_period
        self.reset_mechanism = reset_mechanism
        FastSigmoidSurrogate.slope = surrogate_slope

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nöron durumunu başlatır: (Membrane Voltage V, Refractory Counter)
        """
        v_mem = torch.full((batch_size, self.num_neurons), self.v_rest, device=device)
        ref_cnt = torch.zeros((batch_size, self.num_neurons), dtype=torch.int32, device=device)
        return v_mem, ref_cnt

    def forward(
        self,
        input_current: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tek bir zaman adımı t için zar potansiyeli güncellemesi ve spike üretimi.
        """
        v_mem, ref_cnt = state
        
        # 1. Refrakter dönemdeki nöronların zar potansiyelini V_reset seviyesinde sabit tut
        is_refractory = (ref_cnt > 0)
        
        # 2. Sızıntılı İntegrasyon (Leaky Integration)
        # V[t] = beta * (V[t-1] - V_rest) + V_rest + I[t]
        v_decayed = self.beta * (v_mem - self.v_rest) + self.v_rest + input_current
        
        # Refrakter olan nöronların potansiyelini güncelleme
        v_next = torch.where(is_refractory, self.v_reset, v_decayed)
        
        # 3. Ateşleme (Spike Generation via Surrogate Gradient)
        spikes = FastSigmoidSurrogate.apply(v_next, self.v_threshold)
        
        # 4. Sıfırlama Mekanizması (Reset Mechanism)
        if self.reset_mechanism == "zero":
            # Hard reset: Potansiyel doğrudan V_reset'e çekilir
            v_post_spike = torch.where(spikes > 0.5, self.v_reset, v_next)
        else:
            # Soft reset: Potansiyel eşik kadar düşürülür
            v_post_spike = torch.where(spikes > 0.5, v_next - self.v_threshold, v_next)

        # 5. Refrakter Sayacı Güncellemesi
        new_ref_cnt = torch.where(
            spikes > 0.5,
            torch.tensor(self.refractory_period, device=input_current.device, dtype=torch.int32),
            torch.clamp(ref_cnt - 1, min=0)
        )
        
        return spikes, (v_post_spike, new_ref_cnt)


class LIFSpikingLayer(nn.Module):
    """
    Ağırlık Matrisi (Synaptic Weights) + LIF Nöron Hücresini Birleştiren Tam Katman.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.85,
        v_threshold: float = 1.0,
        refractory_period: int = 2,
        reset_mechanism: str = "zero",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.synapse = nn.Linear(in_features, out_features)
        self.lif_cell = LIFNeuronCell(
            num_neurons=out_features,
            beta=beta,
            v_threshold=v_threshold,
            refractory_period=refractory_period,
            reset_mechanism=reset_mechanism
        )

    def forward(self, input_spikes_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Girdi: (Batch, Time_Steps, In_Features)
        Çıktı: 
            spikes_seq: (Batch, Time_Steps, Out_Features)
            mem_seq: (Batch, Time_Steps, Out_Features)
        """
        batch_size, time_steps, _ = input_spikes_seq.shape
        device = input_spikes_seq.device
        
        state = self.lif_cell.init_state(batch_size, device)
        
        spikes_list = []
        mem_list = []
        
        for t in range(time_steps):
            x_t = input_spikes_seq[:, t, :]
            i_t = self.synapse(x_t)  # Sinaptik Akım I(t) = W * S(t) + b
            spikes_t, state = self.lif_cell(i_t, state)
            
            spikes_list.append(spikes_t)
            mem_list.append(state[0])  # Mempotansiyeli V(t)
            
        spikes_seq = torch.stack(spikes_list, dim=1)
        mem_seq = torch.stack(mem_list, dim=1)
        return spikes_seq, mem_seq


class SNNClassifier(nn.Module):
    """
    Çok Katmanlı Spiking Sinir Ağı Sınıflandırıcısı (Multi-Layer SNN Classifier).
    
    Mimari:
        Poisson Encoder -> LIF Layer 1 -> LIF Layer 2 -> Rate-based Readout
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        time_steps: int = 50,
        beta: float = 0.85,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.encoder = PoissonEncoder(time_steps=time_steps)
        self.layer1 = LIFSpikingLayer(in_features, hidden_features, beta=beta, v_threshold=v_threshold)
        self.layer2 = LIFSpikingLayer(hidden_features, num_classes, beta=beta, v_threshold=v_threshold)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Girdi: (Batch, Features)
        Çıktı: 
            logits: (Batch, Num_Classes)
            info_dict: Zar potansiyelleri, spike dizileri ve seyreklik istatistikleri.
        """
        input_spikes = self.encoder(x)  # (Batch, T, In_Features)
        
        spikes1, mem1 = self.layer1(input_spikes)  # (Batch, T, Hidden)
        spikes2, mem2 = self.layer2(spikes1)        # (Batch, T, Out_Classes)
        
        # Readout: Zaman adımları boyunca ortalama spike sıklığı (Rate Coding)
        logits = torch.mean(spikes2, dim=1)  # (Batch, Num_Classes)
        
        # Seyreklik & Aktivasyon İstatistikleri
        sparsity_l1 = torch.mean(spikes1).item()
        sparsity_l2 = torch.mean(spikes2).item()
        
        info_dict = {
            "input_spikes": input_spikes,
            "spikes1": spikes1,
            "mem1": mem1,
            "spikes2": spikes2,
            "mem2": mem2,
            "sparsity_layer1": sparsity_l1,
            "sparsity_layer2": sparsity_l2,
        }
        return logits, info_dict
